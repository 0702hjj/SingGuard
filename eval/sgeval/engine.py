"""Inference backends.

vllm : spawn a local `vllm serve` subprocess (OpenAI-compatible), hammer it with async
       chat completions, then tear it down. Used for all Qwen-VL-architecture guards.
hf   : in-process transformers fallback for models vLLM cannot load. Batched, slower.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import mimetypes
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from openai import AsyncOpenAI

log = logging.getLogger("sgeval")

_HF_CACHE: dict = {}   # model_path -> (processor, model), reused across datasets


def _cfg_tag(adapter, gen_args: dict, chat_template_kwargs: dict) -> str:
    """Fingerprint of the inference configuration a record was produced under. Resume logic
    compares this so predictions from a different mode/max_tokens are never silently reused
    (a real incident: fast-slow predictions were re-labelled as fast-mode results)."""
    mode = (chat_template_kwargs or {}).get("thinking_type", "")
    return f"{getattr(adapter, 'name', 'adapter')}|{mode}|{gen_args.get('max_tokens', '')}"


class ServerDeadError(RuntimeError):
    """Raised when >50% of a dataset's requests failed with connection errors,
    meaning the vLLM engine died (e.g. CUDA OOM). The runner aborts this model."""

_MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
         ".webp": "image/webp", ".gif": "image/gif", ".bmp": "image/bmp"}


def image_to_data_url(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"image not found: {path}")
    mime = _MIME.get(p.suffix.lower(), mimetypes.guess_type(str(p))[0] or "image/jpeg")
    b64 = base64.b64encode(p.read_bytes()).decode()
    return f"data:{mime};base64,{b64}"


class VLLMServer:
    """Manage one `vllm serve` subprocess per model."""

    def __init__(self, model_path: str, port: int = 8199, max_model_len: int = 8192,
                 gpu_memory_utilization: float = 0.90, log_file: str | None = None,
                 timeout_s: int = 2400, max_num_seqs: int = 32):
        self.model_path = model_path
        self.port = port
        self.max_model_len = max_model_len
        self.gpu_mem = gpu_memory_utilization
        self.timeout_s = timeout_s
        self.max_num_seqs = max_num_seqs
        self.log_file = log_file
        self.proc: subprocess.Popen | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def start(self):
        # Port pre-flight: a leftover server from a killed run would answer /health and
        # silently serve the *previous* model, poisoning this model's results.
        import urllib.error
        import urllib.request
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=3):
                raise RuntimeError(
                    f"port {self.port} already serves a live vLLM -- a leftover server from "
                    f"a previous run must be killed first (refusing to reuse it)")
        except urllib.error.URLError:
            pass  # nothing listening: good
        vllm_bin = shutil.which("vllm") or shutil.which("vllm.exe")
        if vllm_bin:
            cmd = [vllm_bin, "serve", self.model_path]
        else:
            cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
                   "--model", self.model_path]
        cmd += [
            "--served-model-name", "guard",
            "--port", str(self.port),
            "--dtype", "bfloat16",
            "--max-model-len", str(self.max_model_len),
            "--gpu-memory-utilization", str(self.gpu_mem),
            "--max-num-seqs", str(self.max_num_seqs),   # cap concurrent seqs: mm-heavy prefill OOMs on shared GPUs
            "--trust-remote-code",
            # vLLM 0.11's multimodal preprocessor cache can assert mid-run
            # ("Expected a cached item for mm_hash=...") and kill the engine; the cache only
            # helps with repeated identical images, which our benchmarks barely have.
            "--disable-mm-preprocessor-cache",
        ]
        # SingGuard-style models ship the guard prompt as a standalone chat_template.jinja
        # (tokenizer_config.chat_template is empty). Older vLLM only reads the string field
        # and silently falls back to the base model's template -- without the risk-category
        # system prompt. Always point vLLM at the jinja file when present.
        jinja = Path(self.model_path) / "chat_template.jinja"
        if jinja.exists():
            cmd += ["--chat-template", str(jinja)]
        log.info("starting vLLM: %s", " ".join(cmd))
        logf = open(self.log_file, "ab") if self.log_file else subprocess.DEVNULL
        # own process group so stop() can reap the whole tree (vllm renames engine
        # children to "VLLM::EngineCore", which defeats name-based pkill)
        self.proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                                     start_new_session=True)

        import urllib.request
        deadline = time.time() + self.timeout_s
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"vLLM exited with code {self.proc.returncode}; see {self.log_file}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=5):
                    log.info("vLLM healthy on port %s", self.port)
                    return
            except Exception:
                time.sleep(10)
        raise TimeoutError(f"vLLM not healthy after {self.timeout_s}s; see {self.log_file}")

    def stop(self):
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)   # whole group: api_server + EngineCore children
            except (ProcessLookupError, PermissionError):
                self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    self.proc.kill()
        self.proc = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()


async def run_dataset_vllm(server: VLLMServer, samples: list[dict], adapter, gen_args: dict,
                           chat_template_kwargs: dict, data_dir: Path, concurrency: int = 32,
                           retries: int = 3, progress_desc: str = "") -> list[dict]:
    """Evaluate one dataset against a live server. Returns prediction records."""
    client = AsyncOpenAI(base_url=server.base_url, api_key="EMPTY", timeout=600, max_retries=2)
    sem = asyncio.Semaphore(concurrency)
    url_cache: dict[str, str] = {}

    def urls_for(s: dict) -> list[str]:
        urls = []
        v = s.get("image")
        if isinstance(v, str):
            v = [v] if v else []
        for rel in (v or []):            # a sample may carry several images (e.g. MMDS)
            if rel not in url_cache:
                url_cache[rel] = image_to_data_url(str(data_dir / rel))
            urls.append(url_cache[rel])
        return urls

    from tqdm.asyncio import tqdm_asyncio

    async def one(s: dict) -> dict:
        try:
            msgs = adapter.messages(s, urls_for(s))
        except Exception as e:  # noqa: BLE001  (e.g. image file missing)
            log.error("input build failed for %s: %s", s["id"], e)
            # <INPUT_ERROR> is permanent (no retry) and must NOT count toward the
            # engine-death ratio -- one missing image would otherwise abort the model
            return {"id": s["id"], "gold": s["label"], "pred": None,
                    "raw": f"<INPUT_ERROR {e}>"[:400],
                    "cfg": _cfg_tag(adapter, gen_args, chat_template_kwargs)}
        kw = dict(chat_template_kwargs or {})
        try:
            extra_kw = adapter.template_kwargs(s)
            if extra_kw:
                kw.update(extra_kw)
        except Exception as e:  # noqa: BLE001
            log.warning("template_kwargs failed for %s: %s", s["id"], e)
        extra = {"chat_template_kwargs": kw} if kw else {}
        for attempt in range(retries):
            try:
                async with sem:   # actually throttle client-side (was previously a no-op)
                    r = await client.chat.completions.create(
                        model="guard", messages=msgs, temperature=gen_args.get("temperature", 0.0),
                        max_tokens=gen_args.get("max_tokens", 256), extra_body=extra)
                text = r.choices[0].message.content or ""
                pred = adapter.parse(text, sample=s)
                # store head + tail: the trailing <answer> lands in the tail, so audits can
                # later tell truncation / provisional-vs-final apart (head-only lost it)
                raw = text[:300] + ("…" + text[-250:] if len(text) > 550 else text[300:])
                return {"id": s["id"], "gold": s["label"], "pred": pred, "raw": raw,
                        "cfg": _cfg_tag(adapter, gen_args, chat_template_kwargs),
                        "finish_reason": getattr(r.choices[0], "finish_reason", None)}
            except Exception as e:  # noqa: BLE001
                if attempt == retries - 1:
                    log.error("request failed for %s: %s", s["id"], e)
                    return {"id": s["id"], "gold": s["label"], "pred": None,
                            "raw": f"<ERROR {e}>"[:400],
                            "cfg": _cfg_tag(adapter, gen_args, chat_template_kwargs)}
                await asyncio.sleep(3 * (attempt + 1))

    tasks = [one(s) for s in samples]
    records = await tqdm_asyncio.gather(*tasks, desc=progress_desc)
    return records


def run_dataset_hf(model_path: str, samples: list[dict], adapter, gen_args: dict,
                   chat_template_kwargs: dict, data_dir: Path, batch_size: int = 4,
                   progress_desc: str = "") -> list[dict]:
    """Transformers fallback: batched generate with the model's own chat template.

    The (model, processor) pair is cached across datasets so a multi-dataset run loads
    weights once. Note: this backend renders the plain (query [+ response]) conversation
    through the model's chat template, ignoring adapter system prompts -- it exists so
    exotic architectures can still run; the vLLM backend is the reference path.
    """
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    global _HF_CACHE
    if model_path not in _HF_CACHE:
        log.info("loading %s with transformers ...", model_path)
        _HF_CACHE[model_path] = (
            AutoProcessor.from_pretrained(model_path, trust_remote_code=True),
            AutoModelForImageTextToText.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, device_map="auto",
                trust_remote_code=True).eval())
    processor, model = _HF_CACHE[model_path]

    def pil(path):
        return Image.open(data_dir / path).convert("RGB")

    def render(s: dict) -> tuple[str, list]:
        msgs = [{"role": "user", "content": [{"type": "text", "text": s["query"]}]}]
        if s.get("image"):
            msgs[0]["content"] = [{"type": "image"}] + msgs[0]["content"]
        if s.get("response"):
            msgs.append({"role": "assistant",
                         "content": [{"type": "text", "text": s["response"]}]})
        # chat_template_kwargs must be passed as a dict argument -- top-level kwargs like
        # thinking_type are silently ignored by processor.apply_chat_template.
        prompt = processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            chat_template_kwargs=dict(chat_template_kwargs or {}))
        images = [pil(s["image"])] if s.get("image") else []
        return prompt, images

    # Keep image / text-only samples in separate batches so processor() never sees mixed lists.
    out, order = [], []
    groups = ([s for s in samples if s.get("image")], [s for s in samples if not s.get("image")])
    from tqdm import tqdm
    for group in groups:
        for i in tqdm(range(0, len(group), batch_size),
                      desc=progress_desc, disable=not group):
            batch = group[i:i + batch_size]
            rendered = [render(s) for s in batch]
            inputs = processor(
                text=[r[0] for r in rendered],
                images=[r[1][0] for r in rendered] if batch and batch[0].get("image") else None,
                return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                gen = model.generate(**inputs, do_sample=False,
                                     max_new_tokens=gen_args.get("max_tokens", 256))
            trimmed = [o[len(inp):] for inp, o in zip(inputs.input_ids, gen)]
            texts = processor.batch_decode(trimmed, skip_special_tokens=True)
            for s, t in zip(batch, texts):
                out.append({"id": s["id"], "gold": s["label"],
                            "pred": adapter.parse(t, sample=s),
                            "raw": t[:400]})
            order.extend(s["id"] for s in batch)

    by_id = {r["id"]: r for r in out}
    return [by_id[i] for i in order]


def finalize_records(records: list[dict]) -> tuple[list[int], list[int], int]:
    """Apply the paper's failure policy: unparsable => counted as incorrect (pred flipped)."""
    golds, preds = [], []
    unparsable = 0
    for r in records:
        golds.append(r["gold"])
        if r["pred"] is None:
            unparsable += 1
            preds.append(1 - r["gold"])
        else:
            preds.append(r["pred"])
    return golds, preds, unparsable
