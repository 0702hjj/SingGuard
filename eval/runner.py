#!/usr/bin/env python
"""Table-4 evaluation runner.

  uv run python runner.py --models sing-guard-8b                          # one model, all datasets
  uv run python runner.py --models sing-guard-8b --datasets vlguard --smoke 5
  uv run python runner.py --models all                                   # enabled sweep
  uv run python runner.py --models llavaguard-7b --engine hf             # transformers fallback

Behavior:
- vLLM backend: one server per model, async requests (concurrency), auto start/stop.
- Resume: predictions are appended to results/preds/<model>__<dataset>.jsonl; already-done
  sample ids are skipped, so re-running continues where it stopped.
- Results are appended to results/results.csv after each (model, dataset).
- Parse failures follow the paper: counted as incorrect, tracked separately.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import sys
import time
from pathlib import Path

import yaml

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))

from sgeval.adapters import get_adapter
from sgeval.engine import (ServerDeadError, VLLMServer, finalize_records,
                           run_dataset_hf, run_dataset_vllm)
from sgeval.metrics import prf

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("runner")

RESULTS = EVAL_DIR / "results"
PREDS = RESULTS / "preds"
LOGS = EVAL_DIR / "logs"


def load_yaml(name: str) -> dict:
    return yaml.safe_load((EVAL_DIR / "configs" / name).read_text())


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.open() if l.strip()]


def append_result(row: dict) -> None:
    RESULTS.mkdir(exist_ok=True)
    exists = (RESULTS / "results.csv").exists()
    with (RESULTS / "results.csv").open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            w.writeheader()
        w.writerow(row)


def eval_dataset(model_key: str, mcfg: dict, ds_key: str, samples: list[dict], args) -> dict:
    adapter = get_adapter(mcfg["adapter"])
    # debug runs (smoke/limit) use a separate file so they can never pollute or truncate
    # the canonical predictions used for the final table
    suffix = ".debug" if (args.smoke or args.limit) else ""
    preds_file = PREDS / f"{model_key}__{ds_key}{suffix}.jsonl"
    preds_file.parent.mkdir(parents=True, exist_ok=True)

    from sgeval.engine import _cfg_tag
    cur_cfg = _cfg_tag(adapter, mcfg.get("gen", {}), mcfg.get("chat_template_kwargs") or {})

    def _resumable(rec: dict) -> bool:
        # a completion counts only if it was produced under the CURRENT configuration and
        # is not a transport failure
        return (not str(rec.get("raw", "")).startswith("<ERROR")
                and rec.get("cfg") == cur_cfg)

    done = set()
    if preds_file.exists() and not args.no_resume:
        n_other = 0
        for l in preds_file.open():
            if not l.strip():
                continue
            rec = json.loads(l)
            if _resumable(rec):
                done.add(rec["id"])
            elif "cfg" in rec or not str(rec.get("raw", "")).startswith("<ERROR"):
                n_other += 1
        if n_other:
            log.info("resume: %d record(s) from a different config will be REDONE "
                     "(cfg mismatch)", n_other)
        log.info("resume: %d/%d already done for %s x %s", len(done), len(samples),
                 model_key, ds_key)
    todo = [s for s in samples if s["id"] not in done]

    if todo:
        if args.engine == "hf" or (args.engine == "auto" and mcfg.get("engine") == "hf"):
            records = run_dataset_hf(
                str(EVAL_DIR / mcfg["path"]), todo, adapter, mcfg.get("gen", {}),
                mcfg.get("chat_template_kwargs") or {}, EVAL_DIR / "data",
                batch_size=args.batch_size, progress_desc=f"{model_key}/{ds_key}")
        else:
            server: VLLMServer = args._server
            records = asyncio.run(run_dataset_vllm(
                server, todo, adapter, mcfg.get("gen", {}),
                mcfg.get("chat_template_kwargs") or {}, EVAL_DIR / "data",
                concurrency=args.concurrency, progress_desc=f"{model_key}/{ds_key}"))
        with preds_file.open("a") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        # engine-death check AFTER persisting: partial good work must survive so the
        # rerun resumes around it instead of redoing everything
        n_err = sum(1 for r in records if str(r.get("raw", "")).startswith("<ERROR"))
        if records and n_err / len(records) > 0.5:
            raise ServerDeadError(
                f"{n_err}/{len(records)} requests failed this pass "
                f"(engine died mid-dataset); see the vLLM log")

    # dedupe by id (retried samples append a second record; last wins) and drop
    # records from ids outside the current sample set (e.g. re-normalized dataset)
    by_id = {r["id"]: r for r in read_jsonl(preds_file)}
    valid_ids = {s["id"] for s in samples}
    all_records = [r for i, r in by_id.items() if i in valid_ids]
    with preds_file.open("w") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    golds, preds, unparsable = finalize_records(all_records)
    m = prf(golds, preds)
    row = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_key,
        "label": mcfg.get("label", model_key),
        "dataset": ds_key,
        "engine": args.engine if args.engine != "auto" else mcfg.get("engine", "vllm"),
        "unparsable": unparsable,
        "n_error": sum(1 for r in all_records if str(r.get("raw", "")).startswith("<ERROR")),
        "n_input_error": sum(1 for r in all_records
                             if str(r.get("raw", "")).startswith("<INPUT_ERROR")),
        "thinking": (mcfg.get("chat_template_kwargs") or {}).get("thinking_type", ""),
        "max_tokens": (mcfg.get("gen") or {}).get("max_tokens", ""),
        "limit": args.limit or "",
        "smoke": bool(args.smoke),
        **m,
    }
    row["n_missing"] = len(samples) - m["n"]
    append_result(row)
    log.info("RESULT %s x %s: F1=%.4f P=%.4f R=%.4f (n=%d, unparsable=%d)",
             model_key, ds_key, m["f1"], m["precision"], m["recall"], m["n"], unparsable)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="",
                    help="comma-separated keys or 'all' (enabled set); default: sing-guard trio")
    ap.add_argument("--datasets", default="all")
    ap.add_argument("--engine", choices=["auto", "vllm", "hf"], default="auto")
    ap.add_argument("--limit", type=int, default=None, help="override per-dataset sample cap")
    ap.add_argument("--smoke", type=int, default=0,
                    help="debug: only first N samples per dataset (still recorded as .smoke)")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--max-seqs", type=int, default=32,
                    help="vLLM --max-num-seqs; caps engine-side concurrency (OOM guard)")
    ap.add_argument("--batch-size", type=int, default=4, help="hf backend batch size")
    ap.add_argument("--port", type=int, default=8199)
    ap.add_argument("--gpu-util", type=float, default=None,
                    help="override gpu_memory_utilization (use ~0.6 on a shared GPU)")
    ap.add_argument("--max-tokens", type=int, default=None, help="override gen.max_tokens")
    ap.add_argument("--thinking", choices=["fast", "fast-slow"], default=None,
                    help="override chat_template_kwargs.thinking_type. NOTE: the released "
                         "template only defines formats for 'fast' and 'fast-slow'; any other "
                         "value (e.g. 'slow') silently renders the FAST prompt while declaring "
                         "the other mode, so it is rejected here")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()
    args._server = None

    models = load_yaml("models.yaml")
    datasets = load_yaml("datasets.yaml")
    models.pop("defaults", None)      # YAML anchor key, not a model
    datasets.pop("defaults", None)    # ditto

    if args.max_tokens is not None or args.thinking:
        for mk_ in models.values():
            if not isinstance(mk_, dict):
                continue
            if args.max_tokens is not None:
                mk_.setdefault("gen", {})["max_tokens"] = args.max_tokens
            if args.thinking:
                kw = mk_.setdefault("chat_template_kwargs", {})
                kw["thinking_type"] = args.thinking

    if args.models in ("", "all"):
        mkeys = [k for k, v in models.items() if isinstance(v, dict) and v.get("enabled") is not False] \
            if args.models == "all" else ["sing-guard-2b", "sing-guard-4b", "sing-guard-8b"]
    else:
        mkeys = args.models.split(",")
    unknown = [k for k in mkeys if k not in models or not isinstance(models[k], dict)]
    if unknown:
        print(f"unknown model keys: {unknown}", file=sys.stderr)
        return 2

    if args.datasets == "all":
        dkeys = list(datasets)
    else:
        dkeys = args.datasets.split(",")

    LOGS.mkdir(exist_ok=True)
    rc = 0
    for mk in mkeys:
        mcfg = models[mk]
        if not (EVAL_DIR / mcfg["path"] / "config.json").exists():
            print(f"[skip] {mk}: {mcfg['path']} missing -- run scripts/download_models.py",
                  file=sys.stderr)
            rc = 1
            continue

        use_hf = args.engine == "hf" or (args.engine == "auto" and mcfg.get("engine") == "hf")
        server = None
        if not use_hf:
            server = VLLMServer(
                str(EVAL_DIR / mcfg["path"]), port=args.port,
                max_model_len=mcfg.get("max_model_len", 8192),
                gpu_memory_utilization=(args.gpu_util if args.gpu_util is not None
                                        else mcfg.get("gpu_memory_utilization", 0.90)),
                max_num_seqs=args.max_seqs,
                log_file=str(LOGS / f"vllm_{mk}.log"))
            try:
                server.start()
            except Exception as e:  # noqa: BLE001
                log.error("failed to start vLLM for %s: %s (logs/%s) -- try --engine hf",
                          mk, e, f"vllm_{mk}.log")
                rc = 1
                continue
            args._server = server

        for dk in dkeys:
            if dk not in datasets:
                log.warning("unknown dataset %s -- skipped", dk)
                continue
            ds_dir = EVAL_DIR / "data" / dk
            jf = ds_dir / "test.jsonl"
            if not jf.exists():
                log.warning("dataset %s not prepared -- skipped (run prepare_data.py)", dk)
                continue
            samples = read_jsonl(jf)
            if args.limit:
                # deterministic subsample for quick comparisons
                samples = samples[:args.limit]
            if args.smoke:
                samples = samples[:args.smoke]
                log.info("SMOKE mode: %s x %s limited to %d samples (results NOT comparable)",
                         mk, dk, len(samples))
            try:
                eval_dataset(mk, mcfg, dk, samples, args)
            except KeyboardInterrupt:
                log.warning("interrupted -- predictions so far are kept; re-run to resume")
                if server:
                    server.stop()
                return 130
            except ServerDeadError as e:
                # engine died (CUDA OOM etc.): continuing would only log connection
                # errors for every remaining dataset -- abort this model, free the GPU.
                log.error("ABORT model %s: %s", mk, e)
                rc = 1
                if server:
                    server.stop()
                    args._server = None
                break
            except Exception as e:  # noqa: BLE001
                log.error("%s x %s failed: %s", mk, dk, e)
                rc = 1

        if server:
            server.stop()
            args._server = None

    # Auto-refresh the Table-4 CSVs (reproduction + delta vs paper) after every run.
    if (RESULTS / "results.csv").exists():
        try:
            import aggregate
            print("\n" + "=" * 70)
            aggregate.main()
            print("=" * 70)
        except Exception as e:  # noqa: BLE001
            log.warning("auto-aggregation failed: %s (run `python aggregate.py` manually)", e)
    else:
        print("\nnext: uv run python aggregate.py   # -> outputs/table4_repro.csv")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
