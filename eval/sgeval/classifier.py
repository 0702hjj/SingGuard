"""Image-classifier guard path (ShieldGemma-2).

ShieldGemma-2 (`ShieldGemma2ForImageClassification`, SigLIP + Gemma-3-4B) is NOT a chat
model: it takes an image and returns a policy-violation probability. There is no text
input, so samples without an image cannot be scored and are skipped (reported via the
`n_missing` column of results.csv).

Runs on transformers directly -- no vLLM, no chat template, greedy-free (the forward pass
is deterministic). Records carry the same fields as the vLLM path, including the config
fingerprint, so resume/aggregation work unchanged.
"""
from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger("sgeval")


def _score_batch(model, inputs) -> list[float]:
    """Return one violation probability per image, absorbing the output-shape variants
    of the custom model class (scalar / per-policy vector / two-class logits)."""
    out = model(**inputs)
    probs = getattr(out, "probabilities", None)
    if probs is None:                       # fall back to softmax over logits
        import torch
        logits = getattr(out, "logits", None)
        if logits is None:
            raise RuntimeError("model returned neither .probabilities nor .logits")
        probs = torch.softmax(logits.float(), dim=-1)
    if probs.dim() == 1:
        return probs.float().tolist()
    if probs.dim() == 2:
        if probs.shape[-1] == 2:            # (batch, [no, yes])
            return probs[:, -1].float().tolist()
        return probs.max(dim=-1).values.float().tolist()   # (batch, n_policies)
    return probs.reshape(probs.shape[0], -1).max(dim=-1).values.float().tolist()


def run_dataset_classifier(model_path: str, samples: list[dict], data_dir: Path,
                           threshold: float = 0.5, batch_size: int = 8,
                           progress_desc: str = "") -> list[dict]:
    import torch
    from PIL import Image
    from transformers import AutoProcessor, ShieldGemma2ForImageClassification

    log.info("loading classifier %s", model_path)
    processor = AutoProcessor.from_pretrained(model_path)
    # Plain .to(device) rather than device_map="auto": the latter pulls in `accelerate`,
    # which the offline environment does not have, and CUDA_VISIBLE_DEVICES already pins
    # the process to one GPU so there is nothing for a device map to decide.
    model = ShieldGemma2ForImageClassification.from_pretrained(
        model_path, torch_dtype=torch.bfloat16).eval().to("cuda")

    def first_image(s: dict) -> str | None:
        v = s.get("image")
        if isinstance(v, list):
            return v[0] if v else None
        return v or None

    usable = [s for s in samples if first_image(s)]
    skipped = len(samples) - len(usable)
    if skipped:
        log.info("classifier: %d/%d samples have no image and are skipped "
                 "(image-only model)", skipped, len(samples))

    cfg = f"shieldgemma2-cls|cls|{threshold}"
    records: list[dict] = []
    from tqdm import tqdm
    for i in tqdm(range(0, len(usable), batch_size), desc=progress_desc):
        batch = usable[i:i + batch_size]
        imgs, batch_cfg = [], []
        for s in batch:
            try:
                imgs.append(Image.open(data_dir / first_image(s)).convert("RGB"))
                batch_cfg.append(True)
            except Exception as e:  # noqa: BLE001
                records.append({"id": s["id"], "gold": s["label"], "pred": None,
                                "raw": f"<INPUT_ERROR {e}>"[:400], "cfg": cfg})
                batch_cfg.append(False)
        if not imgs:
            continue
        inputs = processor(images=imgs, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            scores = _score_batch(model, inputs)
        for s, ok, score in zip(batch, batch_cfg, scores):
            if not ok:
                continue
            pred = 1 if score >= threshold else 0
            records.append({"id": s["id"], "gold": s["label"], "pred": pred,
                            "raw": f"cls_prob={score:.4f}", "cfg": cfg})
    # keep sample order for stable diffs
    order = {s["id"]: i for i, s in enumerate(samples)}
    records.sort(key=lambda r: order.get(r["id"], 1 << 30))
    return records
