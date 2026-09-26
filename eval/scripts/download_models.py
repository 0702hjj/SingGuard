#!/usr/bin/env python
"""Download model checkpoints: ModelScope (CN direct) first, hf-mirror fallback.

Usage:
  uv run python scripts/download_models.py                 # all enabled models
  uv run python scripts/download_models.py --models sing-guard-8b,qwen3-vl-8b
  uv run python scripts/download_models.py --list
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

EVAL_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EVAL_DIR))


def load_models() -> dict:
    cfg = yaml.safe_load((EVAL_DIR / "configs" / "models.yaml").read_text())
    return {k: v for k, v in cfg.items() if isinstance(v, dict) and k != "defaults"}


def looks_complete(dest: Path) -> bool:
    """Only trust our own marker file written after a successful snapshot_download.
    Heuristics on filenames are unreliable: interrupted ModelScope transfers keep the
    final filename on partial shards."""
    return (dest / ".complete").exists()


def download_modelscope(repo_id: str, dest: Path) -> None:
    from modelscope import snapshot_download
    # max_workers is the number of files fetched concurrently. The default serialises them,
    # which caps throughput at one stream (~3.5 MB/s measured) -- a 237 GB checkpoint would
    # take ~19 h instead of ~5. Shards are independent, so fetching several at once is safe.
    snapshot_download(repo_id, local_dir=str(dest), max_workers=8)


def download_hf(repo_id: str, dest: Path) -> None:
    # CN-friendly mirror; override by exporting HF_ENDPOINT (or empty to use huggingface.co).
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id, local_dir=str(dest))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="", help="comma-separated keys; default = enabled set")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-download even if dir looks complete")
    args = ap.parse_args()

    models = load_models()
    if args.list:
        for k, v in models.items():
            state = "off " if v.get("enabled") is False else "on  "
            print(f"{state} {k:28s} adapter={v.get('adapter','-'):18s} "
                  f"ms={v.get('ms_id','-'):32s} hf={v.get('hf_id','-')}")
        return 0

    keys = [k for k in args.models.split(",") if k] if args.models else \
           [k for k, v in models.items() if v.get("enabled") is not False]
    unknown = [k for k in keys if k not in models]
    if unknown:
        print(f"unknown model keys: {unknown}; run --list", file=sys.stderr)
        return 2

    (EVAL_DIR / "models").mkdir(exist_ok=True)
    failures = []
    for k in keys:
        v = models[k]
        dest = EVAL_DIR / v["path"]
        dest.mkdir(parents=True, exist_ok=True)
        if looks_complete(dest) and not args.force:
            print(f"[skip] {k}: {dest} already looks complete")
            continue
        done = False
        if v.get("ms_id"):
            print(f"[modelscope] {k} <- {v['ms_id']}")
            try:
                download_modelscope(v["ms_id"], dest)
                done = True
            except Exception as e:  # noqa: BLE001
                print(f"  modelscope failed: {e}")
        if not done and v.get("hf_id") and v["hf_id"] != "TBD":
            print(f"[hf-mirror] {k} <- {v['hf_id']}")
            try:
                download_hf(v["hf_id"], dest)
                done = True
            except Exception as e:  # noqa: BLE001
                print(f"  hf download failed: {e}")
                print("  (gated model? accept the license on the HF web page, then "
                      "export HF_TOKEN=<token> and retry)")
        if not done:
            failures.append(k)
            print(f"[FAIL] {k}")
        else:
            (dest / ".complete").write_text("ok\n")
            print(f"[ok] {k} -> {dest}")

    if failures:
        print(f"\nfailed: {failures}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
