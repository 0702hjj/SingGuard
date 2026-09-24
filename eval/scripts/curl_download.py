#!/usr/bin/env python
"""Server-side model downloader that bypasses huggingface_hub's xet routing.

Why this exists: huggingface_hub >= 1.x routes weight blobs through HuggingFace's xet
storage (cas-bridge.xethub.hf.co), which is unreachable from the GPU servers, and
HF_HUB_DISABLE_XET has no effect there. hf-mirror.com happily serves the same bytes over
plain HTTPS (verified: HTTP 206, ~2 MB/s), so we fetch the file list from the mirror API
and pull each file with `curl -C -` in a retry loop -- the same technique that recovered
the VLSBench dataset.

  python scripts/curl_download.py --repo RealSafe/LLaVAShield-v1.0-7B --dest models/llavashield-v1-7b
  python scripts/curl_download.py --repo yueliu1999/GuardReasoner-VL-7B --dest models/guardreasoner-vl-7b

Writes a .complete marker on success (same contract as download_models.py).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

DEFAULT_ENDPOINT = "https://hf-mirror.com"


def list_files(repo: str, endpoint: str) -> list[dict]:
    """hf-mirror blocks its tree API (403), so the file list can also be supplied as JSON
    produced elsewhere (e.g. locally via huggingface_hub.list_repo_files + file sizes)."""
    url = f"{endpoint}/api/models/{repo}/tree/main?recursive=true"
    with urllib.request.urlopen(url, timeout=30) as r:
        entries = json.load(r)
    return [e for e in entries if e.get("type") == "file"]


def fetch(url: str, dest: Path, retries: int = 200) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        # note: --retry-all-errors needs curl >= 7.71; the outer loop provides the retries
        cmd = ["curl", "-sSL", "-C", "-", "--connect-timeout", "20",
               "--max-time", "1800", "-o", str(dest), url]
        if subprocess.run(cmd).returncode == 0:
            return True
        time.sleep(min(2 + attempt, 15))
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="hf-mirror repo id, e.g. RealSafe/LLaVAShield-v1.0-7B")
    ap.add_argument("--dest", required=True, help="local destination directory")
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    ap.add_argument("--file-list", default=None,
                    help="JSON list of {path,size} to fetch instead of the (403-blocked) mirror API")
    args = ap.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    if args.file_list:
        files = json.loads(Path(args.file_list).read_text())
    else:
        files = list_files(args.repo, args.endpoint)
    total = sum(f.get("size", 0) for f in files)
    print(f"[curl-dl] {args.repo}: {len(files)} files, {total / 1e9:.1f} GB -> {dest}")

    for i, f in enumerate(files, 1):
        rel, size = f["path"], f.get("size", 0)
        path = dest / rel
        if path.exists() and (size == 0 or path.stat().st_size == size):
            print(f"[{i}/{len(files)}] skip (complete) {rel}")
            continue
        url = f"{args.endpoint}/{args.repo}/resolve/main/{rel}"
        print(f"[{i}/{len(files)}] {rel} ({size / 1e6:.0f} MB)")
        if not fetch(url, path):
            print(f"[curl-dl] FAILED on {rel}", file=sys.stderr)
            return 1
        got = path.stat().st_size
        if size and got != size:
            print(f"[curl-dl] size mismatch on {rel}: {got} != {size}", file=sys.stderr)
            return 1

    (dest / ".complete").write_text("ok\n")
    print(f"[curl-dl] DONE {args.repo} -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
