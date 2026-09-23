#!/usr/bin/env python
"""Validate prepared dataset artifacts against the protocol expectations.

Catches the failure mode that motivated it: a stale local copy (e.g. VLGuard generated
by an older loader that kept assistant responses) being rsynced to the GPU server, which
silently corrupts an entire table column.

  uv run python scripts/verify_data.py            # check every dataset present
  uv run python scripts/verify_data.py --data-dir /path/to/eval/data

Exit code 0 = all checks pass; 1 = at least one violation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent.parent

# dataset key -> (expect_image, forbid_response, require_response, min_rows)
#   forbid_response: query-side benchmark -- a response in the artifact means a stale
#                    copy from before the protocol fix (VLGuard recall collapses to 0)
#   require_response: response-side benchmark -- rows without a response are unusable
RULES = {
    "vlguard":       dict(expect_image=True,  forbid_response=True,  require_response=False,
                          min_image_frac=0.95),
    "spa-vl":        dict(expect_image=True,  forbid_response=True,  require_response=False,
                          min_image_frac=0.95),
    "jailbreakv":    dict(expect_image=None,  forbid_response=True,  require_response=False,
                          min_image_frac=0.0),   # many images only on Google Drive
    "vlsbench":      dict(expect_image=True,  forbid_response=True,  require_response=False,
                          min_image_frac=1.0),
    "mm-safety":     dict(expect_image=True,  forbid_response=True,  require_response=False,
                          min_image_frac=1.0),   # image variants only (Text_only excluded)
    "beavertails-v": dict(expect_image=True,  forbid_response=False, require_response=True,
                          min_image_frac=1.0),
    "vlguard-q":     dict(expect_image=True,  forbid_response=True,  require_response=False,
                          min_image_frac=0.95),
}


def check(key: str, jf: Path, rules: dict) -> list[str]:
    problems: list[str] = []
    rows = [json.loads(l) for l in jf.open() if l.strip()]
    if not rows:
        return [f"{key}: empty file {jf}"]
    n = len(rows)
    n_img = sum(1 for r in rows if r.get("image"))
    n_resp = sum(1 for r in rows if r.get("response"))
    n_pos = sum(1 for r in rows if r.get("label") == 1)

    frac = rules.get("min_image_frac", 1.0)
    if rules["expect_image"] is True and n_img < frac * n:
        problems.append(
            f"{key}: only {n_img}/{n} rows carry an image (expected >= {frac:.0%}) "
            f"-- stale artifact (wrong variant mix / failed extraction); regenerate")
    if rules["expect_image"] is None and n_img == 0:
        print(f"  [info] {key}: 0 images (allowed for this dataset)")
    if rules["forbid_response"] and n_resp:
        problems.append(
            f"{key}: {n_resp}/{n} rows have an assistant response but this benchmark is "
            f"query-side -- STALE artifact from before the protocol fix; regenerate "
            f"(see OPS.md §3)")
    if rules["require_response"] and n_resp < 0.9 * n:
        problems.append(f"{key}: only {n_resp}/{n} rows have a response (response-side "
                        f"benchmark needs them)")

    # image file existence
    missing = 0
    for r in rows:
        if r.get("image"):
            p = Path(r["image"])
            if not p.is_absolute():
                p = EVAL_DIR / "data" / p
            if not p.exists():
                missing += 1
    if missing:
        problems.append(f"{key}: {missing} referenced image files are missing on disk")

    # sanity: labels present and binary, ids unique
    if any(r.get("label") not in (0, 1) for r in rows):
        problems.append(f"{key}: labels outside {{0,1}}")
    if len({r["id"] for r in rows}) != n:
        problems.append(f"{key}: duplicate ids")

    print(f"  {key}: n={n} unsafe={n_pos} with_image={n_img} with_response={n_resp}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(EVAL_DIR / "data"))
    args = ap.parse_args()
    data_dir = Path(args.data_dir)

    all_problems: list[str] = []
    checked = 0
    for key, rules in RULES.items():
        jf = data_dir / key / "test.jsonl"
        if not jf.exists():
            print(f"  {key}: (not present)")
            continue
        checked += 1
        all_problems += check(key, jf, rules)

    print()
    if all_problems:
        print("VIOLATIONS:")
        for p in all_problems:
            print(f"  - {p}")
        return 1
    print(f"all {checked} dataset artifacts OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
