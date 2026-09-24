#!/usr/bin/env python
"""Recompute metrics for (model, dataset) cells from stored predictions -- no GPU needed.

Use when the *sample set* changed (e.g. filtering a dataset variant) or the parsing/metrics
code was updated: re-derives rows in results/results.csv from results/preds/*.jsonl.

  uv run python scripts/recompute.py                          # all cells
  uv run python scripts/recompute.py --datasets mm-safety     # only these
  uv run python scripts/recompute.py --models sing-guard-8b --dry-run
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EVAL_DIR))

from sgeval.engine import finalize_records  # noqa: E402
from sgeval.metrics import prf               # noqa: E402

RESULTS = EVAL_DIR / "results"
PREDS = RESULTS / "preds"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="")
    ap.add_argument("--datasets", default="")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import yaml
    models = yaml.safe_load((EVAL_DIR / "configs" / "models.yaml").read_text())
    models.pop("defaults", None)
    mfilter = set(args.models.split(",")) if args.models else None
    dfilter = set(args.datasets.split(",")) if args.datasets else None

    rows_out = []
    for f in sorted(PREDS.glob("*__*.jsonl")):
        mkey, dkey = f.stem.split("__", 1)
        if mfilter and mkey not in mfilter:
            continue
        if dfilter and dkey not in dfilter:
            continue
        jf = EVAL_DIR / "data" / dkey / "test.jsonl"
        if not jf.exists():
            print(f"[skip] {mkey} x {dkey}: dataset jsonl missing")
            continue
        valid_ids = {json.loads(l)["id"] for l in jf.open() if l.strip()}
        by_id = {r["id"]: r for r in (json.loads(l) for l in f.open() if l.strip())}
        # mirror runner semantics: <ERROR/> records are kept and scored as incorrect
        records = [r for i, r in by_id.items() if i in valid_ids]
        if not records:
            continue
        golds, preds, unparsable = finalize_records(records)
        m = prf(golds, preds)
        row = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": mkey,
            "label": models.get(mkey, {}).get("label", mkey),
            "dataset": dkey,
            "engine": "recompute",
            "unparsable": unparsable,
            "smoke": False,
            **m,
        }
        print(f"{mkey:16s} x {dkey:16s} n={m['n']:5d} F1={m['f1']:.4f} "
              f"P={m['precision']:.4f} R={m['recall']:.4f} unparsable={unparsable}")
        rows_out.append(row)

    if args.dry_run:
        print(f"\n(dry-run: {len(rows_out)} rows would be appended)")
        return 0

    if rows_out:
        path = RESULTS / "results.csv"
        exists_before = path.exists()
        # Column order must come from the file's own header, not from this row's key order:
        # results.csv is written by several producers and DictWriter emits values in
        # `fieldnames` order, so a differently-ordered row silently shifts every column.
        if path.exists():
            with path.open(newline="") as fh:
                header = next(csv.reader(fh))
            dropped = sorted({k for r in rows_out for k in r if k not in header})
            if dropped:
                print(f"[warn] keys absent from the results.csv header, dropped: {dropped}")
        else:
            header = list(rows_out[0])
        with path.open("a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=header, extrasaction="ignore")
            if not exists_before:
                w.writeheader()
            w.writerows(rows_out)
        print(f"\nappended {len(rows_out)} rows to results.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
