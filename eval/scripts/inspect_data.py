#!/usr/bin/env python
"""Inspect a downloaded raw dataset: file tree, schema, first row, label distribution.

Run this after prepare_data.py --download (or on schema errors) and before trusting numbers.

  uv run python scripts/inspect_data.py vlguard
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

EVAL_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = EVAL_DIR / "data"


def tree(raw: Path, max_entries=40):
    entries = sorted(p for p in raw.rglob("*") if p.is_file())
    print(f"files: {len(entries)}")
    for p in entries[:max_entries]:
        print(f"  {p.relative_to(raw)}  ({p.stat().st_size // 1024} KB)")
    if len(entries) > max_entries:
        print(f"  ... and {len(entries) - max_entries} more")
    return entries


def describe_row(row: dict, i: int):
    print(f"\n--- row {i} ---")
    for k, v in row.items():
        s = repr(v)
        if len(s) > 160:
            s = s[:160] + "…"
        print(f"  {k}: {type(v).__name__} = {s}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--rows", type=int, default=3)
    args = ap.parse_args()

    raw = DATA_DIR / "raw" / args.dataset
    if not raw.exists():
        print(f"no raw dir at {raw}; run prepare_data.py --datasets {args.dataset} first")
        return 1
    entries = tree(raw)

    parquets = [p for p in entries if p.suffix == ".parquet"]
    csvs = [p for p in entries if p.suffix == ".csv"]
    shown = 0
    for p in (parquets or csvs):
        print(f"\n=== {p.name} ===")
        df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
        print(f"shape: {df.shape}")
        print(f"columns: {list(df.columns)}")
        for c in df.columns:
            try:
                print(f"  {c}: dtype={df[c].dtype}, nunique={df[c].nunique(dropna=True)}, "
                      f"sample={repr(df[c].dropna().iloc[0])[:120]}")
            except Exception:  # noqa: BLE001
                pass
        for i, row in enumerate(df.head(args.rows).to_dict("records")):
            describe_row(row, i)
        shown += 1
        if shown >= 3:
            break

    if not shown:
        for p in [e for e in entries if e.suffix in (".json", ".jsonl")][:3]:
            if p.name in ("manifest.json",):
                continue
            print(f"\n=== {p.name} ===")
            try:
                obj = json.loads(p.read_text())
                rows = obj if isinstance(obj, list) else [obj]
                for i, row in enumerate(rows[:args.rows]):
                    describe_row(row, i)
            except Exception as e:  # noqa: BLE001
                print(f"  unreadable: {e}")

    done = DATA_DIR / args.dataset / "test.jsonl"
    if done.exists():
        labels = [json.loads(l)["label"] for l in done.open()]
        print(f"\nnormalized: {done} -> {len(labels)} rows, {sum(labels)} unsafe")
    else:
        print(f"\n(no normalized output yet at {done})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
