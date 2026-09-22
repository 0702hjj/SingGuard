#!/usr/bin/env python
"""Aggregate results/results.csv into a Table-4-shaped comparison against the paper.

  uv run python aggregate.py            # writes outputs/*.csv and prints a markdown table
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

EVAL_DIR = Path(__file__).resolve().parent

# dataset key -> paper column name
COLMAP = {
    "vlguard": "VLGuard", "jailbreakv": "JailBreakV", "spa-vl": "SPA-VL",
    "mmds-q": "MMDS-Q", "mmds-r": "MMDS-R", "vlsbench": "VLSBench",
    "mm-safety": "MM-Safety", "beavertails-v": "BeaverTails-V",
}
PAPER_ORDER = list(COLMAP.values())


def main() -> int:
    results_csv = EVAL_DIR / "results" / "results.csv"
    paper_csv = EVAL_DIR / "configs" / "table4_paper.csv"
    if not results_csv.exists():
        print("no results yet -- run runner.py first")
        return 1

    df = pd.read_csv(results_csv)
    if "smoke" in df.columns:  # smoke runs are for pipeline debugging only
        df = df[~df["smoke"].astype(str).str.lower().isin(("true", "1"))]
    df = df.sort_values("ts").groupby(["model", "dataset"], as_index=False).last()

    models_yaml = yaml.safe_load((EVAL_DIR / "configs" / "models.yaml").read_text())
    label_of = {k: v.get("label", k) for k, v in models_yaml.items() if isinstance(v, dict)}
    df["Model"] = df["model"].map(label_of)

    repro = df.pivot_table(index="Model", columns="dataset", values="f1", aggfunc="last")
    repro = repro.rename(columns=COLMAP).reindex(columns=PAPER_ORDER)
    repro["Avg"] = repro.mean(axis=1, skipna=True).round(4)
    repro = repro.round(4).reset_index()

    out_dir = EVAL_DIR / "outputs"
    out_dir.mkdir(exist_ok=True)
    repro.to_csv(out_dir / "table4_repro.csv", index=False)

    paper = pd.read_csv(paper_csv).rename(columns={"model": "Model"})
    comp = repro.merge(paper, on="Model", how="outer", suffixes=("_repro", "_paper"))
    for c in PAPER_ORDER + ["Avg"]:
        comp[f"{c} Δ"] = (comp[f"{c}_repro"] - comp[f"{c}_paper"]).round(4)
    # rows: paper-table order first, then any extra reproduced models
    order = {m: i for i, m in enumerate(paper["Model"])}
    comp = comp.assign(_o=comp["Model"].map(lambda m: order.get(m, 900 + len(m)))) \
        .sort_values("_o").drop(columns="_o")
    comp.to_csv(out_dir / "table4_compare.csv", index=False)

    # ---- console summary -------------------------------------------------
    print("\n== Reproduction (F1, unsafe class) ==")
    print(repro.to_markdown(index=False))

    merged = repro.merge(paper, on="Model", how="inner", suffixes=("_repro", "_paper"))
    if not merged.empty:
        print("\n== Δ vs paper (negative = below paper) ==")
        delta = merged[["Model"]].copy()
        for c in PAPER_ORDER:
            delta[c] = (merged[f"{c}_repro"] - merged[f"{c}_paper"]).round(3)
        delta["Avg"] = (merged["Avg_repro"] - merged["Avg_paper"]).round(3)
        print(delta.to_markdown(index=False))

    missing = [c for c in PAPER_ORDER if c not in repro.columns or repro[c].isna().any()]
    if missing:
        print(f"\nnote: incomplete columns (model/dataset runs missing): {missing}")
    print(f"\nwrote: {out_dir / 'table4_repro.csv'}")
    print(f"wrote: {out_dir / 'table4_compare.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
