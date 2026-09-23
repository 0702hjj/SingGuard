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
    n_all = len(df)
    if "smoke" in df.columns:  # smoke runs are for pipeline debugging only
        df = df[~df["smoke"].astype(str).str.lower().isin(("true", "1"))]
    if "limit" in df.columns:  # a --limit debug run must not displace the canonical cell
        df = df[df["limit"].isna() | (df["limit"].astype(str).str.strip() == "")]
    # debug backends (hf fallback) rank below the reference vLLM path for the same cell
    if "engine" in df.columns:
        df = df.assign(_canon=df["engine"].isin(["vllm", "recompute"]).astype(int))
        df = df.sort_values(["ts", "_canon"]).groupby(["model", "dataset"], as_index=False).last()
    else:
        df = df.sort_values("ts").groupby(["model", "dataset"], as_index=False).last()
    dropped = n_all - len(df)
    if dropped:
        print(f"note: {dropped}/{n_all} result rows excluded (smoke/limit/non-canonical)")

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
        # Avg must compare like with like: restrict to columns present in BOTH sides
        # (MMDS-Q/R are absent, so a raw 6-vs-8-column Avg comparison is misleading)
        both = [c for c in PAPER_ORDER
                if c in merged.columns and merged[f"{c}_repro"].notna().all()
                and f"{c}_paper" in merged.columns]
        if both:
            avg_r = merged[[f"{c}_repro" for c in both]].mean(axis=1)
            avg_p = merged[[f"{c}_paper" for c in both]].mean(axis=1)
            delta[f"Avg({len(both)}/8)"] = (avg_r - avg_p).round(3)
            delta["_avg_cols"] = len(both)
        print(delta.to_markdown(index=False))
        if len(both) < 8:
            print(f"note: Avg Δ computed over the {len(both)} columns present on both sides "
                  f"({', '.join(both)}); missing vs paper: "
                  f"{[c for c in PAPER_ORDER if c not in both]}")

    missing = [c for c in PAPER_ORDER if c not in repro.columns or repro[c].isna().any()]
    if missing:
        print(f"\nnote: incomplete columns (model/dataset runs missing): {missing}")
    print(f"\nwrote: {out_dir / 'table4_repro.csv'}")
    print(f"wrote: {out_dir / 'table4_compare.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
