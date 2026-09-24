#!/usr/bin/env python
"""Build the paper-styled Table 4 reproduction as a standalone LaTeX document (xelatex).

  uv run python scripts/make_report.py \
      --results ../report/data/results_78.csv,../report/data/results_79.csv

Selection rules (config-fingerprint aware):
- drop smoke / --limit debug rows
- SingGuard rows must carry thinking_type=fast (the configuration that reproduces Table 4)
- per (model, dataset) keep the newest row

Output (into <repo>/report/):
  table4_repro.tex     compilable with `xelatex table4_repro.tex`
  table4_compare.tex   same, with per-column deltas vs the paper
Both are complete documents (article + booktabs), so xelatex works without a wrapper.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

EVAL_DIR = Path(__file__).resolve().parent.parent
REPORT_DIR = EVAL_DIR.parent / "report"
COLMAP = {
    "vlguard": "VLGuard", "jailbreakv": "JailBreakV", "spa-vl": "SPA-VL",
    "mmds-q": "MMDS-Q", "mmds-r": "MMDS-R", "vlsbench": "VLSBench",
    "mm-safety": "MM-Safety", "beavertails-v": "BeaverTails-V",
}
PAPER_COLS = list(COLMAP.values())

PREAMBLE = r"""\documentclass[10pt]{article}
\usepackage{booktabs}
\usepackage{geometry}
\geometry{margin=1.6cm, landscape}
\usepackage{caption}
\captionsetup{font=small}
\begin{document}
"""


def tex_escape(s: str) -> str:
    for a, b in (("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#")):
        s = s.replace(a, b)
    return s


def fmt(v) -> str:
    return "--" if pd.isna(v) else f"{v:.4f}"


def load_results(paths: list[Path]) -> pd.DataFrame:
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    if "smoke" in df.columns:
        df = df[~df["smoke"].astype(str).str.lower().isin(("true", "1"))]
    if "limit" in df.columns:
        df = df[df["limit"].isna() | (df["limit"].astype(str).str.strip().isin(("", "nan")))]
    if "thinking" in df.columns:
        is_sg = df["model"].astype(str).str.startswith("sing-guard")
        wrong = is_sg & (df["thinking"].astype(str) != "fast")
        if wrong.any():
            print(f"note: dropped {int(wrong.sum())} SingGuard rows not in fast mode")
        df = df[~wrong]
    return df.sort_values("ts").groupby(["model", "dataset"], as_index=False).last()



NOTES_BLOCK = r"""
\section*{Notes}
\begin{itemize}\itemsep2pt
  \item \textbf{Configuration.} Released checkpoints; SingGuard rows use
        \texttt{thinking\_type=fast} (the paper's Sec.~2.6 mode for high-throughput benchmark
        evaluation), greedy decoding, \texttt{max\_model\_len}=32768.
  \item \textbf{SPA-VL protocol.} The official test sets are EvalHarm (265) + EvalHelp (265);
        their filenames are \emph{not} query-safety labels (EvalHelp measures helpfulness and
        contains hate-annotated images; EvalHarm measures refusal behaviour). We map
        harm$\rightarrow$unsafe / help$\rightarrow$safe, the reading most consistent with the
        paper's numbers for this column; treat its absolute value as protocol-dependent.
  \item \textbf{JailBreakV coverage.} HF hosts only $\sim$360 of the 28{,}000 images, so 662/1000
        of our sampled rows run text-only. On the image-bearing subset (the benchmark's designed
        form) SingGuard-8B scores F1 0.9600 vs the paper's 0.9728.
  \item \textbf{MMDS protocol.} The corpus carries an official \texttt{set} field
        (train 4045 / val 109 / test 330); we score the \emph{test} split, as the paper does.
        Scoring a stratified sample of the whole corpus instead drops SingGuard-8B to
        0.7722/0.7209 --- the earlier gap in this column was a sampling error on our side,
        not a protocol or checkpoint difference.
  \item \textbf{MM-Safety is all-unsafe.} The benchmark is a set of malicious instructions
        (every sampled row is gold-unsafe), so its F1 reduces to a precision-weighted recall
        on the attack set and is sensitive to which attacks are sampled.
  \item \textbf{Baseline harness fidelity.} Each baseline is driven through its own
        released prompt/template, not a common one. Two defects found and fixed this way:
        LlavaGuard's template ships as \texttt{chat\_template.json} with no inline copy, so
        vLLM silently substituted a generic template and the model returned JSON missing its
        leading keys on 26\% of samples; and GuardReasoner-VL's official interface (INSTRUCTION
        as a system message, a \texttt{Human user:/AI assistant:} transcript, verdict in a
        \texttt{<result>} block) differs from a plain chat call. Baseline prompts are still
        approximations in general --- the paper publishes only SingGuard's templates --- so
        their per-column deltas are indicative, not exact.
  \item \textbf{Still missing.} LlamaGuard3-Vision-11B and LlamaGuard4-12B rows (queued);
        LLaVAShield-v1.0-7B ships a LLaVA-NeXT training-format checkpoint
        (\texttt{LlavaQwenForCausalLM}, no \texttt{auto\_map}) that neither vLLM 0.11 nor
        transformers loads directly --- a converted copy is being validated.
  \item \textbf{Qwen3-VL-235B} is not run: bf16 ($\sim$470\,GB) / FP8 ($\sim$235\,GB) does not fit
        a shared 8$\times$A6000 node without multi-GPU tensor parallelism.
  \item Baseline prompts are approximations (the paper publishes only SingGuard's templates);
        per-column deltas for baseline rows are therefore indicative, not exact.
\end{itemize}
"""

def write_documents(df: pd.DataFrame, paper: pd.DataFrame, labels: dict) -> None:
    df = df.assign(Model=df["model"].map(labels))
    repro = df.pivot_table(index="Model", columns="dataset", values="f1", aggfunc="last")
    repro = repro.rename(columns=COLMAP).reindex(columns=PAPER_COLS)
    order = [m for m in paper["Model"] if m in repro.index] + \
            [m for m in repro.index if m not in set(paper["Model"])]
    repro = repro.loc[order]
    repro["Avg"] = repro.mean(axis=1, skipna=True).round(4)
    n_cols = int(repro[PAPER_COLS].notna().any(axis=0).sum())

    REPORT_DIR.mkdir(exist_ok=True)

    # ---------------- reproduction table ----------------
    body = [
        r"\begin{table*}[t]\centering",
        r"\caption{Reproduction of the SingGuard technical report's Table~4: multimodal "
        r"safety benchmark results (F1 on the unsafe class) using the released checkpoints "
        r"in \texttt{thinking\_type=fast} with greedy decoding. "
        + (r"MMDS columns are still running; " if n_cols < 8 else "")
        + r"Columns present: " + f"{n_cols}/8.}}",
        r"\label{tab:table4-repro}",
        r"\begin{tabular}{l" + "c" * (len(PAPER_COLS) + 1) + r"}\toprule",
        "Model & " + " & ".join(PAPER_COLS) + r" & Avg \\\midrule",
    ]
    for model, row in repro.iterrows():
        cells = [fmt(row[c]) for c in PAPER_COLS]
        body.append(f"{tex_escape(str(model))} & " + " & ".join(cells) + f" & {fmt(row['Avg'])} \\\\")
    body += [r"\bottomrule\end{tabular}\end{table*}"]

    # ---------------- comparison table ----------------
    merged = repro.reset_index().merge(paper, on="Model", how="inner", suffixes=("_ours", "_paper"))
    both = [c for c in PAPER_COLS
            if f"{c}_ours" in merged.columns and merged[f"{c}_ours"].notna().any()]
    comp = [
        r"\begin{table*}[t]\centering",
        r"\caption{Reproduction vs the paper's Table~4 (F1). Each model occupies two rows: our "
        r"reproduced score and, beneath it, $\Delta$ = ours $-$ paper. The Avg column averages "
        r"over the " + f"{len(both)}/8" + r" columns present on both sides.}",
        r"\label{tab:table4-compare}",
        r"\begin{tabular}{l" + "c" * len(both) + r"c}\toprule",
        "Model & " + " & ".join(both) + r" & Avg \\\midrule",
    ]
    for _, r in merged.iterrows():
        ours = " & ".join(fmt(r[f"{c}_ours"]) for c in both)
        deltas = " & ".join(
            "--" if (pd.isna(r[f"{c}_ours"]) or pd.isna(r[f"{c}_paper"]))
            else f"{r[f'{c}_ours'] - r[f'{c}_paper']:+.4f}" for c in both)
        o_avg = merged.loc[merged["Model"] == r["Model"], [f"{c}_ours" for c in both]].iloc[0].mean()
        p_avg = merged.loc[merged["Model"] == r["Model"], [f"{c}_paper" for c in both]].iloc[0].mean()
        comp.append(f"{tex_escape(str(r['Model']))} & " + ours + f" & {o_avg:.4f} \\\\")
        comp.append(r"\hspace{1.2em}$\Delta$ vs paper & " + deltas
                    + f" & {o_avg - p_avg:+.4f} \\\\")
    comp += [r"\bottomrule\end{tabular}\end{table*}", r"\end{document}"]

    (REPORT_DIR / "table4_repro.tex").write_text(
        PREAMBLE + "\n".join(body) + "\n" + NOTES_BLOCK + "\n\\end{document}\n")
    (REPORT_DIR / "table4_compare.tex").write_text(
        PREAMBLE + "\n".join(comp).replace("\\end{document}", "") + "\n" + NOTES_BLOCK
        + "\n\\end{document}\n")

    print(f"columns present: {n_cols}/8 ({', '.join(both)})")
    show = merged[["Model"] + [f"{c}_ours" for c in both]].copy()
    print(show.to_string(index=False))
    print(f"\nwrote {REPORT_DIR}/table4_repro.tex and table4_compare.tex "
          f"(compile with: xelatex table4_repro.tex)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="comma-separated results.csv paths")
    args = ap.parse_args()
    paths = [Path(p) for p in args.results.split(",")]
    models = yaml.safe_load((EVAL_DIR / "configs" / "models.yaml").read_text())
    labels = {k: v.get("label", k) for k, v in models.items() if isinstance(v, dict)}
    paper = pd.read_csv(EVAL_DIR / "configs" / "table4_paper.csv").rename(columns={"model": "Model"})
    write_documents(load_results(paths), paper, labels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
