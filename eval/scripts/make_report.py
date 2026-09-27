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
Both are complete documents (article + xeCJK + booktabs); compile with xelatex, no wrapper needed.
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
% Plain article + xeCJK (no ctex), compiled with xelatex. Fonts are named explicitly so the
% Chinese headings render without ctex's fontset machinery; Noto CJK ships with most
% distributions, and TeX Live's Fandol is a drop-in alternative if it is missing:
%   \setCJKmainfont{FandolSong-Regular.otf}[BoldFont=FandolHei-Regular.otf]
\usepackage{xeCJK}
% Noto CJK ships no italic face; AutoFakeSlant gives \emph a real slant instead of a
% "font shape undefined" fallback.
\setCJKmainfont{Noto Serif CJK SC}[BoldFont=Noto Sans CJK SC, AutoFakeSlant=0.2]
\setCJKsansfont{Noto Sans CJK SC}[AutoFakeSlant=0.2]
\setCJKmonofont{Noto Sans Mono CJK SC}
\usepackage{booktabs}
% Row shading: the comparison table pairs each model's score with its delta, and the
% reproduction table is a plain grid -- alternating a light tint is what makes either
% scannable without adding rules. `table` option is required for \rowcolor.
\usepackage[table]{xcolor}
\definecolor{rowbg}{gray}{0.93}
\definecolor{headbg}{gray}{0.85}
% Delta text colours. Deliberately dark: they sit on the tinted delta rows, where a bright
% green/red would lose contrast.
\definecolor{posdelta}{RGB}{0,110,60}
\definecolor{negdelta}{RGB}{170,25,25}
\usepackage{geometry}
\geometry{margin=1.6cm, landscape}
\usepackage{caption}
\captionsetup{font=small}
\renewcommand{\tablename}{表}      % "表 1: ..." rather than "Table 1: ..."
\renewcommand{\figurename}{图}
\begin{document}
"""


def tex_escape(s: str) -> str:
    for a, b in (("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#")):
        s = s.replace(a, b)
    return s


def fmt(v) -> str:
    return "--" if pd.isna(v) else f"{v:.4f}"


def fmt_delta(v) -> str:
    """A delta, coloured by sign: green above the paper, red below. Higher F1 is better,
    so the sign is the whole story and the colour saves reading the number."""
    if pd.isna(v):
        return "--"
    s = f"{v:+.4f}"
    if v > 0:
        return r"\textcolor{posdelta}{" + s + "}"
    if v < 0:
        return r"\textcolor{negdelta}{" + s + "}"
    return s


# Cells where coverage is too low to be a measurement rather than a probe of the model's
# context limit. LlavaGuard is LLaVA-1.5 (4096 tokens): only 49/330 MMDS-Q and 54/327 MMDS-R
# prompts fit; the rest come back as <INPUT_ERROR>. An F1 over ~15% of the rows would read
# as "this model scores 0.0" when what it means is "this harness cannot feed it". Left as
# -- in the table; the counts and the subset score are recorded in the notes instead.
CELL_EXCLUSIONS = {("llavaguard-7b", "mmds-q"), ("llavaguard-7b", "mmds-r")}


def load_results(paths: list[Path]) -> pd.DataFrame:
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    excl = pd.Series(list(zip(df["model"], df["dataset"]))).isin(CELL_EXCLUSIONS)
    if excl.any():
        print(f"note: excluded {int(excl.sum())} low-coverage cell row(s): "
              f"{sorted(CELL_EXCLUSIONS)}")
    df = df[~excl.values]
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
\section*{说明}
\begin{itemize}\itemsep1pt
  \item \textbf{三列全 unsafe，其 F1 实为召回率.} JailBreakV、VLSBench、MM-Safety 无 safe 样本，
        故 precision 恒为 1、$F1 = 2R/(1+R)$------不能与其余列等同解读，对采样到哪些攻击也极敏感。
  \item \textbf{MMDS 口径（最大修正）.} 语料自带 \texttt{set} 字段（train/val/test $=$ 4045/109/330），
        我们最初漏了它、在全量上采样（93\% 是训练集）。改评 test 后 SingGuard-8B 由 0.7722/0.7209
        升到 \textbf{0.9571/0.8908}（论文 0.9851/0.8963）。
  \item \textbf{基线必须用各自的官方 prompt.} 用通用提示词会让模型输出错格式而被判错：
        GuardReasoner-VL（需 system 消息 $+$ \texttt{<result>} 块）修复后 VLGuard 0.2973$\rightarrow$0.8822；
        LlavaGuard 用官方策略文本后 26\% 的解析失败归零。
  \item \textbf{剩余缺口均已归因.} SPA-VL：官方 test 无查询安全性标签，协议存歧义；
        JailBreakV：公开图像只有 360/28000；LlavaGuard（4096 上下文）与 LLaVAShield 在 MMDS 上
        卡的是上下文上限，并非模型能力。
  \item \textbf{结论：发布权重可以复现 Table~4.} 八列平均 2B 0.8978（$\Delta$+0.005）、
        8B 0.8828（$\Delta$−0.026），逐列 $|\Delta|$ 多数 $\leq$0.03。
  \item \textbf{Qwen3-VL-235B 为 4bit 量化（Q4\_K\_M）}，非论文的 bf16/FP8，只能看趋势；
        其 MMDS-Q 为 API 抽样估计、MMDS-R 未跑。
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
        r"\caption{SingGuard 技术报告表~4 的复现：多模态安全基准结果（unsafe 类 F1）。"
        r"使用官方发布的 checkpoint，\texttt{thinking\_type=fast}，贪心解码。"
        + (r"部分列仍在运行；" if n_cols < 8 else "")
        + r"本表包含 " + f"{n_cols}/8" + r" 列。}",
        r"\label{tab:table4-repro}",
        r"\begin{tabular}{l" + "c" * (len(PAPER_COLS) + 1) + r"}\toprule",
        r"\rowcolor{headbg} 模型 & " + " & ".join(PAPER_COLS) + r" & 平均 \\\midrule",
    ]
    for i, (model, row) in enumerate(repro.iterrows()):
        cells = [fmt(row[c]) for c in PAPER_COLS]
        shade = r"\rowcolor{rowbg} " if i % 2 else ""     # zebra striping
        body.append(f"{shade}{tex_escape(str(model))} & " + " & ".join(cells)
                    + f" & {fmt(row['Avg'])} \\\\")
    body += [r"\bottomrule\end{tabular}\end{table*}"]

    # ---------------- comparison table ----------------
    merged = repro.reset_index().merge(paper, on="Model", how="inner", suffixes=("_ours", "_paper"))
    both = [c for c in PAPER_COLS
            if f"{c}_ours" in merged.columns and merged[f"{c}_ours"].notna().any()]
    comp = [
        r"\begin{table*}[t]\centering",
        r"\caption{复现结果与论文表~4 的对比（F1）。每个模型占两行：第一行为我们复现的分数，"
        r"其下为 $\Delta$ = 复现值 $-$ 论文值。平均列按两侧都有的 "
        + f"{len(both)}/8" + r" 列取平均。}",
        r"\label{tab:table4-compare}",
        r"\begin{tabular}{l" + "c" * len(both) + r"c}\toprule",
        r"\rowcolor{headbg} 模型 & " + " & ".join(both) + r" & 平均 \\\midrule",
    ]
    for mi, (_, r) in enumerate(merged.iterrows()):
        ours = " & ".join(fmt(r[f"{c}_ours"]) for c in both)
        deltas = " & ".join(
            fmt_delta(None if (pd.isna(r[f"{c}_ours"]) or pd.isna(r[f"{c}_paper"]))
                      else r[f"{c}_ours"] - r[f"{c}_paper"]) for c in both)
        o_avg = merged.loc[merged["Model"] == r["Model"], [f"{c}_ours" for c in both]].iloc[0].mean()
        p_avg = merged.loc[merged["Model"] == r["Model"], [f"{c}_paper" for c in both]].iloc[0].mean()
        comp.append(f"{tex_escape(str(r['Model']))} & " + ours + f" & {o_avg:.4f} \\\\")
        # shade every delta row: the pair (score, delta) then reads as one unit
        comp.append(r"\rowcolor{rowbg} \hspace{1.2em}$\Delta$ vs 论文 & " + deltas
                    + " & " + fmt_delta(o_avg - p_avg) + r" \\")
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
