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
\section*{说明}
\begin{itemize}\itemsep2pt
  \item \textbf{配置.} 使用官方发布的 checkpoint；SingGuard 各行采用论文
        Sec.~2.6 中面向高吞吐基准评测的 \texttt{thinking\_type=fast}，贪心解码，
        \texttt{max\_model\_len}=32768。
  \item \textbf{MMDS 口径.} 语料自带官方 \texttt{set} 字段
        （train 4045 / val 109 / test 330），我们与论文一致地评测 \emph{test} 划分。
        若改为在全量语料上分层采样，SingGuard-8B 会掉到 0.7722/0.7209------该列此前的
        差距源于我们这边的采样错误，而非协议或权重的差异。
  \item \textbf{SPA-VL 协议.} 官方测试集为 EvalHarm (265) + EvalHelp (265)；其文件名
        \emph{并非}查询安全性标签（EvalHelp 衡量有用性，且包含带仇恨标注的图像；EvalHarm
        衡量拒答行为）。我们按 harm$\rightarrow$unsafe / help$\rightarrow$safe 映射，
        这是与该列论文数值最自洽的读法；其绝对值应视为依赖协议。
  \item \textbf{JailBreakV 覆盖率.} HF 上仅有约 360 张（共 28{,}000 张）图像，
        因此我们采样的 1000 行中有 662 行为纯文本。在带图像子集（该基准的设计形态）上，
        SingGuard-8B 的 F1 为 0.9600，论文为 0.9728。
  \item \textbf{MM-Safety 全为 unsafe.} 该基准由恶意指令构成（我们采样的每一行 gold 均为
        unsafe），因此其 F1 退化为攻击集上的加权召回，对采样到哪些攻击敏感。
  \item \textbf{基线实现保真度.} 每条基线都走其自身发布的 prompt/模板，而非统一模板。
        由此发现并修复了两处缺陷：LlavaGuard 的模板以 \texttt{chat\_template.json} 形式发布
        且 \texttt{tokenizer\_config} 中没有内联副本，导致 vLLM 静默改用通用模板，
        26\% 的样本返回缺失开头键的 JSON 残片；GuardReasoner-VL 的官方接口
        （INSTRUCTION 作为 system message、\texttt{Human user:/AI assistant:} 转录、
        结论置于 \texttt{<result>} 块）并非常规 chat 调用。总体而言基线 prompt 仍是近似------
        论文只公布了 SingGuard 的模板------故其逐列 $\Delta$ 仅供参考，并非精确值。
  \item \textbf{尚缺.} LlamaGuard3-Vision-11B 与 LlamaGuard4-12B 两行（已排队）；
        LLaVAShield-v1.0-7B 发布的是 LLaVA-NeXT 训练格式 checkpoint
        （\texttt{LlavaQwenForCausalLM}，无 \texttt{auto\_map}），vLLM 0.11 与 transformers
        均无法直接加载------转换版本正在验证中。
  \item \textbf{Qwen3-VL-235B} 未运行：bf16（约 470\,GB）/ FP8（约 235\,GB）无法在共享的
        8$\times$A6000 节点上容纳（除非使用多卡张量并行）。
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
        "模型 & " + " & ".join(PAPER_COLS) + r" & 平均 \\\midrule",
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
        r"\caption{复现结果与论文表~4 的对比（F1）。每个模型占两行：第一行为我们复现的分数，"
        r"其下为 $\Delta$ = 复现值 $-$ 论文值。平均列按两侧都有的 "
        + f"{len(both)}/8" + r" 列取平均。}",
        r"\label{tab:table4-compare}",
        r"\begin{tabular}{l" + "c" * len(both) + r"c}\toprule",
        "模型 & " + " & ".join(both) + r" & 平均 \\\midrule",
    ]
    for _, r in merged.iterrows():
        ours = " & ".join(fmt(r[f"{c}_ours"]) for c in both)
        deltas = " & ".join(
            "--" if (pd.isna(r[f"{c}_ours"]) or pd.isna(r[f"{c}_paper"]))
            else f"{r[f'{c}_ours'] - r[f'{c}_paper']:+.4f}" for c in both)
        o_avg = merged.loc[merged["Model"] == r["Model"], [f"{c}_ours" for c in both]].iloc[0].mean()
        p_avg = merged.loc[merged["Model"] == r["Model"], [f"{c}_paper" for c in both]].iloc[0].mean()
        comp.append(f"{tex_escape(str(r['Model']))} & " + ours + f" & {o_avg:.4f} \\\\")
        comp.append(r"\hspace{1.2em}$\Delta$ vs 论文 & " + deltas
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
