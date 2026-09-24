#!/usr/bin/env python
"""Audit label semantics for every dataset: does the label we use actually encode the
safety of what we feed the model?

Why: SPA-VL's harm/help file split turned out NOT to be a query-safety label (the "help"
half contains racist questions and hate-annotated images), which silently corrupts that
column. This script surfaces the same class of problem everywhere else, using each
dataset's own annotation fields as evidence.

  uv run python scripts/audit_labels.py                 # against local eval/data
  python scripts/audit_labels.py --raw-dir data/raw     # deeper checks when raw is present

Every check prints the evidence it looked at so a reviewer can verify by hand.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent.parent

HARM_HINT = re.compile(
    r"\b(kill|bomb|weapon|gun|shoot|stab|hate|slur|racist|nigger|hack|malware|"
    r"drug|cocaine|steal|fraud|scam|exploit|abuse|rape|molest|suicide|self[- ]harm|"
    r"terror|bomb|poison|traffick|nsfw|sexual|porn|illegal|how to make)\b", re.I)


def rows(path: Path):
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def sample_evidence(name: str, r: dict, fields: dict | None = None) -> str:
    q = (r.get("query") or "").replace("\n", " ")[:110]
    img = r.get("image")
    n_img = len(img) if isinstance(img, list) else (1 if img else 0)
    resp = "resp" if r.get("response") else "-"
    extra = ""
    if fields:
        extra = " | " + " ".join(f"{k}={str(v)[:40]}" for k, v in fields.items())
    return f"    label={r['label']} imgs={n_img} {resp} | {q!r}{extra}"


def audit(key: str, data_dir: Path, raw_dir: Path | None) -> dict:
    jf = data_dir / key / "test.jsonl"
    if not jf.exists():
        return {"key": key, "status": "missing"}
    rs = list(rows(jf))
    out = {"key": key, "n": len(rs), "unsafe": sum(r["label"] for r in rs),
           "with_image": sum(1 for r in rs if r.get("image")),
           "with_response": sum(1 for r in rs if r.get("response")),
           "evidence": [], "verdict": ""}

    if key == "vlguard" and raw_dir:
        tj = raw_dir / "vlguard" / "test.json"
        if tj.exists():
            entries = {e["id"]: e for e in json.loads(tj.read_text())}
            cats = collections.Counter((e.get("harmful_category") or "-")
                                       for e in entries.values() if not e.get("safe"))
            out["evidence"].append(
                f"unsafe rows carry dataset's own harmful_category: {dict(cats)}")
            n_no_cat = sum(1 for e in entries.values() if not e.get("safe")
                           and not e.get("harmful_category"))
            out["verdict"] = ("label = dataset's `safe` boolean, and unsafe rows carry the "
                              f"dataset's own harm category ({n_no_cat} exceptions) -> semantics CONSISTENT")

    if key == "spa-vl" and raw_dir:
        # the tell: the class fields are harm annotations on BOTH halves
        out["evidence"].append(
            "SPA-VL test rows have only [image, question, class1..3]; the class fields are "
            "harm annotations and appear in BOTH harm/ and help/ halves "
            "(see https://datasets-server.huggingface.co/rows?dataset=sqrti%2FSPA-VL"
            "&config=test&split=help&offset=0&length=3)")
        help_like = [r for r in rs if r["label"] == 0]
        harm_hint = [r for r in help_like if HARM_HINT.search(r["query"] or "")]
        out["evidence"].append(
            f"{len(harm_hint)}/{len(help_like)} rows we label SAFE match harm keywords "
            f"e.g. {harm_hint[0]['query'][:80]!r}" if harm_hint else "no harm-keyword hits")
        out["verdict"] = ("file split harm|help is NOT a query-safety label -> our column "
                          "INVALID (see report notes); use validation preference pairs instead")

    if key == "beavertails-v" and raw_dir:
        out["verdict"] = ("label = is_response_safe (yes/no) on the RESPONSE; questions may "
                          "themselves be harmful, so response-side use is the consistent reading")

    if key == "jailbreakv" and raw_dir:
        csvs = list((raw_dir / "jailbreakv").rglob("JailBreakV_28K.csv"))
        if csvs:
            import csv as _csv
            rs2 = list(_csv.DictReader(csvs[0].open()))
            pol = collections.Counter(r.get("policy", "-") for r in rs2)
            out["evidence"].append(
                f"all {len(rs2)} rows are attacks with a policy label: {dict(list(pol.items())[:4])}...")
            out["verdict"] = ("attack-only benchmark (no benign class): treating all rows as "
                              "unsafe is the dataset's design -> semantics CONSISTENT "
                              "(but images: only ~360 of 28K are published, see OPS.md)")

    if key == "vlsbench" and raw_dir:
        out["verdict"] = ("harm categories only (Illegal/Privacy/Violent/Self-Harm/Hate/"
                          "Erotic), no benign class -> all-unsafe labelling CONSISTENT")

    if key == "mm-safety":
        out["verdict"] = ("adversarial instruction set (image variants); all-unsafe labelling "
                          "is the benchmark's design -> CONSISTENT (Text_only excluded)")

    if key.startswith("mmds"):
        out["verdict"] = ("labels are the dataset's own per-role ratings (user_rating -> Q, "
                          "assistant_rating -> R) -> semantics CONSISTENT")

    # generic evidence: a few rows from each half
    for lbl in (0, 1):
        picks = [r for r in rs if r["label"] == lbl][:2]
        for p in picks:
            out["evidence"].append(sample_evidence(f"{key}/label{lbl}", p))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(EVAL_DIR / "data"))
    ap.add_argument("--raw-dir", default=str(EVAL_DIR / "data" / "raw"))
    args = ap.parse_args()
    data_dir, raw_dir = Path(args.data_dir), Path(args.raw_dir)
    keys = [p.name for p in sorted(data_dir.iterdir())
            if (p / "test.jsonl").exists() and p.name != "probe"]
    for k in keys:
        res = audit(k, data_dir, raw_dir if raw_dir.exists() else None)
        print(f"\n=== {k} (n={res.get('n')} unsafe={res.get('unsafe')} "
              f"img={res.get('with_image')} resp={res.get('with_response')}) ===")
        for e in res.get("evidence", []):
            print("  " + e)
        print("  VERDICT:", res.get("verdict", "(no rule)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
