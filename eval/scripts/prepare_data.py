#!/usr/bin/env python
"""Download + normalize + stratified-sample the 8 Table-4 datasets.

Output per dataset: data/<key>/test.jsonl with rows
    {"id", "image": <path relative to eval/data/> | null,
     "query": str, "response": str|null, "label": 0(safe)|1(unsafe)}
plus data/<key>/manifest.json (counts, seed, source).

IMPORTANT: the loaders were written from each dataset's public schema *without* a local
download. After `--download`, run scripts/inspect_data.py <key> first; loaders raise loud
SchemaError instead of guessing when a field is missing, so adapting them on the server is
a 2-minute fix in one place.

Usage:
  uv run python scripts/prepare_data.py                      # all datasets
  uv run python scripts/prepare_data.py --datasets vlguard,vlsbench
  uv run python scripts/prepare_data.py --no-download        # re-normalize only
  uv run python scripts/prepare_data.py --sample 500 --seed 42
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import pandas as pd
import yaml

EVAL_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EVAL_DIR))

DATA_DIR = EVAL_DIR / "data"
QUERY_COLS = ["query", "question", "prompt", "jailbreak_query", "instruction",
              "text", "query_text", "target", "conv"]
RESPONSE_COLS = ["response", "answer", "response_safe", "response_unsafe",
                 "assistant_response", "reply"]
LABEL_COLS = ["label", "safe", "is_safe", "safety", "query_label", "pred_label",
              "is_unsafe", "y"]


class SchemaError(RuntimeError):
    pass


def load_datasets_cfg() -> dict:
    return yaml.safe_load((EVAL_DIR / "configs" / "datasets.yaml").read_text())


# ----------------------------------------------------------------- download

def download_raw(hf_id: str, raw_dir: Path, allow_patterns: list[str] | None = None) -> None:
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # xet CDN transfers proved flaky
    from huggingface_hub import snapshot_download
    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"[download] {hf_id} -> {raw_dir}")
    # snapshot_download resumes: existing complete files are kept, missing ones fetched
    snapshot_download(hf_id, repo_type="dataset", local_dir=str(raw_dir),
                      allow_patterns=allow_patterns, max_workers=4)


# ----------------------------------------------------------------- generic helpers

IMAGE_FAILURES: list[str] = []   # names whose image could not be materialized


def materialize_image(val, cache: Path, name: str, raw_dir: Path) -> str | None:
    """HF datasets represent images in several ways; turn any of them into a file on disk.
    Returns path relative to DATA_DIR (or None). Failures are counted in IMAGE_FAILURES
    so manifests can show how many rows silently became text-only."""
    cache.mkdir(parents=True, exist_ok=True)
    try:
        if val in (None, "", []):
            return None
        if isinstance(val, dict):
            if val.get("bytes"):
                ext = Path(str(val.get("path") or "")).suffix or ".jpg"
                dest = cache / f"{name}{ext}"
                dest.write_bytes(val["bytes"])
                return str(dest.relative_to(DATA_DIR))
            if val.get("path"):
                p = raw_dir / val["path"]
                if p.exists():
                    return str(p.relative_to(DATA_DIR))
                return None
        if isinstance(val, (bytes, bytearray)):
            dest = cache / f"{name}.jpg"
            dest.write_bytes(val)
            return str(dest.relative_to(DATA_DIR))
        if isinstance(val, str):
            p = raw_dir / val
            if p.exists():
                return str(p.relative_to(DATA_DIR))
            return None
        if hasattr(val, "save"):  # PIL image
            dest = cache / f"{name}.png"
            val.save(dest)
            return str(dest.relative_to(DATA_DIR))
        if isinstance(val, (list, tuple)) and val:
            return materialize_image(val[0], cache, name, raw_dir)
    except Exception as e:  # noqa: BLE001
        IMAGE_FAILURES.append(name)
        print(f"  [warn] image materialization failed for {name}: {e}")
    return None


def pick(row: dict, cols: list[str]):
    for c in cols:
        if c in row and row[c] not in (None, "", [], {}):
            return row[c]
    for c in row:
        for want in cols:
            if c.lower() == want.lower() and row[c] not in (None, "", [], {}):
                return row[c]
    return None


def parse_label(val) -> int:
    if isinstance(val, bool):
        return 0 if val else 1          # safe-polarity columns: True/"yes" -> safe
    if isinstance(val, (int, float)):
        v = int(val)
        if v in (0, 1):
            return v                    # assume 1=unsafe
        raise SchemaError(f"ambiguous numeric label {val}")
    s = str(val).strip().lower()
    if s in ("unsafe", "1", "true_unsafe", "bad", "no", "false"):
        return 1
    if s in ("safe", "0", "good", "yes", "true"):
        return 0
    raise SchemaError(f"unrecognized label value {val!r}")


def iter_rows(raw_dir: Path):
    """Yield dict rows from the first readable source: parquet > csv > json(jsonl)."""
    parquets = sorted(raw_dir.rglob("*.parquet"))
    if parquets:
        for p in parquets:
            if "train" in p.name.lower() and any("test" in q.name.lower() or "val" in q.name.lower()
                                                 for q in parquets):
                continue  # prefer test/val splits when present
            for row in pd.read_parquet(p).to_dict("records"):
                yield row, p.name
        return
    csvs = sorted(raw_dir.rglob("*.csv"))
    if csvs:
        for p in csvs:
            if "train" in p.name.lower() and any("test" in q.name.lower() for q in csvs):
                continue
            for row in pd.read_csv(p).fillna("").to_dict("records"):
                yield row, p.name
        return
    jsons = sorted(raw_dir.rglob("*.json*"))
    for p in jsons:
        if p.name == "manifest.json":
            continue
        try:
            obj = json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            continue
        rows = obj if isinstance(obj, list) else obj.get("data", [])
        for row in rows:
            yield row, p.name


# ----------------------------------------------------------------- loaders

def load_vlguard(raw_dir: Path, key: str):
    """VLGuard test (1,000), evaluated QUERY-SIDE. Actual HF layout (verified): test.json
    (flat list with id/image/safe/instr-resp) + test.zip -> test_extracted/test/<image>.

    IMPORTANT: the response is deliberately dropped. VLGuard gold-unsafe items pair a
    harmful instruction with a *safe refusal* response; SingGuard's documented behavior
    (model card: "Refusals and safe redirections can be classified as safe") then judges the
    conversation safe and recall collapses to ~0 (empirically verified). The labels refer to
    the image+instruction side, so this is a query-side benchmark (paper Sec 4.1: query-side
    vs response-side benchmarks are evaluated separately)."""
    out = []
    tj = raw_dir / "test.json"
    img_root = raw_dir / "test_extracted" / "test"
    if tj.exists() and img_root.exists():
        entries = json.loads(tj.read_text())
        for e in entries:
            pair = (e.get("instr-resp") or [{}])[0]
            q = pair.get("instruction") or pair.get("safe_instruction") or ""
            imgp = img_root / e["image"]
            img = str(imgp.relative_to(DATA_DIR)) if imgp.exists() else None
            out.append({"image": img, "query": str(q), "response": None,
                        "label": 0 if e.get("safe") else 1, "src": "test.json"})
        n_pos = sum(r["label"] for r in out)
        n_img = sum(1 for r in out if r["image"])
        print(f"  [note] VLGuard (query-side): {n_pos}/{len(out)} unsafe, {n_img} with image")
        if not out:
            raise SchemaError("VLGuard: test.json parsed to 0 rows")
        return out

    # fallback: parquet-with-messages or *_safe/*_unsafe.json legacy layouts
    cache = DATA_DIR / "extracted" / key
    rows = list(iter_rows(raw_dir))
    if rows and isinstance(rows[0][0], dict) and rows[0][0].get("messages"):
        for i, (row, src) in enumerate(rows):
            label = pick(row, LABEL_COLS)
            if label is None:
                raise SchemaError("VLGuard parquet has no label column")
            msgs = row["messages"]
            query = next((c.get("text") for m in msgs if m["role"] == "user"
                          for c in m["content"] if c.get("type") == "text"), "")
            response = next((c.get("text") for m in msgs if m["role"] == "assistant"
                             for c in m["content"] if c.get("type") == "text"), None)
            img = materialize_image(pick(row, ["images", "image"]),
                                    cache, f"vlguard_{i}", raw_dir)
            out.append({"image": img, "query": query or "", "response": response,
                        "label": parse_label(label), "src": src})
    if not out:
        raise SchemaError("VLGuard: no rows parsed (unzip test.zip first)")
    return out


def load_jailbreakv(raw_dir: Path, key: str):
    """JailBreakV-28K. Full csv = JailBreakV_28K/JailBreakV_28K.csv (28,000 adversarial
    queries, gold unsafe). NOTE: the HF repo only hosts ~100 images per source dir (~300
    total); the rest live on Google Drive in the original release. We force-include every
    image-resolvable row and fill up with a deterministic text-only selection so the 1,000
    sample stays multimodal. mini_/RedTeam_2K csvs are auxiliary and excluded."""
    csv_path = raw_dir / "JailBreakV_28K" / "JailBreakV_28K.csv"
    if not csv_path.exists():
        raise SchemaError(f"JailBreakV: {csv_path} missing")
    img_base = raw_dir / "JailBreakV_28K"
    with_img, text_only = [], []
    for row in csv.DictReader(csv_path.open()):
        q = row.get("jailbreak_query") or row.get("query") or ""
        if not q:
            continue
        img = None
        rel = (row.get("image_path") or "").strip()
        if rel and (img_base / rel).exists():
            img = str((img_base / rel).relative_to(DATA_DIR))
        (with_img if img else text_only).append(
            {"image": img, "query": q, "response": None, "label": 1, "src": "JailBreakV_28K.csv"})
    print(f"  [note] JailBreakV: {len(with_img)} rows with local image, "
          f"{len(text_only)} text-only (HF hosts only ~300 of the 28K images)")
    rng = random.Random(42)
    fill = rng.sample(text_only, min(700, len(text_only)))
    return with_img + fill


def load_spavl(raw_dir: Path, key: str):
    """SPA-VL test split: harm-*.parquet (unsafe queries) + help-*.parquet (safe queries).
    Label comes from the filename; an explicit label column wins if present."""
    rows = list(iter_rows(raw_dir))
    if not rows:
        raise SchemaError("SPA-VL: no rows found (did the test/* download complete?)")
    out = []
    for i, (row, src) in enumerate(rows):
        q = pick(row, QUERY_COLS)
        label = pick(row, LABEL_COLS)
        if label is None:
            low = str(src).lower()
            if "harm" in low:
                label = 1
            elif "help" in low:
                label = 0
            else:
                raise SchemaError(
                    f"SPA-VL: no label column and filename {src!r} is neither harm/help")
        img = materialize_image(pick(row, ["image", "images", "image_path"]),
                                DATA_DIR / "extracted" / key, f"spa_{i}", raw_dir)
        out.append({"image": img, "query": str(q or ""), "response": None,
                    "label": parse_label(label), "src": str(src)})
    n_pos = sum(r["label"] for r in out)
    print(f"  [note] SPA-VL: {n_pos}/{len(out)} unsafe (from harm/help files)")
    return out


def load_vlsbench(raw_dir: Path, key: str):
    """VLSBench (2,247 rows): visual jailbreak attacks across 6 harm categories, all
    gold-unsafe (verified: no benign class in the release). Query = instruction."""
    rows = list(iter_rows(raw_dir))
    if not rows:
        raise SchemaError("VLSBench: no rows found")
    out = []
    for i, (row, src) in enumerate(rows):
        q = pick(row, QUERY_COLS)
        if not q:
            continue
        label = pick(row, LABEL_COLS)
        img = materialize_image(pick(row, ["image", "images"]),
                                DATA_DIR / "extracted" / key, f"vls_{i}", raw_dir)
        out.append({"image": img, "query": str(q), "response": None,
                    "label": parse_label(label) if label is not None else 1, "src": str(src)})
    n_pos = sum(r["label"] for r in out)
    print(f"  [note] VLSBench: {n_pos}/{len(out)} unsafe (all-attack benchmark)")
    return out


def load_mmsafety(raw_dir: Path, key: str):
    """MM-SafetyBench: harmful queries with SD/TYPO/SD+TYPO images, all gold-unsafe.
    The release also ships a Text_only ablation variant per topic; the paper's scale
    (5,040 = 13 topics x 3 image variants x ~129) shows the benchmark is the image
    variants only, so Text_only rows are excluded here."""
    rows = list(iter_rows(raw_dir))
    if not rows:
        raise SchemaError(
            "MM-SafetyBench: nothing parsed. If the HF repo is unavailable, download from the "
            "GitHub release (isXinLiu/MM-SafetyBench) and drop the unzipped folders into "
            "data/raw/mm-safety/, then re-run with --no-download.")
    out = []
    for i, (row, src) in enumerate(rows):
        if "Text_only" in str(src):
            continue
        q = pick(row, QUERY_COLS)
        if not q:
            continue
        label = pick(row, LABEL_COLS)
        img = materialize_image(pick(row, ["image", "image_path", "img_path"]),
                                DATA_DIR / "extracted" / key, f"mms_{i}", raw_dir)
        out.append({"image": img, "query": str(q), "response": None,
                    "label": parse_label(label) if label is not None else 1, "src": src})
    if not out:
        raise SchemaError("MM-SafetyBench: no usable rows")
    return out


def load_beavertailsv(raw_dir: Path, key: str):
    """BeaverTails-V (1,180): preference-style pairs -> expand each prompt into a safe and an
    unsafe response-side sample; judge (query, image, response) jointly."""
    rows = list(iter_rows(raw_dir))
    if not rows:
        raise SchemaError(
            "BeaverTails-V: nothing parsed. Check the HF repo id in datasets.yaml "
            "(PKU-Alignment/BeaverTails-V) or place files under data/raw/beavertails-v/.")
    out = []
    for i, (row, src) in enumerate(rows):
        q = pick(row, QUERY_COLS + ["prompt"])
        img = materialize_image(pick(row, ["image", "images", "image_path"]),
                                DATA_DIR / "extracted" / key, f"btv_{i}", raw_dir)
        base = {"image": img, "query": str(q or ""), "src": src}
        safe_r = pick(row, ["response_safe", "safe_response", "chosen"])
        unsafe_r = pick(row, ["response_unsafe", "unsafe_response", "rejected"])
        is_safe = pick(row, ["is_safe", "is_response_safe"])
        if safe_r and unsafe_r:
            out.append({**base, "response": str(safe_r), "label": 0})
            out.append({**base, "response": str(unsafe_r), "label": 1})
        elif pick(row, RESPONSE_COLS) is not None and is_safe is not None:
            r = pick(row, RESPONSE_COLS)
            out.append({**base, "response": str(r), "label": parse_label(is_safe)})
        else:
            raise SchemaError(
                f"BeaverTails-V row lacks response pairs / is_safe (cols={list(row.keys())})")
    return out


def load_mmds(raw_dir: Path, key: str):
    """MMDS: manual placement. Put a normalized CSV at data/raw/mmds/mmds_normalized.csv with
    columns: query, image_path, response, label, side  (side in {q, r}). Emits BOTH
    data/mmds-q/ and data/mmds-r/ test sets; image_path is relative to data/raw/mmds/."""
    src = raw_dir / "mmds_normalized.csv"
    if not src.exists():
        print(f"[mmds] {src} not found -- MMDS-Q / MMDS-R columns will be skipped.\n"
              "       Get MMDS from the LLaVAShield project (https://leost123456.github.io),\n"
              "       normalize it into that CSV, then re-run prepare_data.py --datasets mmds-q,mmds-r")
        return {}
    df = pd.read_csv(src).fillna("")
    by_side: dict[str, list] = {"q": [], "r": []}
    for i, row in df.iterrows():
        side = str(row.get("side", "q")).lower().strip()
        if side not in by_side:
            continue
        img = None
        if str(row.get("image_path", "")):
            cand = raw_dir / str(row["image_path"])
            img = str(cand.relative_to(DATA_DIR)) if cand.exists() else None
        by_side[side].append({"image": img, "query": str(row.get("query", "")),
                              "response": str(row["response"]) if str(row.get("response", "")) else None,
                              "label": parse_label(row["label"]), "src": "mmds_manual"})
    return {"q": by_side["q"], "r": by_side["r"]}


LOADERS = {
    "vlguard": load_vlguard,
    "jailbreakv": load_jailbreakv,
    "spavl": load_spavl,
    "vlsbench": load_vlsbench,
    "mmsafety": load_mmsafety,
    "beavertailsv": load_beavertailsv,
    "mmds": load_mmds,
}


# ----------------------------------------------------------------- sampling / writing

def stratified_sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    if n >= len(rows):
        return sorted(rows, key=lambda r: str(r.get("id", "")))  # stable
    rng = random.Random(seed)
    by_label: dict[int, list] = {}
    for r in rows:
        by_label.setdefault(r["label"], []).append(r)
    for v in by_label.values():
        rng.shuffle(v)
    take = {}
    total = len(rows)
    for lbl, group in by_label.items():
        # proportionally allocate, but never drop a non-empty class to zero
        take[lbl] = min(len(group), max(1, round(n * len(group) / total)))
    # fix rounding drift toward the largest class
    while sum(take.values()) > n:
        largest = max(by_label, key=lambda l: len(by_label[l]))
        if take[largest] > 0:
            take[largest] -= 1
    while sum(take.values()) < n:
        room = [l for l, g in by_label.items() if take[l] < len(g)]
        if not room:
            break
        take[rng.choice(room)] += 1
    picked = [r for lbl, group in by_label.items() for r in group[:take[lbl]]]
    return picked


def write_split(rows: list[dict], out_dir: Path, key: str, sample: int, seed: int,
                id_prefix: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    picked = stratified_sample(rows, sample, seed)
    # deterministic shuffle so `runner.py --limit N` takes a balanced-ish prefix
    random.Random(seed + 1).shuffle(picked)
    with (out_dir / "test.jsonl").open("w") as f:
        for i, r in enumerate(picked):
            f.write(json.dumps({
                "id": f"{id_prefix}-{i:05d}",
                "image": r.get("image"),
                "query": r["query"],
                "response": r.get("response"),
                "label": r["label"],
            }, ensure_ascii=False) + "\n")
    manifest = {"dataset": key, "source_rows": len(rows), "sampled": len(picked),
                "seed": seed, "n_unsafe": sum(r["label"] for r in picked),
                "n_with_image": sum(1 for r in picked if r.get("image")),
                "n_image_failures": len(IMAGE_FAILURES)}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[ok] {key}: {len(picked)} samples "
          f"({manifest['n_unsafe']} unsafe) -> {out_dir / 'test.jsonl'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="", help="comma-separated keys; default = all")
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = load_datasets_cfg()
    keys = [k for k in args.datasets.split(",") if k] or list(cfg.keys())
    unknown = [k for k in keys if k not in cfg]
    if unknown:
        print(f"unknown dataset keys: {unknown}", file=sys.stderr)
        return 2

    DATA_DIR.mkdir(exist_ok=True)
    rc = 0
    for k in keys:
        c = cfg[k]
        if k == "mmds-r":        # handled together with mmds-q (manual loader writes both)
            continue
        sample = args.sample if args.sample is not None else c.get("sample", 1000)
        seed = args.seed if args.seed is not None else c.get("seed", 42)
        raw_dir = DATA_DIR / "raw" / k
        if c.get("loader") == "mmds":
            if not args.no_download:
                pass  # manual dataset; nothing to fetch
            sides = load_mmds(raw_dir, k) or {}
            if "q" in sides:
                write_split(sides["q"], DATA_DIR / "mmds-q", "mmds-q", sample, seed, "mmdsq")
                write_split(sides["r"], DATA_DIR / "mmds-r", "mmds-r", sample, seed, "mmdsr")
            continue
        if not args.no_download:
            try:
                download_raw(c["hf_id"], raw_dir, c.get("allow_patterns"))
            except Exception as e:  # noqa: BLE001
                print(f"[FAIL] {k}: download {c['hf_id']}: {e}", file=sys.stderr)
                rc = 1
                continue
        try:
            rows = LOADERS[c["loader"]](raw_dir, k)
            write_split(rows, DATA_DIR / k, k, sample, seed, k.replace("-", ""))
        except SchemaError as e:
            print(f"[schema] {k}: {e}\n         run: uv run python scripts/inspect_data.py {k}",
                  file=sys.stderr)
            rc = 1
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] {k}: {e}", file=sys.stderr)
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
