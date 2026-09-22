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

def download_raw(hf_id: str, raw_dir: Path) -> None:
    if any(p for p in raw_dir.rglob("*") if p.is_file()):
        print(f"[skip] {raw_dir} already populated")
        return
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from huggingface_hub import snapshot_download
    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"[download] {hf_id} -> {raw_dir}")
    snapshot_download(hf_id, repo_type="dataset", local_dir=str(raw_dir))


# ----------------------------------------------------------------- generic helpers

def materialize_image(val, cache: Path, name: str, raw_dir: Path) -> str | None:
    """HF datasets represent images in several ways; turn any of them into a file on disk.
    Returns path relative to DATA_DIR (or None)."""
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
        return 0 if val else 1          # safe=True -> 0
    if isinstance(val, (int, float)):
        v = int(val)
        if v in (0, 1):
            return v                    # assume 1=unsafe
        raise SchemaError(f"ambiguous numeric label {val}")
    s = str(val).strip().lower()
    if s in ("unsafe", "1", "true_unsafe", "bad"):
        return 1
    if s in ("safe", "0", "false", "good"):
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
    """VLGuard test (1,000): original repo ships per-task JSON files whose names carry the
    label (text_query_safe.json / image_unsafe.json / ...); the HF parquet variant carries a
    messages column. Handle both."""
    cache = DATA_DIR / "extracted" / key
    rows = list(iter_rows(raw_dir))
    out = []
    if rows and rows[0][0].get("messages"):
        for i, (row, src) in enumerate(rows):
            label = pick(row, LABEL_COLS)
            if label is None:
                raise SchemaError(
                    "VLGuard parquet has no label column; inspect with inspect_data.py "
                    "and fall back to the original JSON-file layout")
            label = parse_label(label)
            msgs = row["messages"]
            query = next((c.get("text") for m in msgs if m["role"] == "user"
                          for c in m["content"] if c.get("type") == "text"), "")
            response = next((c.get("text") for m in msgs if m["role"] == "assistant"
                             for c in m["content"] if c.get("type") == "text"), None)
            img = materialize_image(pick(row, ["images", "image"]) or
                                    next((c for m in msgs for c in m["content"]
                                          if c.get("type") == "image"), None),
                                    cache, f"vlguard_{i}", raw_dir)
            out.append({"image": img, "query": query or "", "response": response,
                        "label": label, "src": src})
    else:
        # original layout: test/<task>_<label>.json + img/ referenced inside entries
        for jf in sorted(raw_dir.rglob("*.json")):
            name = jf.name.lower()
            if "_safe" not in name and "_unsafe" not in name:
                continue
            label = 1 if "unsafe" in name else 0
            try:
                entries = json.loads(jf.read_text())
            except Exception:  # noqa: BLE001
                continue
            for i, e in enumerate(entries):
                msgs = e.get("messages", [])
                query = next((c.get("text") for m in msgs if m.get("role") == "user"
                              for c in m.get("content", []) if c.get("type") == "text"), "")
                response = next((c.get("text") for m in msgs if m.get("role") == "assistant"
                                 for c in m.get("content", []) if c.get("type") == "text"), None)
                img_rel = (e.get("images") or [None])[0] or \
                    next((c.get("path") for m in msgs for c in m.get("content", [])
                          if c.get("type") == "image"), None)
                img = None
                if img_rel:
                    cand = [raw_dir / img_rel, raw_dir / "img" / img_rel,
                            raw_dir / "img" / "test" / Path(img_rel).name,
                            raw_dir / Path(img_rel).name]
                    img = next((str(c.relative_to(DATA_DIR)) for c in cand if c.exists()), None)
                out.append({"image": img, "query": query or "", "response": response,
                            "label": label, "src": jf.name})
    if not out:
        raise SchemaError("VLGuard: no rows parsed")
    return out


def load_jailbreakv(raw_dir: Path, key: str):
    """JailBreakV-28K: CSV of jailbreak queries (+ image paths for SD/FigStep/MEME sources).
    All entries are adversarial queries -> gold unsafe; if a label column exists, use it."""
    rows = list(iter_rows(raw_dir))
    if not rows:
        raise SchemaError("JailBreakV: no CSV rows found")
    out = []
    for i, (row, src) in enumerate(rows):
        q = pick(row, QUERY_COLS)
        if not q:
            raise SchemaError(f"JailBreakV: no query column in {row.keys()}")
        label = pick(row, LABEL_COLS)
        img = materialize_image(pick(row, ["image", "image_path", "img_path", "image_file"]),
                                DATA_DIR / "extracted" / key, f"jbv_{i}", raw_dir)
        out.append({"image": img, "query": str(q), "response": None,
                    "label": parse_label(label) if label is not None else 1, "src": src})
    n_pos = sum(r["label"] for r in out)
    if n_pos != len(out):
        print(f"  [note] JailBreakV: {len(out) - n_pos}/{len(out)} rows are labeled safe "
              f"(paper likely evaluates all-unsafe; verify against inspect_data.py)")
    return out


def load_spavl(raw_dir: Path, key: str):
    """SPA-VL: preference pairs; used query-side. Label must come from an explicit column
    (the paper's F1 range implies a mixed label distribution)."""
    rows = list(iter_rows(raw_dir))
    if not rows:
        raise SchemaError("SPA-VL: no rows found")
    out = []
    for i, (row, src) in enumerate(rows):
        q = pick(row, QUERY_COLS)
        label = pick(row, LABEL_COLS)
        if label is None:
            raise SchemaError(
                f"SPA-VL row has no label column (cols={list(row.keys())}). Inspect the raw "
                "data, then either map the right column in LABEL_COLS or (if the split really "
                "is all-unsafe) default label=1 with a printed warning.")
        img = materialize_image(pick(row, ["image", "images", "image_path"]),
                                DATA_DIR / "extracted" / key, f"spa_{i}", raw_dir)
        out.append({"image": img, "query": str(q or ""), "response": None,
                    "label": parse_label(label), "src": src})
    return out


def load_vlsbench(raw_dir: Path, key: str):
    """VLSBench (2,241): image + query + safe/unsafe label."""
    rows = list(iter_rows(raw_dir))
    if not rows:
        raise SchemaError("VLSBench: no rows found")
    out = []
    for i, (row, src) in enumerate(rows):
        q = pick(row, QUERY_COLS)
        label = pick(row, LABEL_COLS)
        if label is None:
            raise SchemaError(f"VLSBench: no label column (cols={list(row.keys())})")
        img = materialize_image(pick(row, ["image", "images", "image_path"]),
                                DATA_DIR / "extracted" / key, f"vls_{i}", raw_dir)
        out.append({"image": img, "query": str(q or ""), "response": None,
                    "label": parse_label(label), "src": src})
    return out


def load_mmsafety(raw_dir: Path, key: str):
    """MM-SafetyBench (5,040): harmful queries with SD/TYPO/SD+OCR images. Queries are
    adversarial -> gold unsafe unless a label column says otherwise."""
    rows = list(iter_rows(raw_dir))
    if not rows:
        raise SchemaError(
            "MM-SafetyBench: nothing parsed. If the HF repo is unavailable, download from the "
            "GitHub release (isXinLiu/MM-SafetyBench) and drop the unzipped folders into "
            "data/raw/mm-safety/, then re-run with --no-download.")
    out = []
    for i, (row, src) in enumerate(rows):
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
        is_safe = pick(row, ["is_safe"])
        if safe_r and unsafe_r:
            out.append({**base, "response": str(safe_r), "label": 0})
            out.append({**base, "response": str(unsafe_r), "label": 1})
        elif pick(row, RESPONSE_COLS) is not None and is_safe is not None:
            r = pick(row, RESPONSE_COLS)
            out.append({**base, "response": str(r), "label": 0 if str(is_safe).lower() in
                        ("true", "1") else 1})
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
        take[lbl] = min(len(group), round(n * len(group) / total))
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
                "seed": seed, "n_unsafe": sum(r["label"] for r in picked)}
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
                download_raw(c["hf_id"], raw_dir)
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
