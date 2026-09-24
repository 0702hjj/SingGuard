#!/usr/bin/env python
"""Convert RealSafe/LLaVAShield-v1.0-7B (LLaVA-NeXT training format, model_type
"llava" / LlavaQwenForCausalLM, no auto_map) to an HF
``LlavaOnevisionForConditionalGeneration`` checkpoint that vLLM 0.11.0 can serve.

The rename is purely mechanical (no reshaping, no transposes): LLaVA-NeXT stores the
vision tower one level deeper (``vision_tower.vision_tower`` = wrapper + HF model) and
the projector as an ``nn.Sequential`` (``mm_projector.0``/``.2``).

Target config values are copied from the *official* HF conversion of the identical base
checkpoint, ``llava-hf/llava-onevision-qwen2-7b-ov-hf``, which is the reference vLLM 0.11.0
supports.  Two of them are load-bearing and easy to get wrong:

  * ``vision_config.num_hidden_layers = 26``.  The released LLaVA-OneVision vision tower
    really does ship 26 SigLIP layers (0..25), not the 27 of google/siglip-so400m-patch14-384;
    ``lmms-lab/llava-onevision-qwen2-7b-ov``'s own index has exactly the same 765 keys.
  * ``vision_feature_layer = -1`` (not the -2 of the LLaVA-NeXT config).  HF's
    SiglipVisionModel returns ``num_hidden_layers + 1`` hidden states (embeddings + each
    layer's output), so -1 on a 26-layer tower is out(L25) -- the same tensor LLaVA-NeXT's
    select_layer=-2 takes from a 27-layer tower (its hidden_states has 28 entries, so
    [-2] is also out(L25)).  Using -2 here would silently select out(L24): off by one.
  * ``vision_feature_select_strategy = "full"``, not "default".  "default" drops token 0 as
    a CLS token, but SigLIP has no CLS token -- it would drop the first image patch.
  * ``vision_use_head = False`` so the (absent) SiglipMultiheadAttentionPoolingHead is not
    expected.

Usage: python scripts/convert_llavashield_hf.py [--src SRC] [--dst DST]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoTokenizer, LlavaOnevisionConfig, LlavaOnevisionForConditionalGeneration

EVAL = Path(__file__).resolve().parent.parent
DEFAULT_SRC = EVAL / "models" / "llavashield-v1-7b"
DEFAULT_DST = EVAL / "models" / "llavashield-v1-7b-hf"

# --- image_grid_pinpoints, verbatim from the LLaVAShield / LLaVA-OneVision config -------
GRID_PINPOINTS = [
    [384, 384], [384, 768], [384, 1152], [384, 1536], [384, 1920], [384, 2304],
    [768, 384], [768, 768], [768, 1152], [768, 1536], [768, 1920], [768, 2304],
    [1152, 384], [1152, 768], [1152, 1152], [1152, 1536], [1152, 1920], [1152, 2304],
    [1536, 384], [1536, 768], [1536, 1152], [1536, 1536], [1536, 1920], [1536, 2304],
    [1920, 384], [1920, 768], [1920, 1152], [1920, 1536], [1920, 1920], [1920, 2304],
    [2304, 384], [2304, 768], [2304, 1152], [2304, 1536], [2304, 1920], [2304, 2304],
]

CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\n'}}"
    "{# Render all images first #}"
    "{% for content in message['content'] | selectattr('type', 'equalto', 'image') %}"
    "{{ '<image>' }}{% endfor %}"
    "{# Render all video then #}"
    "{% for content in message['content'] | selectattr('type', 'equalto', 'video') %}"
    "{{ '<video>' }}{% endfor %}"
    "{# Render all text next #}"
    "{% if message['role'] != 'assistant' %}"
    "{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}"
    "{{ '\n' + content['text'] }}{% endfor %}"
    "{% else %}"
    "{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}"
    "{% generation %}{{ '\n' + content['text'] }}{% endgeneration %}{% endfor %}"
    "{% endif %}"
    "{{'<|im_end|>'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


def build_config() -> LlavaOnevisionConfig:
    text_config = dict(
        model_type="qwen2",
        architectures=["Qwen2ForCausalLM"],
        hidden_size=3584,
        num_hidden_layers=28,
        num_attention_heads=28,
        num_key_value_heads=4,
        intermediate_size=18944,
        vocab_size=152064,          # LLaVAShield's own vocab (the HF base was resized to 152128)
        max_position_embeddings=32768,
        rope_theta=1000000.0,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        attention_dropout=0.0,
        use_sliding_window=False,
        sliding_window=131072,
        tie_word_embeddings=False,
        bos_token_id=151643,
        eos_token_id=151645,
    )
    vision_config = dict(
        model_type="siglip_vision_model",
        hidden_size=1152,
        intermediate_size=4304,
        num_hidden_layers=26,
        num_attention_heads=16,
        image_size=384,
        patch_size=14,
        layer_norm_eps=1e-6,
        hidden_act="gelu_pytorch_tanh",
        vision_use_head=False,
    )
    return LlavaOnevisionConfig(
        text_config=text_config,
        vision_config=vision_config,
        image_token_index=151646,
        video_token_index=151647,
        vision_feature_layer=-1,
        vision_feature_select_strategy="full",
        image_grid_pinpoints=GRID_PINPOINTS,
        vision_aspect_ratio="anyres_max_9",
        projector_hidden_act="gelu",
        multimodal_projector_bias=True,
        image_seq_length=729,
        ignore_index=-100,
        tie_word_embeddings=False,
        bos_token_id=151643,
        eos_token_id=151645,
    )


def map_key(k: str) -> str | None:
    """LLaVA-NeXT key -> transformers>=4.52 LlavaOnevisionForConditionalGeneration key."""
    if k == "lm_head.weight":
        return "lm_head.weight"
    if k.startswith("model.vision_tower.vision_tower."):
        # drop ONE level: LLaVA's CLIPVisionTower wrapper + the HF SiglipVisionModel
        return "model.vision_tower." + k[len("model.vision_tower.vision_tower."):]
    if k.startswith("model.mm_projector.0."):
        return "model.multi_modal_projector.linear_1." + k[len("model.mm_projector.0."):]
    if k.startswith("model.mm_projector.2."):
        return "model.multi_modal_projector.linear_2." + k[len("model.mm_projector.2."):]
    if k == "model.image_newline":
        return "model.image_newline"
    if k.startswith("model."):
        # embed_tokens / layers.N.* / norm.*  ->  under language_model
        return "model.language_model." + k[len("model."):]
    return None


def plan(src: Path, cfg: LlavaOnevisionConfig, model: torch.nn.Module):
    """Key-level reconciliation: no target left unfilled, no source left unmapped."""
    idx = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    src_keys = set(idx)
    mapping = {}
    unmapped = []
    for k in sorted(src_keys):
        t = map_key(k)
        if t is None:
            unmapped.append(k)
        else:
            mapping[k] = t

    tgt_shapes = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    bad_shape = []
    for s, t in mapping.items():
        if t not in tgt_shapes:
            bad_shape.append((s, t, "no such target key"))
    # every target key must be filled, except ones this architecture does not ship
    filled = set(mapping.values())
    unfilled = sorted(set(tgt_shapes) - filled)
    return idx, mapping, unmapped, bad_shape, unfilled, tgt_shapes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--dst", type=Path, default=DEFAULT_DST)
    ap.add_argument("--shard-size", default="5GB")
    args = ap.parse_args()
    src, dst = args.src, args.dst

    cfg = build_config()
    print(f"building {type(cfg).__name__} (vision {cfg.vision_config.num_hidden_layers} layers,"
          f" feature_layer={cfg.vision_feature_layer}, select={cfg.vision_feature_select_strategy})")
    model = LlavaOnevisionForConditionalGeneration._from_config(cfg, dtype=torch.bfloat16)
    model.eval()
    got = next(model.parameters()).dtype
    if got != torch.bfloat16:                      # fall back for older _from_config
        print(f"  _from_config gave {got}; casting")
        model = model.to(torch.bfloat16)
    n_tgt = len(model.state_dict())
    print(f"  target state_dict: {n_tgt} tensors")

    idx, mapping, unmapped, bad_shape, unfilled, tgt_shapes = plan(src, cfg, model)
    print(f"source tensors: {len(idx)}   mapped: {len(mapping)}")
    if unmapped:
        print(f"!! UNMAPPED SOURCE KEYS ({len(unmapped)}):")
        for k in unmapped[:20]:
            print("   ", k)
    if bad_shape:
        print(f"!! SOURCE KEYS WITH NO TARGET ({len(bad_shape)}):")
        for s, t, why in bad_shape[:20]:
            print(f"    {s} -> {t} ({why})")
    if unfilled:
        print(f"!! TARGET KEYS LEFT UNFILLED ({len(unfilled)}):")
        for k in unfilled[:20]:
            print("   ", k, tgt_shapes[k])
    if unmapped or bad_shape or unfilled:
        print("refusing to continue: mapping is not a bijection")
        return 1
    print("mapping OK: every source tensor maps to a target, every target is filled")

    # ---- stream the weights in, one source shard at a time (keeps peak RSS ~ source shard)
    by_file: dict[str, list[str]] = {}
    for k, f in idx.items():
        by_file.setdefault(f, []).append(k)
    for f in sorted(by_file):
        chunk = {}
        with safe_open(src / f, framework="pt", device="cpu") as fh:
            for k in by_file[f]:
                t = fh.get_tensor(k)
                want = tgt_shapes[mapping[k]]
                if tuple(t.shape) != want:
                    print(f"!! shape mismatch {k} {tuple(t.shape)} -> {mapping[k]} {want}")
                    return 1
                chunk[mapping[k]] = t
        missing, unexpected = model.load_state_dict(chunk, strict=False)
        if missing or unexpected:
            print(f"!! {f}: missing={missing[:5]} unexpected={unexpected[:5]}")
            return 1
        print(f"  loaded {f} ({len(chunk)} tensors)")
        del chunk

    left = [k for k, p in model.state_dict().items() if k in tgt_shapes]
    print(f"all {n_tgt} target tensors loaded")

    # ---- write the checkpoint
    dst.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(dst, max_shard_size=args.shard_size, safe_serialization=True)
    print(f"saved weights -> {dst}")

    # ---- tokenizer: LLaVAShield's own Qwen2 tokenizer, plus <video> (id 151647) so we
    #      match llava-hf/llava-onevision-qwen2-7b-ov-hf exactly.  The inline Qwen2 chat
    #      template is REMOVED: it has no <image> marker, so the image would be dropped.
    tok = AutoTokenizer.from_pretrained(src)
    tok.save_pretrained(dst)
    _add_video_token(dst)
    tc = json.loads((dst / "tokenizer_config.json").read_text())
    tc["chat_template"] = None
    tc["processor_class"] = "LlavaOnevisionProcessor"
    tc["model_max_length"] = 16384
    (dst / "tokenizer_config.json").write_text(json.dumps(tc, indent=2) + "\n")
    (dst / "chat_template.json").write_text(
        json.dumps({"chat_template": CHAT_TEMPLATE}, indent=2) + "\n")

    # ---- image processor + processor config, verbatim from the official HF conversion
    (dst / "preprocessor_config.json").write_text(json.dumps({
        "do_convert_rgb": True, "do_normalize": True, "do_pad": True, "do_rescale": True,
        "do_resize": True, "image_grid_pinpoints": GRID_PINPOINTS,
        "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
        "image_processor_type": "LlavaOnevisionImageProcessor",
        "processor_class": "LlavaOnevisionProcessor", "resample": 3,
        "rescale_factor": 0.00392156862745098, "size": {"height": 384, "width": 384},
    }, indent=2) + "\n")
    (dst / "processor_config.json").write_text(json.dumps({
        "image_token": "<image>", "num_image_tokens": 729,
        "processor_class": "LlavaOnevisionProcessor", "video_token": "<video>",
        "vision_feature_select_strategy": "full",
    }, indent=2) + "\n")
    (dst / "generation_config.json").write_text(json.dumps({
        "_from_model_config": True, "bos_token_id": 151643, "eos_token_id": 151645,
    }, indent=2) + "\n")
    for f in ("merges.txt", "vocab.txt"):
        if (src / f).exists():
            shutil.copyfile(src / f, dst / f)
    (dst / ".complete").write_text("")
    print(f"wrote config/tokenizer/processor files -> {dst}")
    return 0


def _add_video_token(dst: Path) -> None:
    """Register <video> = 151647 (as llava-hf/llava-onevision-qwen2-7b-ov-hf does)."""
    entry = {"content": "<video>", "lstrip": False, "normalized": False,
             "rstrip": False, "single_word": False, "special": True}
    tj = dst / "tokenizer.json"
    d = json.loads(tj.read_text())
    if not any(t.get("content") == "<video>" for t in d.get("added_tokens", [])):
        d.setdefault("added_tokens", []).append({"id": 151647, **entry})
        tj.write_text(json.dumps(d))
    for name, key in (("added_tokens.json", None), ("special_tokens_map.json", None)):
        p = dst / name
        if p.exists():
            j = json.loads(p.read_text())
            if name == "added_tokens.json":
                j["<video>"] = 151647
            p.write_text(json.dumps(j, indent=2) + "\n")


if __name__ == "__main__":
    sys.exit(main())
