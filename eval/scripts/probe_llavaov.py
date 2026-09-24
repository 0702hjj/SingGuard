"""Probe HF LlavaOnevision structure/conventions before converting LLaVAShield.

Run with the eval venv on the GPU box. Prints:
  1. vision-tower hidden_states convention (length vs num_hidden_layers)
  2. LlavaOnevisionForConditionalGeneration state_dict keys + shapes
  3. LlavaOnevisionConfig field list
"""
import json
import sys

import torch
from transformers import (
    AutoConfig,
    AutoModel,
    LlavaOnevisionConfig,
    LlavaOnevisionForConditionalGeneration,
)

CKPT = "/mnt/data/intern3/research/SingGuard/eval/models/llavashield-v1-7b"

SIGLIP_SO400M_384 = dict(
    hidden_size=1152,
    intermediate_size=4304,
    num_hidden_layers=27,
    num_attention_heads=16,
    image_size=384,
    patch_size=14,
    layer_norm_eps=1e-6,
    hidden_act="gelu_pytorch_tanh",
)


def build_config(vision_layers=27, vision_feature_layer=-2):
    text = dict(
        model_type="qwen2",
        hidden_size=3584,
        num_hidden_layers=28,
        num_attention_heads=28,
        num_key_value_heads=4,
        intermediate_size=18944,
        vocab_size=152064,
        rope_theta=1000000.0,
        max_position_embeddings=32768,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        attention_dropout=0.0,
        use_sliding_window=False,
        tie_word_embeddings=False,
        bos_token_id=151643,
        eos_token_id=151645,
    )
    vis = dict(SIGLIP_SO400M_384)
    vis["num_hidden_layers"] = vision_layers
    cfg = LlavaOnevisionConfig(
        text_config=text,
        vision_config=vis,
        image_token_index=151646,
        vision_feature_layer=vision_feature_layer,
        vision_feature_select_strategy="default",
        image_seq_length=729,
        multimodal_projector_bias=False,
        projector_hidden_act="gelu",
        tie_word_embeddings=False,
    )
    return cfg


def probe_siglip(cfg):
    print("=" * 70)
    print("PROBE 1: SiglipVisionModel hidden_states convention")
    vt = AutoModel.from_config(cfg.vision_config)
    print("  vision tower class:", type(vt).__name__)
    px = torch.randn(1, 3, 384, 384)
    for ohs in (True,):
        out = vt(pixel_values=px, output_hidden_states=ohs)
        hs = getattr(out, "hidden_states", None)
        print(f"  output_hidden_states={ohs}: type={type(out).__name__}")
        print(f"    has .hidden_states = {hs is not None}"
              + (f" len={len(hs)}" if hs is not None else ""))
        print(f"    last_hidden_state shape = {tuple(out.last_hidden_state.shape)}")
    return vt


def probe_full(cfg):
    print("=" * 70)
    print("PROBE 2: LlavaOnevisionForConditionalGeneration state_dict")
    m = LlavaOnevisionForConditionalGeneration(cfg)
    sd = m.state_dict()
    print("  num params:", len(sd))
    return m, sd


def probe_config_fields():
    print("=" * 70)
    print("PROBE 3: LlavaOnevisionConfig default fields")
    c = LlavaOnevisionConfig()
    d = {k: v for k, v in c.to_dict().items()}
    for k in sorted(d):
        if not isinstance(d[k], dict):
            print(f"    {k} = {d[k]}")
    print("  text_config type:", c.text_config.model_type)
    print("  vision_config type:", c.vision_config.model_type)


if __name__ == "__main__":
    probe_config_fields()
    cfg = build_config()
    print("=" * 70)
    print("config.vision_config.hidden_size:", cfg.vision_config.hidden_size)
    print("config.vision_config.model_type:", cfg.vision_config.model_type)
    print("config.image_seq_length:", cfg.image_seq_length)
    print("config.multimodal_projector_bias:", cfg.multimodal_projector_bias)
    probe_siglip(cfg)
    m, sd = probe_full(cfg)
    out = "/mnt/data/intern3/research/SingGuard/eval/scripts/_target_keys.json"
    with open(out, "w") as f:
        json.dump({k: list(v.shape) for k, v in sd.items()}, f, indent=1)
    print("  wrote", out)
