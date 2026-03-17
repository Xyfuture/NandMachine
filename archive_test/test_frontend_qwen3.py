import json
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers import Qwen3Config

from nandmachine.frontend.network.qwen3 import (
    Qwen3Attention,
    Qwen3DecoderLayer,
)


MODEL_CARD_PATH = (
    Path(__file__).resolve().parents[1] / "model_cards" / "qwen3-8B.json"
)


def load_qwen3_config() -> Qwen3Config:
    return Qwen3Config.from_dict(json.loads(MODEL_CARD_PATH.read_text()))


def test_qwen3_attention_keeps_hidden_shape():
    config = load_qwen3_config()
    with torch.device("meta"):
        module = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            head_dim=getattr(config, "head_dim", None),
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            rope_theta=getattr(config, "rope_theta", 1000000.0),
        )
        x = torch.empty(2, 5, config.hidden_size, device="meta")

    y = module(x)

    assert y.shape == x.shape


def test_qwen3_attention_adds_qk_norm_without_bias():
    config = load_qwen3_config()
    with torch.device("meta"):
        module = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            head_dim=getattr(config, "head_dim", None),
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=False,
            rope_theta=getattr(config, "rope_theta", 1000000.0),
        )
        x = torch.empty(2, 3, config.hidden_size, device="meta")

    y = module(x)

    assert hasattr(module, "q_norm")
    assert hasattr(module, "k_norm")
    assert y.shape == x.shape


def test_qwen3_decoder_layer_matches_tp1_shapes():
    config = load_qwen3_config()
    with torch.device("meta"):
        layer = Qwen3DecoderLayer(config)
        x = torch.empty(2, 4, config.hidden_size, device="meta")

    y = layer(x)

    assert y.shape == x.shape
    assert layer.self_attn.qkv_proj.weight.shape == (
        (config.num_attention_heads + 2 * config.num_key_value_heads) * config.head_dim,
        config.hidden_size,
    )
    assert layer.self_attn.o_proj.weight.shape == (
        config.hidden_size,
        config.hidden_size,
    )
    assert layer.mlp.gate_up_proj.weight.shape == (
        config.intermediate_size * 2,
        config.hidden_size,
    )
    assert layer.mlp.down_proj.weight.shape == (
        config.hidden_size,
        config.intermediate_size,
    )
