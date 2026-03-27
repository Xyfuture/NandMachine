import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from nandmachine.config.model_config import Qwen3ModelConfig
from nandmachine.frontend.network.qwen3 import (
    Qwen3Attention,
    Qwen3DecoderLayer,
)


MODEL_CARD_PATH = (
    Path(__file__).resolve().parents[1] / "model_cards" / "qwen3-8B.json"
)


def load_qwen3_config() -> Qwen3ModelConfig:
    return Qwen3ModelConfig.from_dict(json.loads(MODEL_CARD_PATH.read_text()))


def test_qwen3_attention_keeps_hidden_shape():
    config = load_qwen3_config()
    with torch.device("meta"):
        module = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            tp_size=1,
            max_position=config.max_position_embeddings,
            head_dim=config.head_dim,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=config.attention_bias,
            rope_theta=config.rope_theta,
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
            tp_size=1,
            max_position=config.max_position_embeddings,
            head_dim=config.head_dim,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=False,
            rope_theta=config.rope_theta,
        )
        x = torch.empty(2, 3, config.hidden_size, device="meta")

    y = module(x)

    assert hasattr(module, "q_norm")
    assert hasattr(module, "k_norm")
    assert y.shape == x.shape


def test_qwen3_decoder_layer_matches_tp1_shapes():
    config = load_qwen3_config()
    head_dim = config.head_dim or (config.hidden_size // config.num_attention_heads)
    with torch.device("meta"):
        layer = Qwen3DecoderLayer(config, tp_size=1)
        x = torch.empty(2, 4, config.hidden_size, device="meta")

    y = layer(x)

    assert y.shape == x.shape
    assert layer.self_attn.qkv_proj.weight.shape == (
        (config.num_attention_heads + 2 * config.num_key_value_heads) * head_dim,
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


def test_qwen3_decoder_layer_matches_tp2_local_shapes():
    config = load_qwen3_config()
    tp_size = 2
    head_dim = config.head_dim or (config.hidden_size // config.num_attention_heads)
    with torch.device("meta"):
        layer = Qwen3DecoderLayer(config, tp_size=tp_size)
        x = torch.empty(2, 4, config.hidden_size, device="meta")

    y = layer(x)

    assert y.shape == x.shape
    assert layer.self_attn.num_heads == config.num_attention_heads // tp_size
    assert layer.self_attn.num_kv_heads == config.num_key_value_heads // tp_size
    assert layer.self_attn.qkv_proj.weight.shape == (
        ((config.num_attention_heads + 2 * config.num_key_value_heads) * head_dim) // tp_size,
        config.hidden_size,
    )
    assert layer.self_attn.o_proj.weight.shape == (
        config.hidden_size,
        config.hidden_size // tp_size,
    )
    assert layer.mlp.gate_up_proj.weight.shape == (
        (config.intermediate_size * 2) // tp_size,
        config.hidden_size,
    )
    assert layer.mlp.down_proj.weight.shape == (
        config.hidden_size,
        config.intermediate_size // tp_size,
    )


def test_qwen3_decoder_layer_rejects_invalid_tp_split():
    config = Qwen3ModelConfig.from_dict(
        {
            "hidden_size": 4096,
            "num_attention_heads": 30,
            "num_key_value_heads": 8,
            "max_position_embeddings": 40960,
            "intermediate_size": 12288,
            "hidden_act": "silu",
        }
    )

    with pytest.raises(ValueError, match="num_heads must be divisible by tp_size"):
        Qwen3DecoderLayer(config, tp_size=4)


def test_qwen3_model_config_uses_optional_defaults():
    config = Qwen3ModelConfig.from_dict(
        {
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "max_position_embeddings": 40960,
            "intermediate_size": 12288,
            "hidden_act": "silu",
        }
    )

    assert config.head_dim is None
    assert config.rms_norm_eps == 1e-6
    assert config.attention_bias is False
    assert config.rope_theta == 10000.0
    assert config.attention_type == "gqa"


def test_qwen3_model_config_reads_explicit_attention_type():
    config = Qwen3ModelConfig.from_dict(
        {
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "max_position_embeddings": 40960,
            "intermediate_size": 12288,
            "hidden_act": "silu",
            "attention_type": "mha",
        }
    )

    assert config.attention_type == "mha"
