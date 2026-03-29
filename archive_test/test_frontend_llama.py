import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from nandmachine.config.model_config import LlamaModelConfig
from nandmachine.frontend.network.llama import LlamaAttention, LlamaDecoderLayer


MODEL_CARD_PATH = (
    Path(__file__).resolve().parents[1] / "model_cards" / "llama-405B.json"
)


def load_llama_config() -> LlamaModelConfig:
    return LlamaModelConfig.from_dict(json.loads(MODEL_CARD_PATH.read_text()))


def test_llama_attention_keeps_hidden_shape():
    config = load_llama_config()
    with torch.device("meta"):
        module = LlamaAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            head_dim=config.head_dim,
            attention_bias=config.attention_bias,
            rope_theta=config.rope_theta,
        )
        x = torch.empty(2, 5, config.hidden_size, device="meta")

    y = module(x)

    assert y.shape == x.shape


def test_llama_decoder_layer_matches_tp1_shapes():
    config = load_llama_config()
    head_dim = config.head_dim or (config.hidden_size // config.num_attention_heads)
    with torch.device("meta"):
        layer = LlamaDecoderLayer(config)
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


def test_llama_model_config_reads_model_card_fields():
    config = load_llama_config()

    assert config.mlp_bias is False
    assert config.rope_theta == 500000.0
    assert config.rope_scaling is not None
    assert config.rope_scaling["rope_type"] == "llama3"
    assert config.attention_type == "gqa"
