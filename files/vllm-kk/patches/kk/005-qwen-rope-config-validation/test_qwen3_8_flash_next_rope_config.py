# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only regression tests for the Qwen3.8-Flash-Next rope config port.

Lineage patch 005 on karmic-kraken-beta: the checkpoint carries
``max_position_embeddings`` and the M-RoPE metadata only inside
``text_config``. transformers' rope validation (YaRN implicit-factor check)
and vLLM's max-position retrieval read the attribute on the composite
config, so the composite must mirror it, and the vLLM-side M-RoPE keys
(``mrope_section``, ``mrope_interleaved``) must be ignored by rope
validation instead of raising "Unrecognized keys in rope_parameters".
"""

from vllm.models.qwen3_8_flash_next.config import (
    Qwen3_8FlashNextConfig,
    Qwen3_8FlashNextTextConfig,
)

_TEXT_CONFIG = {
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 1,
    "head_dim": 16,
    "intermediate_size": 128,
    "vocab_size": 256,
    "layer_types": ["full_attention", "linear_attention"],
    "hc_count": 4,
    "max_position_embeddings": 524288,
    "rope_parameters": {
        "rope_type": "yarn",
        "factor": 4.0,
        "original_max_position_embeddings": 262144,
        "mrope_section": [16, 24, 24],
        "mrope_interleaved": True,
    },
}


def _composite() -> Qwen3_8FlashNextConfig:
    return Qwen3_8FlashNextConfig(
        text_config=dict(_TEXT_CONFIG),
        vision_config={},
    )


def test_composite_mirrors_text_max_position_embeddings():
    config = _composite()
    # The crash being fixed: attribute access on the composite must work and
    # must equal the text config value (checkpoint carries it only there).
    assert config.max_position_embeddings == 524288
    assert config.text_config.max_position_embeddings == 524288


def test_rope_validation_ignores_mrope_metadata():
    # Constructing with mrope_* keys must not raise; validation ignores them.
    config = _composite()
    ignored = config.ignore_keys_at_rope_validation
    assert {"mrope_section", "mrope_interleaved"} <= set(
        ignored if isinstance(ignored, (set, list, tuple)) else {ignored}
    )


def test_text_config_accepts_mrope_metadata():
    text = Qwen3_8FlashNextTextConfig(**_TEXT_CONFIG)
    assert text.rope_parameters["mrope_section"] == [16, 24, 24]


def test_explicit_composite_max_position_wins():
    config = Qwen3_8FlashNextConfig(
        text_config=dict(_TEXT_CONFIG),
        vision_config={},
        max_position_embeddings=131072,
    )
    assert config.max_position_embeddings == 131072
