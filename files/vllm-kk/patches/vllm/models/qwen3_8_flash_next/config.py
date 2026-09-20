# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3.8-Flash-Next model configuration."""

from typing import Any, ClassVar

from transformers import PretrainedConfig

from vllm.models.qwen4_exp.config import (
    Qwen4ExpConfig,
    Qwen4ExpTextConfig,
    Qwen4ExpVisionConfig,
)


class Qwen3_8FlashNextVisionConfig(Qwen4ExpVisionConfig):
    model_type = "qwen3_8_flash_next"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)


class Qwen3_8FlashNextTextConfig(Qwen4ExpTextConfig):
    model_type = "qwen3_8_flash_next_text"

    def __init__(self, **kwargs: Any) -> None:
        # M-RoPE metadata keys are vLLM-side extensions; transformers'
        # rope validation must not reject them (lineage patch 005).
        kwargs["ignore_keys_at_rope_validation"] = {
            "mrope_section",
            "mrope_interleaved",
        }
        super().__init__(**kwargs)


class Qwen3_8FlashNextConfig(Qwen4ExpConfig):
    model_type = "qwen3_8_flash_next"
    sub_configs: ClassVar[dict[str, type[PretrainedConfig]]] = {
        "vision_config": Qwen3_8FlashNextVisionConfig,
        "text_config": Qwen3_8FlashNextTextConfig,
    }

    def __init__(self, **kwargs: Any) -> None:
        # Mirror the text config's context limit onto the composite (lineage
        # patch 005, adapted to the qwen4_exp subclass shell). The checkpoint
        # carries max_position_embeddings only inside text_config, but
        # transformers v5 rope standardization reads
        # self.max_position_embeddings during __post_init__, before undeclared
        # kwargs are assigned — so set the attribute directly on self, and
        # pass the M-RoPE metadata keys via kwargs where they are declared.
        text_config = kwargs.get("text_config")
        if isinstance(text_config, dict):
            text_max_pos = text_config.get("max_position_embeddings")
        else:
            text_max_pos = getattr(text_config, "max_position_embeddings", None)
        if kwargs.get("max_position_embeddings") is None and text_max_pos is not None:
            kwargs["max_position_embeddings"] = text_max_pos
        if kwargs.get("max_position_embeddings") is not None:
            self.max_position_embeddings = kwargs["max_position_embeddings"]
        kwargs["ignore_keys_at_rope_validation"] = {
            "mrope_section",
            "mrope_interleaved",
        }
        super().__init__(**kwargs)


__all__ = [
    "Qwen3_8FlashNextConfig",
    "Qwen3_8FlashNextTextConfig",
    "Qwen3_8FlashNextVisionConfig",
    "Qwen4ExpConfig",
    "Qwen4ExpTextConfig",
    "Qwen4ExpVisionConfig",
]
