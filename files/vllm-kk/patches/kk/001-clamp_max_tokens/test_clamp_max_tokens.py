# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the clamp_max_tokens request flag.

When set, the renderer must not reserve output tokens from the input cap
(``TokenizeParams.max_output_tokens`` stays 0) so the early-fail guard only
checks ``prompt <= max_model_len``; the serving layer's ``get_max_tokens``
then clamps the requested output to the remaining headroom.
"""

from unittest.mock import Mock

from vllm.config import ModelConfig
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.serve.utils.api_utils import get_max_tokens


def _model_config() -> Mock:
    model_config = Mock(spec=ModelConfig)
    model_config.max_model_len = 262144
    return model_config


def test_completion_clamp_zeroes_output_reservation():
    cfg = _model_config()

    default = CompletionRequest(model="m", prompt="hi", max_tokens=64000)
    assert default.build_tok_params(cfg).max_output_tokens == 64000

    clamped = CompletionRequest(
        model="m", prompt="hi", max_tokens=64000, clamp_max_tokens=True
    )
    assert clamped.build_tok_params(cfg).max_output_tokens == 0
    # The prompt cap must be the full model length so a prompt that fits in
    # max_model_len passes the renderer guard.
    assert clamped.build_tok_params(cfg).max_input_tokens == 262144


def test_completion_clamp_default_off():
    cfg = _model_config()
    req = CompletionRequest(model="m", prompt="hi")
    assert req.clamp_max_tokens is False


def test_chat_clamp_zeroes_output_reservation():
    cfg = _model_config()
    messages = [{"role": "user", "content": "hi"}]

    default = ChatCompletionRequest(
        model="m", messages=messages, max_tokens=64000
    )
    assert default.build_tok_params(cfg).max_output_tokens == 64000

    clamped = ChatCompletionRequest(
        model="m", messages=messages, max_tokens=64000, clamp_max_tokens=True
    )
    assert clamped.build_tok_params(cfg).max_output_tokens == 0

    # max_completion_tokens branch must be clamped too.
    clamped_mct = ChatCompletionRequest(
        model="m",
        messages=messages,
        max_completion_tokens=64000,
        clamp_max_tokens=True,
    )
    assert clamped_mct.build_tok_params(cfg).max_output_tokens == 0


def test_get_max_tokens_clamps_to_headroom():
    # 230000-token prompt, 64000 requested -> clamped to 32144.
    assert (
        get_max_tokens(262144, 64000, 230000, {}) == 262144 - 230000
    )
    # Requested output below headroom is untouched.
    assert get_max_tokens(262144, 1000, 230000, {}) == 1000
