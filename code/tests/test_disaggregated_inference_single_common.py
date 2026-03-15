from __future__ import annotations

import torch

from ch15.disaggregated_inference_single_common import (
    _flatten_prompt_batches,
    _format_batched_decode_output,
)


def test_flatten_prompt_batches_preserves_request_order() -> None:
    prompts = torch.arange(2 * 3 * 4, dtype=torch.long).reshape(2, 3, 4)

    flattened = _flatten_prompt_batches(prompts)

    assert flattened.shape == (6, 4)
    assert torch.equal(flattened[0], prompts[0, 0])
    assert torch.equal(flattened[1], prompts[0, 1])
    assert torch.equal(flattened[3], prompts[1, 0])


def test_format_batched_decode_output_matches_legacy_batch_size_one_layout() -> None:
    final_tokens = torch.tensor([[11], [22], [33]], dtype=torch.long)

    formatted = _format_batched_decode_output(
        final_tokens,
        requests_per_rank=3,
        batch_size=1,
    )

    assert formatted.shape == (3,)
    assert torch.equal(formatted, torch.tensor([11, 22, 33], dtype=torch.long))


def test_format_batched_decode_output_matches_legacy_multi_batch_layout() -> None:
    final_tokens = torch.tensor([[1], [2], [3], [4]], dtype=torch.long)

    formatted = _format_batched_decode_output(
        final_tokens,
        requests_per_rank=2,
        batch_size=2,
    )

    assert formatted.shape == (4, 1)
    assert torch.equal(formatted, final_tokens)
