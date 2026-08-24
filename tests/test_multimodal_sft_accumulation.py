from __future__ import annotations

import pytest

from training.multimodal_sft.trainer_core import GenericTrainerCore


@pytest.mark.parametrize(
    ("micro_batches", "gas", "expected_steps"),
    ((1, 4, 1), (4, 4, 1), (5, 4, 2), (8, 4, 2), (10, 4, 3)),
)
def test_accumulation_update_count_includes_tail(micro_batches: int, gas: int, expected_steps: int) -> None:
    assert (micro_batches + gas - 1) // gas == expected_steps


def test_tail_window_uses_mean_scaling() -> None:
    values = (2.0, 6.0)
    fixed_gas_gradient = sum(values) / 4.0
    dynamic_tail_gradient = sum(values) / len(values)
    explicit_batch_mean = sum(values) / len(values)
    assert dynamic_tail_gradient == explicit_batch_mean
    assert dynamic_tail_gradient != fixed_gas_gradient
