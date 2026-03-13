import torch

from core.harness.verify_output_codec import deserialize_verify_tensor, serialize_verify_tensor


def test_verify_output_codec_round_trips_integer_dtype() -> None:
    tensor = torch.tensor([[11, 12, 13], [21, 22, 23]], dtype=torch.int64)

    payload = serialize_verify_tensor(tensor)
    restored = deserialize_verify_tensor(payload)

    assert restored.dtype == torch.int64
    assert torch.equal(restored, tensor)


def test_verify_output_codec_round_trips_bool_dtype() -> None:
    tensor = torch.tensor([[True, False], [False, True]], dtype=torch.bool)

    payload = serialize_verify_tensor(tensor)
    restored = deserialize_verify_tensor(payload)

    assert restored.dtype == torch.bool
    assert torch.equal(restored, tensor)


def test_verify_output_codec_round_trips_scalar_tensor() -> None:
    tensor = torch.tensor(7, dtype=torch.int64)

    payload = serialize_verify_tensor(tensor)
    restored = deserialize_verify_tensor(payload)

    assert restored.shape == torch.Size([])
    assert restored.dtype == torch.int64
    assert torch.equal(restored, tensor)
