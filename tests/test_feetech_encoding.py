import pytest

from so101_zero_shot.hardware.feetech import _to_bytes, decode_sign_magnitude, encode_sign_magnitude


@pytest.mark.parametrize("v,bit", [(0, 15), (1, 15), (-1, 15), (4095, 15), (-2047, 11), (2047, 11)])
def test_sign_magnitude_roundtrip(v, bit):
    assert decode_sign_magnitude(encode_sign_magnitude(v, bit), bit) == v


def test_sign_magnitude_overflow():
    with pytest.raises(ValueError):
        encode_sign_magnitude(2048, 11)


def test_to_bytes_little_endian():
    assert _to_bytes(0x1234, 2) == [0x34, 0x12]
    with pytest.raises(ValueError):
        _to_bytes(256, 1)
