"""Shared dense row-wise quantization helpers."""

from __future__ import annotations


INTEGER_BITS = tuple(range(1, 9))
SUPPORTED_BITS = (*INTEGER_BITS, 16, 32)


def quantization_divisor(bits: int) -> int:
    if bits not in INTEGER_BITS:
        raise ValueError(f"unsupported integer quantization Q{bits}")
    if bits <= 2:
        return 1
    return (1 << (bits - 1)) - 1


def quantize_value(value: float, scale: float, bits: int) -> int:
    if bits not in INTEGER_BITS:
        raise ValueError(f"unsupported integer quantization Q{bits}")
    if scale == 0.0:
        return 0
    if bits == 1:
        return -1 if value < 0.0 else 1
    if bits == 2:
        ratio = value / scale
        return -1 if ratio < -0.5 else (1 if ratio > 0.5 else 0)
    limit = quantization_divisor(bits)
    return max(-limit, min(limit, round(value / scale)))


def quantized_code(value: int, bits: int) -> int:
    if bits == 1:
        return 1 if value > 0 else 0
    if bits == 2:
        return value + 1
    return value & ((1 << bits) - 1)


def pack_quantized(values, scale: float, bits: int) -> bytes:
    packed = bytearray((len(values) * bits + 7) // 8)
    mask = (1 << bits) - 1
    for index, value in enumerate(values):
        code = quantized_code(quantize_value(value, scale, bits), bits) & mask
        bit_position = index * bits
        byte = bit_position >> 3
        shift = bit_position & 7
        packed[byte] |= (code << shift) & 0xFF
        if shift + bits > 8:
            packed[byte + 1] |= code >> (8 - shift)
    return bytes(packed)


def unpack_quantized(packed: bytes, index: int, bits: int) -> int:
    bit_position = index * bits
    byte = bit_position >> 3
    shift = bit_position & 7
    code = packed[byte] >> shift
    if shift + bits > 8:
        code |= packed[byte + 1] << (8 - shift)
    code &= (1 << bits) - 1
    if bits == 1:
        return 1 if code else -1
    if bits == 2:
        return code - 1
    sign = 1 << (bits - 1)
    return code - (1 << bits) if code & sign else code
