"""32-bit integer helpers and the Jenkins string hash."""

from __future__ import annotations


# --------------------------------------------------------------------------- #
# Jenkins lookup2 string hash (UTF-8 bytes -> signed 32-bit int).
# --------------------------------------------------------------------------- #

_FH_MASK = 0xFFFFFFFF


def _to_uint32(number: int) -> int:
    return number & _FH_MASK


def _to_int32(number: int) -> int:
    number &= _FH_MASK
    return number - 0x100000000 if number >= 0x80000000 else number


def _int32_xor(number: int, other_number: int) -> int:
    """ToInt32 of the 32-bit xor of the operands' uint32 forms."""
    return _to_int32(_to_uint32(number) ^ _to_uint32(other_number))


def _int32_left_shift(number: int, other_number: int) -> int:
    """(signed 32-bit result)."""
    return _to_int32((_to_uint32(number) << (other_number & 31)) & _FH_MASK)


def _uint32_right_shift(number: int, other_number: int) -> int:
    """(unsigned right shift)."""
    return _to_uint32(number) >> (other_number & 31)


def _little_endian_signed_word(byte_values: list, off: int) -> int:
    """Return a little-endian four-byte word with each byte sign-extended."""
    def _sign_extend_byte(number: int) -> int:
        return number - 256 if number > 127 else number
    return (_sign_extend_byte(byte_values[off]) + (_sign_extend_byte(byte_values[off + 1]) << 8)
            + (_sign_extend_byte(byte_values[off + 2]) << 16) + (_sign_extend_byte(byte_values[off + 3]) << 24))


def _utf8_bytes_from_utf16_units(text: str) -> list[int]:
    """Encode by walking UTF-16 code units, preserving lone surrogates."""
    raw = text.encode("utf-16-le", "surrogatepass")
    units = [raw[index] | (raw[index + 1] << 8) for index in range(0, len(raw), 2)]
    output_bytes: list[int] = []
    index = 0
    while index < len(units):
        value = units[index]
        if value < 128:
            output_bytes.append(value)
        elif value < 2048:
            output_bytes.append((value >> 6) | 192)
            output_bytes.append((value & 63) | 128)
        else:
            if (
                (value & 0xFC00) == 0xD800
                and index + 1 < len(units)
                and (units[index + 1] & 0xFC00) == 0xDC00
            ):
                index += 1
                value = 0x10000 + ((value & 1023) << 10) + (units[index] & 1023)
                output_bytes.append((value >> 18) | 240)
                output_bytes.append(((value >> 12) & 63) | 128)
            else:
                output_bytes.append((value >> 12) | 224)
            output_bytes.append(((value >> 6) & 63) | 128)
            output_bytes.append((value & 63) | 128)
        index += 1
    return output_bytes


def _jenkins_mix(mix_state: list) -> int:
    """Jenkins lookup2 mix over the 3-word state ``[a, b, c]``."""
    secondary_item, candidate_item, reference_item = mix_state
    secondary_item = _int32_xor(secondary_item - candidate_item - reference_item, _uint32_right_shift(reference_item, 13))
    candidate_item = _int32_xor(candidate_item - reference_item - secondary_item, _int32_left_shift(secondary_item, 8))
    reference_item = reference_item - secondary_item
    reference_item = _int32_xor(reference_item - candidate_item, _uint32_right_shift(candidate_item, 13))
    secondary_item = secondary_item - candidate_item
    secondary_item = secondary_item - reference_item
    secondary_item = _int32_xor(secondary_item, _uint32_right_shift(reference_item, 12))
    candidate_item = _int32_xor(candidate_item - reference_item - secondary_item, _int32_left_shift(secondary_item, 16))
    reference_item = reference_item - secondary_item
    reference_item = _int32_xor(reference_item - candidate_item, _uint32_right_shift(candidate_item, 5))
    secondary_item = secondary_item - candidate_item
    secondary_item = secondary_item - reference_item
    secondary_item = _int32_xor(secondary_item, _uint32_right_shift(reference_item, 3))
    candidate_item = _int32_xor(candidate_item - reference_item - secondary_item, _int32_left_shift(secondary_item, 10))
    reference_item = reference_item - secondary_item
    reference_item = _int32_xor(reference_item - candidate_item, _uint32_right_shift(candidate_item, 15))
    mix_state[0], mix_state[1], mix_state[2] = secondary_item, candidate_item, reference_item
    return reference_item


def jenkins_hash(text: str) -> int:
    """Encode text through the package UTF-16/UTF-8 byte path, then run Jenkins lookup2."""
    byte_values = _utf8_bytes_from_utf16_units(text)
    count_item = len(byte_values)
    mix_state = [-1640531527, -1640531527, 314159265]   # 0x9E3779B9, 0x9E3779B9, seed
    off = 0
    entry_item = count_item
    while entry_item >= 12:
        mix_state[0] = mix_state[0] + _little_endian_signed_word(byte_values, off)
        mix_state[1] = mix_state[1] + _little_endian_signed_word(byte_values, off + 4)
        mix_state[2] = mix_state[2] + _little_endian_signed_word(byte_values, off + 8)
        _jenkins_mix(mix_state)
        entry_item -= 12
        off += 12
    mix_state[2] = mix_state[2] + count_item
    # Tail-byte mixing follows Jenkins lookup2's fall-through layout.
    if entry_item >= 11:
        mix_state[2] = mix_state[2] + _int32_left_shift(byte_values[off + 10], 24)
    if entry_item >= 10:
        mix_state[2] = mix_state[2] + ((byte_values[off + 9] & 255) << 16)
    if entry_item >= 9:
        mix_state[2] = mix_state[2] + ((byte_values[off + 8] & 255) << 8)
    if entry_item >= 8:
        mix_state[1] = mix_state[1] + _little_endian_signed_word(byte_values, off + 4)
        mix_state[0] = mix_state[0] + _little_endian_signed_word(byte_values, off)
    elif entry_item >= 4:
        if entry_item >= 7:
            mix_state[1] = mix_state[1] + ((byte_values[off + 6] & 255) << 16)
        if entry_item >= 6:
            mix_state[1] = mix_state[1] + ((byte_values[off + 5] & 255) << 8)
        if entry_item >= 5:
            mix_state[1] = mix_state[1] + (byte_values[off + 4] & 255)
        mix_state[0] = mix_state[0] + _little_endian_signed_word(byte_values, off)
    else:
        if entry_item >= 3:
            mix_state[0] = mix_state[0] + ((byte_values[off + 2] & 255) << 16)
        if entry_item >= 2:
            mix_state[0] = mix_state[0] + ((byte_values[off + 1] & 255) << 8)
        if entry_item >= 1:
            mix_state[0] = mix_state[0] + (byte_values[off] & 255)
    return _jenkins_mix(mix_state)
