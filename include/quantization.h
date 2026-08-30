#ifndef BANANAMIND_QUANTIZATION_H
#define BANANAMIND_QUANTIZATION_H

#include <stdint.h>

static inline int lm_quantized_weight(const uint8_t *packed,
                                      uint32_t index, uint32_t bits) {
    if (bits == 8u) return (int8_t)packed[index];
    uint32_t bit_position = index * bits;
    uint32_t byte = bit_position >> 3;
    uint32_t shift = bit_position & 7u;
    uint32_t code = (uint32_t)packed[byte] >> shift;
    if (shift + bits > 8u)
        code |= (uint32_t)packed[byte + 1u] << (8u - shift);
    code &= (1u << bits) - 1u;
    if (bits == 1u) return code ? 1 : -1;
    if (bits == 2u) return (int)code - 1;
    uint32_t sign = 1u << (bits - 1u);
    return code & sign ? (int)code - (int)(1u << bits) : (int)code;
}

#endif
