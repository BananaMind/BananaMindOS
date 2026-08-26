#include <stdint.h>
#include <xmmintrin.h>
#if LM_SIMD_LEVEL >= 2
#include <emmintrin.h>
#endif

#include "litemodel_runtime.h"

#ifndef LM_SIMD_FUNCTION
#error LM_SIMD_FUNCTION must name this backend
#endif

static int8_t simd_signed_nibble(uint8_t value) {
    value &= 15u;
    return value & 8u ? (int8_t)(value | 0xF0u) : (int8_t)value;
}

static int simd_weight_at(const uint8_t *packed, uint32_t index, uint32_t bits) {
    if (bits == 8u) return (int8_t)packed[index];
    if (bits == 4u)
        return simd_signed_nibble(packed[index >> 1] >> ((index & 1u) * 4u));
    return (int)((packed[index >> 2] >> ((index & 3u) * 2u)) & 3u) - 1;
}

static float simd_half_to_float(uint16_t half) {
    uint32_t sign = (uint32_t)(half & 0x8000u) << 16;
    uint32_t exponent = (half >> 10) & 31u;
    uint32_t mantissa = half & 1023u;
    union { uint32_t integer; float real; } value;
    if (exponent == 0u) {
        if (mantissa == 0u) value.integer = sign;
        else {
            int shift = 0;
            while (!(mantissa & 1024u)) { mantissa <<= 1; ++shift; }
            mantissa &= 1023u;
            value.integer = sign | ((uint32_t)(113 - shift) << 23) | (mantissa << 13);
        }
    } else if (exponent == 31u) {
        value.integer = sign | 0x7F800000u | (mantissa << 13);
    } else {
        value.integer = sign | ((exponent + 112u) << 23) | (mantissa << 13);
    }
    return value.real;
}

static float horizontal_sum(__m128 value) {
    float lanes[4];
    _mm_storeu_ps(lanes, value);
    return lanes[0] + lanes[1] + lanes[2] + lanes[3];
}

void LM_SIMD_FUNCTION(const struct lm_matrix *matrix,
                      const float *input, float *output) {
    for (uint32_t row = 0; row < matrix->rows; ++row) {
        const uint8_t *record = matrix->data + row * matrix->stride;
        uint32_t column = 0u;
        __m128 accumulated = _mm_setzero_ps();
        if (matrix->bits == 32u) {
            const float *weights = (const float *)record;
            for (; column + 4u <= matrix->cols; column += 4u) {
                __m128 weight = _mm_loadu_ps(&weights[column]);
                __m128 activation = _mm_loadu_ps(&input[column]);
                accumulated = _mm_add_ps(accumulated, _mm_mul_ps(weight, activation));
            }
        } else if (matrix->bits == 16u) {
            const uint16_t *weights = (const uint16_t *)record;
            for (; column + 4u <= matrix->cols; column += 4u) {
                __m128 weight = _mm_set_ps(
                    simd_half_to_float(weights[column + 3u]),
                    simd_half_to_float(weights[column + 2u]),
                    simd_half_to_float(weights[column + 1u]),
                    simd_half_to_float(weights[column])
                );
                accumulated = _mm_add_ps(
                    accumulated, _mm_mul_ps(weight, _mm_loadu_ps(&input[column]))
                );
            }
        } else {
            float scale = *(const float *)record;
            const uint8_t *packed = record + 4u;
#if LM_SIMD_LEVEL >= 2
            if (matrix->bits == 8u) {
                __m128 scale_vector = _mm_set1_ps(scale);
                __m128i zero = _mm_setzero_si128();
                for (; column + 8u <= matrix->cols; column += 8u) {
                    __m128i bytes = _mm_loadl_epi64((const __m128i *)&packed[column]);
                    __m128i sign8 = _mm_cmpgt_epi8(zero, bytes);
                    __m128i words = _mm_unpacklo_epi8(bytes, sign8);
                    __m128i sign16 = _mm_cmpgt_epi16(zero, words);
                    __m128 low = _mm_cvtepi32_ps(_mm_unpacklo_epi16(words, sign16));
                    __m128 high = _mm_cvtepi32_ps(_mm_unpackhi_epi16(words, sign16));
                    low = _mm_mul_ps(low, scale_vector);
                    high = _mm_mul_ps(high, scale_vector);
                    accumulated = _mm_add_ps(
                        accumulated, _mm_mul_ps(low, _mm_loadu_ps(&input[column]))
                    );
                    accumulated = _mm_add_ps(
                        accumulated, _mm_mul_ps(high, _mm_loadu_ps(&input[column + 4u]))
                    );
                }
            }
#endif
            __m128 scale_vector = _mm_set1_ps(scale);
            for (; column + 4u <= matrix->cols; column += 4u) {
                __m128 weight = _mm_set_ps(
                    (float)simd_weight_at(packed, column + 3u, matrix->bits),
                    (float)simd_weight_at(packed, column + 2u, matrix->bits),
                    (float)simd_weight_at(packed, column + 1u, matrix->bits),
                    (float)simd_weight_at(packed, column, matrix->bits)
                );
                weight = _mm_mul_ps(weight, scale_vector);
                accumulated = _mm_add_ps(
                    accumulated, _mm_mul_ps(weight, _mm_loadu_ps(&input[column]))
                );
            }
        }
        float sum = horizontal_sum(accumulated);
        if (matrix->bits == 32u) {
            const float *weights = (const float *)record;
            for (; column < matrix->cols; ++column) sum += weights[column] * input[column];
        } else if (matrix->bits == 16u) {
            const uint16_t *weights = (const uint16_t *)record;
            for (; column < matrix->cols; ++column)
                sum += simd_half_to_float(weights[column]) * input[column];
        } else {
            float scale = *(const float *)record;
            const uint8_t *packed = record + 4u;
            for (; column < matrix->cols; ++column)
                sum += (float)simd_weight_at(packed, column, matrix->bits) * scale * input[column];
        }
        output[row] = sum;
    }
}
