#include <stdint.h>
#include <stddef.h>

#include "litemodel_runtime.h"
#include "quantization.h"

#define LM_MAX_VOCAB 300000u
#define LM_MAX_HIDDEN 1024u
#define LM_MAX_INTERMEDIATE 8192u
#define LM_MAX_HEADS 16u
#define LM_MAX_KV_HEADS 8u
#define LM_MAX_HEAD_DIM 256u
#define LM_MAX_INPUT_TOKENS 4096u
#define LM_MAX_MERGES 600000u

static int bytes_equal(const void *left_, const void *right_, uint32_t count) {
    const uint8_t *left = left_;
    const uint8_t *right = right_;
    while (count--) if (*left++ != *right++) return 0;
    return 1;
}

static void bytes_zero(void *memory_, uint32_t count) {
    uint8_t *memory = memory_;
    while (count--) *memory++ = 0;
}

void *lm_arena_alloc(struct lm_arena *arena, uint32_t size, uint32_t alignment) {
    uintptr_t next = (uintptr_t)arena->next;
    uintptr_t aligned = (next + alignment - 1u) & ~(uintptr_t)(alignment - 1u);
    if (aligned > (uintptr_t)arena->end || size > (uint32_t)((uintptr_t)arena->end - aligned))
        return 0;
    arena->next = (uint8_t *)(aligned + size);
    bytes_zero((void *)aligned, size);
    return (void *)aligned;
}

int lm_range_ok(uint32_t offset, uint32_t size, uint32_t file_size) {
    return offset <= file_size && size <= file_size - offset;
}

const uint8_t *lm_parse_vector(const uint8_t *p, const uint8_t *end,
                               uint32_t count, const float **result) {
    uint32_t bytes = count * 4u;
    if ((uint32_t)(end - p) < bytes) return 0;
    *result = (const float *)p;
    return p + bytes;
}

const uint8_t *lm_parse_matrix(const uint8_t *p, const uint8_t *end,
                               uint32_t rows, uint32_t cols, uint32_t bits,
                               struct lm_matrix *result) {
    uint32_t stride;
    if (bits == 32u) stride = cols * 4u;
    else if (bits == 16u) stride = cols * 2u;
    else stride = 4u + (cols * bits + 7u) / 8u;
    if (rows && stride > 0xFFFFFFFFu / rows) return 0;
    uint32_t bytes = rows * stride;
    if ((uint32_t)(end - p) < bytes) return 0;
    result->data = p;
    result->rows = rows;
    result->cols = cols;
    result->stride = stride;
    result->bits = bits;
    return p + bytes;
}

static float half_to_float(uint16_t half) {
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

void lm_matrix_row(const struct lm_matrix *matrix, uint32_t row, float *output) {
    const uint8_t *record = matrix->data + row * matrix->stride;
    if (matrix->bits == 32u) {
        const float *values = (const float *)record;
        for (uint32_t column = 0; column < matrix->cols; ++column)
            output[column] = values[column];
        return;
    }
    if (matrix->bits == 16u) {
        const uint16_t *values = (const uint16_t *)record;
        for (uint32_t column = 0; column < matrix->cols; ++column)
            output[column] = half_to_float(values[column]);
        return;
    }
    float scale = *(const float *)record;
    const uint8_t *packed = record + 4;
    for (uint32_t column = 0; column < matrix->cols; ++column)
        output[column] = (float)lm_quantized_weight(packed, column, matrix->bits) * scale;
}

static void lm_matvec_scalar(const struct lm_matrix *matrix,
                             const float *input, float *output) {
    for (uint32_t row = 0; row < matrix->rows; ++row) {
        const uint8_t *record = matrix->data + row * matrix->stride;
        float sum = 0.0f;
        if (matrix->bits == 32u) {
            const float *values = (const float *)record;
            for (uint32_t column = 0; column < matrix->cols; ++column)
                sum += values[column] * input[column];
        } else if (matrix->bits == 16u) {
            const uint16_t *values = (const uint16_t *)record;
            for (uint32_t column = 0; column < matrix->cols; ++column)
                sum += half_to_float(values[column]) * input[column];
        } else {
            float scale = *(const float *)record;
            const uint8_t *packed = record + 4;
            for (uint32_t column = 0; column < matrix->cols; ++column)
                sum += (float)lm_quantized_weight(packed, column, matrix->bits) * input[column];
            sum *= scale;
        }
        output[row] = sum;
    }
}

void lm_matvec_sse(const struct lm_matrix *matrix, const float *input, float *output);
void lm_matvec_sse2(const struct lm_matrix *matrix, const float *input, float *output);

static void (*selected_matvec)(const struct lm_matrix *, const float *, float *) =
    lm_matvec_scalar;

void lm_matvec_configure(enum cpu_math_backend backend) {
    if (backend == CPU_MATH_SSE2) selected_matvec = lm_matvec_sse2;
    else if (backend == CPU_MATH_SSE) selected_matvec = lm_matvec_sse;
    else selected_matvec = lm_matvec_scalar;
}

void lm_matvec(const struct lm_matrix *matrix, const float *input, float *output) {
    selected_matvec(matrix, input, output);
}

float lm_sqrt(float value) {
    __asm__ volatile ("fsqrt" : "+t"(value));
    return value;
}

float lm_exp(float value) {
    if (value < -16.0f) return 0.0f;
    if (value > 16.0f) value = 16.0f;
    float transformed = value * 1.4426950409f;
    int power = (int)transformed;
    if ((float)power > transformed) --power;
    float fraction = transformed - (float)power;
    float polynomial = 1.0f + fraction * (0.69314718f + fraction *
        (0.24022651f + fraction * (0.05550411f + fraction * 0.00961813f)));
    union { uint32_t integer; float real; } scale;
    scale.integer = (uint32_t)(power + 127) << 23;
    return polynomial * scale.real;
}

float lm_sigmoid(float value) {
    return 1.0f / (1.0f + lm_exp(-value));
}

float lm_log(float value) {
    if (value <= 0.0f) return -1000000.0f;
    union { uint32_t integer; float real; } bits;
    bits.real = value;
    int exponent = (int)((bits.integer >> 23) & 255u) - 127;
    bits.integer = (bits.integer & 0x007FFFFFu) | 0x3F800000u;
    float y = (bits.real - 1.0f) / (bits.real + 1.0f);
    float y2 = y * y;
    float mantissa_log = 2.0f * y *
        (1.0f + y2 * (0.3333333333f + y2 * (0.2f + y2 * 0.1428571429f)));
    return (float)exponent * 0.6931471806f + mantissa_log;
}

float lm_tanh(float value) {
    if (value > 8.0f) return 1.0f;
    if (value < -8.0f) return -1.0f;
    return 2.0f * lm_sigmoid(2.0f * value) - 1.0f;
}

float lm_softplus(float value) {
    if (value > 16.0f) return value;
    if (value < -16.0f) return lm_exp(value);
    return lm_log(1.0f + lm_exp(value));
}

void lm_rms_norm_offset(struct lm_runtime *runtime, float *output,
                        const float *input, const float *weight,
                        uint32_t size, float weight_offset) {
    float sum = 0.0f;
    for (uint32_t index = 0; index < size; ++index) sum += input[index] * input[index];
    float inverse = 1.0f / lm_sqrt(sum / (float)size + runtime->header->norm_eps);
    if (weight) {
        for (uint32_t index = 0; index < size; ++index)
            output[index] = input[index] * inverse * (weight[index] + weight_offset);
    } else {
        for (uint32_t index = 0; index < size; ++index)
            output[index] = input[index] * inverse;
    }
}

void lm_rms_norm(struct lm_runtime *runtime, float *output, const float *input,
                 const float *weight, uint32_t size) {
    lm_rms_norm_offset(runtime, output, input, weight, size, 0.0f);
}

static uint32_t merge_hash(uint32_t left, uint32_t right, uint32_t mask) {
    return (left * 2654435761u ^ right * 2246822519u) & mask;
}

static int prepare_tokenizer(struct lm_runtime *runtime, struct lm_arena *arena) {
    const struct litemodel_header *header = runtime->header;
    uint32_t slots = 2u;
    while (slots < header->merges_count * 2u) slots <<= 1;
    runtime->merge_slots = slots;
    runtime->merge_table = lm_arena_alloc(
        arena, slots * sizeof(*runtime->merge_table), 16u
    );
    runtime->bpe_tokens = lm_arena_alloc(arena, LM_MAX_INPUT_TOKENS * 4u, 16u);
    runtime->prompt_tokens = lm_arena_alloc(
        arena, runtime->context_capacity * sizeof(*runtime->prompt_tokens), 16u
    );
    if (!runtime->merge_table || !runtime->bpe_tokens || !runtime->prompt_tokens) return 0;
    bytes_zero(runtime->merge_table, slots * sizeof(*runtime->merge_table));
    for (uint32_t index = 0; index < header->merges_count; ++index) {
        uint32_t left, right, result, rank;
        if (header->version == 1u) {
            const struct litemodel_merge_v1 *merge =
                &((const struct litemodel_merge_v1 *)runtime->merges)[index];
            left = merge->left; right = merge->right;
            result = merge->result; rank = merge->rank;
        } else {
            const struct litemodel_merge *merge =
                &((const struct litemodel_merge *)runtime->merges)[index];
            left = merge->left; right = merge->right;
            result = merge->result; rank = merge->rank;
        }
        uint32_t slot = merge_hash(left, right, slots - 1u);
        while (runtime->merge_table[slot].used &&
               (runtime->merge_table[slot].left != left ||
                runtime->merge_table[slot].right != right))
            slot = (slot + 1u) & (slots - 1u);
        runtime->merge_table[slot].used = 1u;
        runtime->merge_table[slot].left = left;
        runtime->merge_table[slot].right = right;
        runtime->merge_table[slot].result = result;
        runtime->merge_table[slot].rank_plus_one = rank + 1u;
    }
    for (uint32_t byte = 0; byte < 256u; ++byte)
        runtime->byte_token[byte] = 0xFFFFFFFFu;
    runtime->marker_token = header->unk_id;
    runtime->chat_end_id = 0xFFFFFFFFu;
    runtime->special_token_count = 0u;
    for (uint32_t id = 0; id < header->vocab_size; ++id) {
        const struct litemodel_token *token = &runtime->tokens[id];
        if (token->length == 1u &&
            runtime->byte_token[runtime->token_data[token->offset]] == 0xFFFFFFFFu)
            runtime->byte_token[runtime->token_data[token->offset]] = id;
        if (token->length == 3u &&
            runtime->token_data[token->offset] == 0xE2u &&
            runtime->token_data[token->offset + 1u] == 0x96u &&
            runtime->token_data[token->offset + 2u] == 0x81u)
            runtime->marker_token = id;
        static const uint8_t chat_end[10] = {
            '<', '|', 'i', 'm', '_', 'e', 'n', 'd', '|', '>'
        };
        if (token->length == sizeof(chat_end) &&
            bytes_equal(runtime->token_data + token->offset,
                        chat_end, sizeof(chat_end)))
            runtime->chat_end_id = id;
        if (header->version >= 2u && (token->flags & LITEMODEL_TOKEN_SPECIAL) &&
            token->length && runtime->special_token_count < LM_MAX_SPECIAL_TOKENS)
            runtime->special_token_ids[runtime->special_token_count++] = id;
    }
    for (uint32_t byte = 0; byte < 256u; ++byte)
        if (runtime->byte_token[byte] == 0xFFFFFFFFu)
            runtime->byte_token[byte] = header->unk_id;
    return 1;
}

static int allocate_transformer(struct lm_runtime *runtime, struct lm_arena *arena) {
    const struct litemodel_header *header = runtime->header;
    uint32_t hidden = header->hidden_size;
    uint32_t kv = header->num_kv_heads * header->head_dim;
    uint32_t intermediate = header->intermediate_size;
    uint32_t cache_values = header->num_layers * runtime->context_capacity * kv;
    runtime->key_cache = lm_arena_alloc(arena, cache_values * 4u, 16u);
    runtime->value_cache = lm_arena_alloc(arena, cache_values * 4u, 16u);
    uint32_t history_values = runtime->transformer.refresh_layers ?
        header->num_layers * hidden * (runtime->transformer.refresh_kernel - 1u) : 0u;
    runtime->refresh_history = history_values ?
        lm_arena_alloc(arena, history_values * 4u, 16u) : 0;
    runtime->x = lm_arena_alloc(arena, hidden * 4u, 16u);
    runtime->normalized = lm_arena_alloc(arena, hidden * 4u, 16u);
    runtime->temporary = lm_arena_alloc(arena, hidden * 4u, 16u);
    runtime->original_embedding = lm_arena_alloc(arena, hidden * 4u, 16u);
    runtime->query = lm_arena_alloc(arena, hidden * 4u, 16u);
    runtime->key = lm_arena_alloc(arena, kv * 4u, 16u);
    runtime->value = lm_arena_alloc(arena, kv * 4u, 16u);
    runtime->attention = lm_arena_alloc(arena, hidden * 4u, 16u);
    runtime->gate = lm_arena_alloc(arena, intermediate * 4u, 16u);
    runtime->up = lm_arena_alloc(arena, intermediate * 4u, 16u);
    runtime->refresh_gate = lm_arena_alloc(arena, hidden * 4u, 16u);
    runtime->refresh_value = lm_arena_alloc(arena, hidden * 4u, 16u);
    runtime->scores = lm_arena_alloc(arena, runtime->context_capacity * 4u, 16u);
    runtime->logits = lm_arena_alloc(arena, header->vocab_size * 4u, 16u);
    return runtime->key_cache && runtime->value_cache && runtime->x &&
        runtime->normalized && runtime->temporary && runtime->original_embedding &&
        runtime->query && runtime->key && runtime->value && runtime->attention &&
        runtime->gate && runtime->up && runtime->refresh_gate && runtime->refresh_value &&
        runtime->scores && runtime->logits &&
        (!history_values || runtime->refresh_history);
}

int litemodel_load_with_context(struct lm_runtime *runtime, const void *base_,
                                uint32_t size, struct lm_arena *arena,
                                uint32_t context_tokens) {
    const uint8_t *base = base_;
    bytes_zero(runtime, sizeof(*runtime));
    if (size < sizeof(struct litemodel_header)) return 0;
    const struct litemodel_header *header = (const struct litemodel_header *)base;
    if (!bytes_equal(header->magic, LITEMODEL_MAGIC, 8u) ||
        header->version < LITEMODEL_MIN_VERSION ||
        header->version > LITEMODEL_VERSION ||
        header->header_size != sizeof(*header) ||
        header->file_size > size ||
        (!header->quant_bits ||
         (header->quant_bits > 8u && header->quant_bits != 16u &&
          header->quant_bits != 32u)) ||
        !header->vocab_size || header->vocab_size > LM_MAX_VOCAB ||
        !header->hidden_size || header->hidden_size > LM_MAX_HIDDEN ||
        !header->num_layers || header->num_layers > LM_MAX_LAYERS ||
        !header->num_heads || header->num_heads > LM_MAX_HEADS ||
        !header->num_kv_heads || header->num_kv_heads > LM_MAX_KV_HEADS ||
        !header->head_dim || header->head_dim > LM_MAX_HEAD_DIM ||
        !header->intermediate_size || header->intermediate_size > LM_MAX_INTERMEDIATE)
        return 0;
    if (context_tokens < 16u) context_tokens = 16u;
    if (context_tokens > LM_MAX_CONTEXT) context_tokens = LM_MAX_CONTEXT;
    if (header->max_seq_len && context_tokens > header->max_seq_len)
        context_tokens = header->max_seq_len;
    runtime->context_capacity = context_tokens;
    if (!lm_range_ok(header->weights_offset, header->weights_size, header->file_size) ||
        !lm_range_ok(header->token_index_offset,
                     header->vocab_size * sizeof(struct litemodel_token), header->file_size) ||
        !lm_range_ok(header->token_data_offset, header->token_data_size, header->file_size) ||
        !lm_range_ok(header->merges_offset,
                     header->merges_count *
                         (header->version == 1u ? sizeof(struct litemodel_merge_v1) :
                                                  sizeof(struct litemodel_merge)),
                     header->file_size) ||
        !lm_range_ok(header->architecture_data_offset,
                     header->architecture_data_size, header->file_size) ||
        header->merges_count > LM_MAX_MERGES)
        return 0;

    runtime->header = header;
    runtime->file_base = base;
    runtime->file_size = size;
    runtime->tokens = (const struct litemodel_token *)(base + header->token_index_offset);
    runtime->token_data = base + header->token_data_offset;
    runtime->merges = base + header->merges_offset;
    for (uint32_t id = 0; id < header->vocab_size; ++id) {
        const struct litemodel_token *token = &runtime->tokens[id];
        if (token->offset > header->token_data_size ||
            token->length > header->token_data_size - token->offset) return 0;
    }

    const uint8_t *config = base + header->architecture_data_offset;
    const uint8_t *weights = base + header->weights_offset;
    const uint8_t *weights_end = weights + header->weights_size;
    int parsed = 0;
    if (header->architecture == LITEMODEL_ARCH_BANANA)
        parsed = lm_arch_banana_load(runtime, config, header->architecture_data_size,
                                     weights, weights_end);
    else if (header->architecture == LITEMODEL_ARCH_LLAMA)
        parsed = lm_arch_llama_load(runtime, config, header->architecture_data_size,
                                    weights, weights_end);
    else if (header->architecture == LITEMODEL_ARCH_GPTX2)
        parsed = lm_arch_gptx_load(runtime, config, header->architecture_data_size,
                                   weights, weights_end);
    else if (header->architecture == LITEMODEL_ARCH_ROSE_X1)
        parsed = lm_arch_rose_load(runtime, config, header->architecture_data_size,
                                   weights, weights_end);
    else if (header->architecture == LITEMODEL_ARCH_MINSPARK)
        parsed = lm_arch_minspark_load(runtime, config, header->architecture_data_size,
                                       weights, weights_end);
    else if (header->architecture == LITEMODEL_ARCH_LFM2)
        parsed = lm_arch_lfm2_load(runtime, config, header->architecture_data_size,
                                   weights, weights_end);
    else if (header->architecture == LITEMODEL_ARCH_GEMMA3)
        parsed = lm_arch_gemma3_load(runtime, config, header->architecture_data_size,
                                     weights, weights_end);
    else if (header->architecture == LITEMODEL_ARCH_QWEN35)
        parsed = lm_arch_qwen35_load(runtime, config, header->architecture_data_size,
                                     weights, weights_end);
    if (!parsed || !prepare_tokenizer(runtime, arena)) return 0;
    if (header->architecture == LITEMODEL_ARCH_MINSPARK) {
        if (!lm_arch_minspark_allocate(runtime, arena)) return 0;
    } else if (header->architecture == LITEMODEL_ARCH_LFM2) {
        if (!lm_arch_lfm2_allocate(runtime, arena)) return 0;
    } else if (header->architecture == LITEMODEL_ARCH_GEMMA3) {
        if (!lm_arch_gemma3_allocate(runtime, arena)) return 0;
    } else if (header->architecture == LITEMODEL_ARCH_QWEN35) {
        if (!lm_arch_qwen35_allocate(runtime, arena)) return 0;
    } else if (!allocate_transformer(runtime, arena)) return 0;
    litemodel_reset(runtime);
    return 1;
}

int litemodel_load(struct lm_runtime *runtime, const void *base, uint32_t size,
                   struct lm_arena *arena) {
    return litemodel_load_with_context(runtime, base, size, arena, LM_CONTEXT);
}

void litemodel_reset(struct lm_runtime *runtime) {
    if (runtime->header->architecture == LITEMODEL_ARCH_LFM2)
        lm_arch_lfm2_reset(runtime);
    else if (runtime->header->architecture == LITEMODEL_ARCH_QWEN35)
        lm_arch_qwen35_reset(runtime);
    else if (runtime->header->architecture != LITEMODEL_ARCH_MINSPARK)
        lm_transformer_reset(runtime);
}

static const struct lm_merge_slot *find_merge(struct lm_runtime *runtime,
                                               uint32_t left, uint32_t right) {
    uint32_t slot = merge_hash(left, right, runtime->merge_slots - 1u);
    while (runtime->merge_table[slot].used) {
        if (runtime->merge_table[slot].left == left &&
            runtime->merge_table[slot].right == right)
            return &runtime->merge_table[slot];
        slot = (slot + 1u) & (runtime->merge_slots - 1u);
    }
    return 0;
}

#define LM_BPE_SPECIAL_BIT 0x80000000u

static uint32_t match_special_token(const struct lm_runtime *runtime,
                                    const uint8_t *input, uint32_t remaining,
                                    uint32_t *matched_length) {
    uint32_t best_id = 0u, best_length = 0u;
    for (uint32_t index = 0u; index < runtime->special_token_count; ++index) {
        uint32_t id = runtime->special_token_ids[index];
        const struct litemodel_token *token = &runtime->tokens[id];
        if (token->length <= best_length || token->length > remaining) continue;
        if (bytes_equal(runtime->token_data + token->offset, input, token->length)) {
            best_id = id;
            best_length = token->length;
        }
    }
    *matched_length = best_length;
    return best_id;
}

uint32_t litemodel_tokenize(struct lm_runtime *runtime, const uint8_t *input,
                            uint32_t length, uint32_t *output, uint32_t capacity) {
    if (length > LM_MAX_INPUT_TOKENS) length = LM_MAX_INPUT_TOKENS;
    uint32_t count = 0u;
    if ((runtime->header->flags & LITEMODEL_FLAG_SPACE_TO_MARKER) &&
        length && input[0] != (uint8_t)' ' && count < LM_MAX_INPUT_TOKENS)
        runtime->bpe_tokens[count++] = runtime->marker_token;
    for (uint32_t index = 0; index < length && count < LM_MAX_INPUT_TOKENS;) {
        uint32_t special_length = 0u;
        uint32_t special = match_special_token(runtime, input + index,
                                               length - index, &special_length);
        if (special_length) {
            runtime->bpe_tokens[count++] = special | LM_BPE_SPECIAL_BIT;
            index += special_length;
            continue;
        }
        if ((runtime->header->flags & LITEMODEL_FLAG_SPACE_TO_MARKER) &&
            input[index] == (uint8_t)' ') {
            runtime->bpe_tokens[count++] = runtime->marker_token;
        } else {
            runtime->bpe_tokens[count++] = runtime->byte_token[input[index]];
        }
        ++index;
    }
    for (;;) {
        uint32_t best_position = count;
        uint32_t best_rank = 0xFFFFFFFFu;
        uint32_t best_result = 0;
        for (uint32_t index = 0; index + 1u < count; ++index) {
            const struct lm_merge_slot *merge = find_merge(
                runtime, runtime->bpe_tokens[index], runtime->bpe_tokens[index + 1u]
            );
            if (merge && (uint32_t)(merge->rank_plus_one - 1u) < best_rank) {
                best_rank = merge->rank_plus_one - 1u;
                best_position = index;
                best_result = merge->result;
            }
        }
        if (best_position == count) break;
        runtime->bpe_tokens[best_position] = best_result;
        for (uint32_t index = best_position + 1u; index + 1u < count; ++index)
            runtime->bpe_tokens[index] = runtime->bpe_tokens[index + 1u];
        --count;
    }
    uint32_t written = 0;
    if (!(runtime->header->flags & LITEMODEL_FLAG_NO_BOS) && written < capacity)
        output[written++] = runtime->header->bos_id;
    uint32_t room = capacity - written;
    uint32_t start = count > room ? count - room : 0u;
    while (start < count && written < capacity)
        output[written++] = runtime->bpe_tokens[start++] & ~LM_BPE_SPECIAL_BIT;
    return written;
}

uint32_t litemodel_forward(struct lm_runtime *runtime, uint32_t token,
                           uint32_t position) {
    if (runtime->header->architecture == LITEMODEL_ARCH_MINSPARK)
        return lm_arch_minspark_forward(runtime, &token, 1u);
    if (runtime->header->architecture == LITEMODEL_ARCH_LFM2)
        return lm_arch_lfm2_forward(runtime, token, position);
    if (runtime->header->architecture == LITEMODEL_ARCH_GEMMA3)
        return lm_arch_gemma3_forward(runtime, token, position);
    if (runtime->header->architecture == LITEMODEL_ARCH_QWEN35)
        return lm_arch_qwen35_forward(runtime, token, position);
    return lm_transformer_forward(runtime, token, position);
}

uint32_t litemodel_forward_sequence(struct lm_runtime *runtime,
                                    const uint32_t *tokens, uint32_t count) {
    if (runtime->header->architecture == LITEMODEL_ARCH_MINSPARK)
        return lm_arch_minspark_forward(runtime, tokens, count);
    if (!count) return 0u;
    uint32_t next = 0u;
    for (uint32_t position = 0; position < count; ++position)
        next = litemodel_forward(runtime, tokens[position], position);
    return next;
}

const struct litemodel_token *litemodel_token(const struct lm_runtime *runtime,
                                               uint32_t id) {
    if (id >= runtime->header->vocab_size) return 0;
    return &runtime->tokens[id];
}

uint32_t litemodel_sample(const struct lm_runtime *runtime,
                          uint32_t temperature_tenths, uint32_t random_value) {
    const struct litemodel_header *header = runtime->header;
    if (!header || !runtime->logits || !header->vocab_size) return 0u;
    uint32_t best = 0u;
    for (uint32_t index = 1u; index < header->vocab_size; ++index)
        if (runtime->logits[index] > runtime->logits[best]) best = index;
    if (!temperature_tenths) return best;
    float temperature = (float)temperature_tenths / 10.0f;
    float maximum = runtime->logits[best];
    float total = 0.0f;
    for (uint32_t index = 0u; index < header->vocab_size; ++index)
        total += lm_exp((runtime->logits[index] - maximum) / temperature);
    float unit = (float)(random_value & 0x00FFFFFFu) / 16777216.0f;
    float target = unit * total;
    float cumulative = 0.0f;
    for (uint32_t index = 0u; index < header->vocab_size; ++index) {
        cumulative += lm_exp((runtime->logits[index] - maximum) / temperature);
        if (cumulative >= target) return index;
    }
    return best;
}

int litemodel_is_stop(const struct lm_runtime *runtime, uint32_t id) {
    if (!runtime || !runtime->header) return 1;
    return id == runtime->header->eos_id || id == runtime->chat_end_id;
}
