#include <stdint.h>

#include "litemodel_runtime.h"

/* min-spark's Meiosis graph is deliberately isolated from the transformer core. */
#define MS_MAX_PRELUDE 2u
#define MS_MAX_BODY 4u
#define MS_MAX_CODA 2u
#define MS_MAX_LOOPS 4u
#define MS_MAX_RANK 64u

struct __attribute__((packed)) ms_file_config {
    uint32_t prelude_layers;
    uint32_t body_blocks;
    uint32_t coda_layers;
    uint32_t max_loops;
    uint32_t train_loops;
    uint32_t lora_rank;
    float ddl_k_eps;
    float ddl_v_scale;
    uint32_t document_eos;
};

struct ms_plain_block {
    const float *attention_norm;
    struct lm_matrix qkv;
    struct lm_matrix attention_output;
    const float *ffn_norm;
    struct lm_matrix gate_up;
    struct lm_matrix down;
};

struct ms_deep_delta {
    struct lm_matrix beta;
    const float *beta_bias;
    struct lm_matrix value;
    const float *value_bias;
};

struct ms_body_block {
    struct ms_plain_block block;
    struct ms_deep_delta attention_delta;
    struct ms_deep_delta ffn_delta;
    struct lm_matrix lora_down[MS_MAX_LOOPS];
    struct lm_matrix lora_up[MS_MAX_LOOPS];
};

struct ms_state {
    struct ms_file_config config;
    uint32_t active_loops;
    struct lm_matrix embedding;
    struct ms_plain_block prelude[MS_MAX_PRELUDE];
    struct ms_body_block body[MS_MAX_BODY];
    struct lm_matrix loop_embedding;
    struct ms_plain_block coda[MS_MAX_CODA];
    const float *final_norm;

    float *x;
    float *qkv;
    float *attention;
    float *normalized;
    float *sublayer_output;
    float *gate_up;
    float *lora_rank;
    float *lora_delta;
    float *loop_embedding_values;
    float *scores;
    float *logits;
    uint32_t *documents;
};

static struct ms_state min_spark;

static void zero_bytes(void *memory_, uint32_t count) {
    uint8_t *memory = memory_;
    while (count--) *memory++ = 0u;
}

static const uint8_t *parse_plain(const uint8_t *position, const uint8_t *end,
                                  const struct litemodel_header *header,
                                  struct ms_plain_block *block) {
    uint32_t hidden = header->hidden_size;
    uint32_t kv = header->num_kv_heads * header->head_dim;
    uint32_t qkv = hidden + 2u * kv;
    position = lm_parse_vector(position, end, hidden, &block->attention_norm);
    if (position) position = lm_parse_matrix(
        position, end, qkv, hidden, header->quant_bits, &block->qkv
    );
    if (position) position = lm_parse_matrix(
        position, end, hidden, hidden, header->quant_bits, &block->attention_output
    );
    if (position) position = lm_parse_vector(
        position, end, hidden, &block->ffn_norm
    );
    if (position) position = lm_parse_matrix(
        position, end, header->intermediate_size * 2u, hidden,
        header->quant_bits, &block->gate_up
    );
    if (position) position = lm_parse_matrix(
        position, end, hidden, header->intermediate_size,
        header->quant_bits, &block->down
    );
    return position;
}

static const uint8_t *parse_deep_delta(const uint8_t *position, const uint8_t *end,
                                       const struct litemodel_header *header,
                                       struct ms_deep_delta *delta) {
    uint32_t hidden = header->hidden_size;
    position = lm_parse_matrix(
        position, end, 1u, hidden, header->quant_bits, &delta->beta
    );
    if (position) position = lm_parse_vector(position, end, 1u, &delta->beta_bias);
    if (position) position = lm_parse_matrix(
        position, end, 1u, hidden, header->quant_bits, &delta->value
    );
    if (position) position = lm_parse_vector(position, end, 1u, &delta->value_bias);
    return position;
}

int lm_arch_minspark_load(struct lm_runtime *runtime, const uint8_t *config,
                          uint32_t config_size, const uint8_t *weights,
                          const uint8_t *weights_end) {
    const struct litemodel_header *header = runtime->header;
    if (config_size != sizeof(struct ms_file_config)) return 0;
    zero_bytes(&min_spark, sizeof(min_spark));
    const struct ms_file_config *file_config = (const struct ms_file_config *)config;
    min_spark.config = *file_config;
    if (!file_config->prelude_layers || file_config->prelude_layers > MS_MAX_PRELUDE ||
        !file_config->body_blocks || file_config->body_blocks > MS_MAX_BODY ||
        !file_config->coda_layers || file_config->coda_layers > MS_MAX_CODA ||
        !file_config->max_loops || file_config->max_loops > MS_MAX_LOOPS ||
        !file_config->train_loops || file_config->train_loops > file_config->max_loops ||
        !file_config->lora_rank || file_config->lora_rank > MS_MAX_RANK ||
        header->head_dim != 48u || header->num_heads * header->head_dim != header->hidden_size)
        return 0;
    min_spark.active_loops = file_config->train_loops;

    uint32_t hidden = header->hidden_size;
    uint32_t kv = header->num_kv_heads * header->head_dim;
    uint32_t qkv = hidden + 2u * kv;
    const uint8_t *position = lm_parse_matrix(
        weights, weights_end, header->vocab_size, hidden,
        header->quant_bits, &min_spark.embedding
    );
    for (uint32_t index = 0; position && index < file_config->prelude_layers; ++index)
        position = parse_plain(position, weights_end, header, &min_spark.prelude[index]);
    for (uint32_t index = 0; position && index < file_config->body_blocks; ++index) {
        struct ms_body_block *body = &min_spark.body[index];
        position = parse_plain(position, weights_end, header, &body->block);
        if (position) position = parse_deep_delta(
            position, weights_end, header, &body->attention_delta
        );
        if (position) position = parse_deep_delta(
            position, weights_end, header, &body->ffn_delta
        );
        for (uint32_t loop = 0; position && loop < file_config->max_loops; ++loop) {
            position = lm_parse_matrix(
                position, weights_end, file_config->lora_rank, hidden,
                header->quant_bits, &body->lora_down[loop]
            );
            if (position) position = lm_parse_matrix(
                position, weights_end, qkv, file_config->lora_rank,
                header->quant_bits, &body->lora_up[loop]
            );
        }
    }
    if (position) position = lm_parse_matrix(
        position, weights_end, file_config->max_loops, hidden,
        header->quant_bits, &min_spark.loop_embedding
    );
    for (uint32_t index = 0; position && index < file_config->coda_layers; ++index)
        position = parse_plain(position, weights_end, header, &min_spark.coda[index]);
    if (position) position = lm_parse_vector(
        position, weights_end, hidden, &min_spark.final_norm
    );
    if (!position || position != weights_end) return 0;
    runtime->embedding = min_spark.embedding;
    runtime->final_norm = min_spark.final_norm;
    runtime->architecture_state = &min_spark;
    return 1;
}

int lm_arch_minspark_allocate(struct lm_runtime *runtime, struct lm_arena *arena) {
    struct ms_state *state = runtime->architecture_state;
    const struct litemodel_header *header = runtime->header;
    if (!state) return 0;
    uint32_t hidden = header->hidden_size;
    uint32_t qkv = hidden + 2u * header->num_kv_heads * header->head_dim;
    state->x = lm_arena_alloc(arena, runtime->context_capacity * hidden * 4u, 16u);
    state->qkv = lm_arena_alloc(arena, runtime->context_capacity * qkv * 4u, 16u);
    state->attention = lm_arena_alloc(arena, hidden * 4u, 16u);
    state->normalized = lm_arena_alloc(arena, hidden * 4u, 16u);
    state->sublayer_output = lm_arena_alloc(arena, hidden * 4u, 16u);
    state->gate_up = lm_arena_alloc(
        arena, header->intermediate_size * 2u * 4u, 16u
    );
    state->lora_rank = lm_arena_alloc(arena, state->config.lora_rank * 4u, 16u);
    state->lora_delta = lm_arena_alloc(arena, qkv * 4u, 16u);
    state->loop_embedding_values = lm_arena_alloc(arena, hidden * 4u, 16u);
    state->scores = lm_arena_alloc(arena, runtime->context_capacity * 4u, 16u);
    state->logits = lm_arena_alloc(arena, header->vocab_size * 4u, 16u);
    state->documents = lm_arena_alloc(arena, runtime->context_capacity * 4u, 16u);
    runtime->logits = state->logits;
    return state->x && state->qkv && state->attention && state->normalized &&
        state->sublayer_output && state->gate_up && state->lora_rank &&
        state->lora_delta && state->loop_embedding_values && state->scores &&
        state->logits && state->documents;
}

static void fast_sincos(float angle, float *sine, float *cosine) {
    const float pi = 3.14159265359f;
    const float tau = 6.28318530718f;
    while (angle > pi) angle -= tau;
    while (angle < -pi) angle += tau;
    float squared = angle * angle;
    *sine = angle * (1.0f + squared *
        (-0.16666667f + squared * (0.00833333f + squared * -0.00019841f)));
    *cosine = 1.0f + squared *
        (-0.5f + squared * (0.04166667f + squared * -0.00138889f));
}

static void apply_rope(float *values, uint32_t heads, uint32_t position) {
    const uint32_t head_dim = 48u;
    const float frequency_ratio = 0.6812920691f;
    for (uint32_t head = 0; head < heads; ++head) {
        float frequency = 1.0f;
        for (uint32_t dimension = 0; dimension < head_dim; dimension += 2u) {
            float sine, cosine;
            fast_sincos((float)position * frequency, &sine, &cosine);
            uint32_t offset = head * head_dim + dimension;
            float even = values[offset];
            float odd = values[offset + 1u];
            values[offset] = even * cosine - odd * sine;
            values[offset + 1u] = even * sine + odd * cosine;
            frequency *= frequency_ratio;
        }
    }
}

static void prepare_qkv(struct lm_runtime *runtime, struct ms_state *state,
                        const struct ms_plain_block *block,
                        const struct ms_body_block *body, uint32_t loop,
                        const float *loop_embedding, uint32_t count) {
    const struct litemodel_header *header = runtime->header;
    uint32_t hidden = header->hidden_size;
    uint32_t qkv_size = hidden + 2u * header->num_kv_heads * header->head_dim;
    for (uint32_t time = 0; time < count; ++time) {
        float *residual = &state->x[time * hidden];
        if (loop_embedding) {
            for (uint32_t index = 0; index < hidden; ++index)
                state->sublayer_output[index] = residual[index] + loop_embedding[index];
            lm_rms_norm(runtime, state->normalized, state->sublayer_output,
                        block->attention_norm, hidden);
        } else {
            lm_rms_norm(runtime, state->normalized, residual,
                        block->attention_norm, hidden);
        }
        float *qkv = &state->qkv[time * qkv_size];
        lm_matvec(&block->qkv, state->normalized, qkv);
        if (body) {
            lm_matvec(&body->lora_down[loop], state->sublayer_output, state->lora_rank);
            lm_matvec(&body->lora_up[loop], state->lora_rank, state->lora_delta);
            for (uint32_t index = 0; index < qkv_size; ++index)
                qkv[index] += state->lora_delta[index];
        }
        apply_rope(qkv, header->num_heads, time);
        apply_rope(qkv + hidden, header->num_kv_heads, time);
    }
}

static void attend(struct lm_runtime *runtime, struct ms_state *state,
                   uint32_t position, uint32_t count) {
    const struct litemodel_header *header = runtime->header;
    uint32_t hidden = header->hidden_size;
    uint32_t head_dim = header->head_dim;
    uint32_t kv_size = header->num_kv_heads * head_dim;
    uint32_t qkv_size = hidden + 2u * kv_size;
    uint32_t repeats = header->num_heads / header->num_kv_heads;
    float scale = 1.0f / lm_sqrt((float)head_dim);
    (void)count;
    for (uint32_t head = 0; head < header->num_heads; ++head) {
        uint32_t kv_head = head / repeats;
        float maximum = -1000000.0f;
        for (uint32_t time = 0; time <= position; ++time) {
            if (state->documents[time] != state->documents[position]) {
                state->scores[time] = -1000000.0f;
                continue;
            }
            float score = 0.0f;
            const float *query = &state->qkv[position * qkv_size + head * head_dim];
            const float *key = &state->qkv[
                time * qkv_size + hidden + kv_head * head_dim
            ];
            for (uint32_t dimension = 0; dimension < head_dim; ++dimension)
                score += query[dimension] * key[dimension];
            state->scores[time] = score * scale;
            if (state->scores[time] > maximum) maximum = state->scores[time];
        }
        float sum = 0.0f;
        for (uint32_t time = 0; time <= position; ++time) {
            if (state->documents[time] != state->documents[position]) {
                state->scores[time] = 0.0f;
                continue;
            }
            state->scores[time] = lm_exp(state->scores[time] - maximum);
            sum += state->scores[time];
        }
        for (uint32_t dimension = 0; dimension < head_dim; ++dimension) {
            float value = 0.0f;
            for (uint32_t time = 0; time <= position; ++time) {
                const float *v = &state->qkv[
                    time * qkv_size + hidden + kv_size + kv_head * head_dim
                ];
                value += state->scores[time] * v[dimension];
            }
            state->attention[head * head_dim + dimension] = value / sum;
        }
    }
}

static void apply_deep_delta(struct lm_runtime *runtime, struct ms_state *state,
                             float *residual, const float *sublayer_output,
                             const float *context, const struct ms_deep_delta *delta) {
    uint32_t hidden = runtime->header->hidden_size;
    float sum = 0.0f;
    for (uint32_t index = 0; index < hidden; ++index)
        sum += sublayer_output[index] * sublayer_output[index];
    float epsilon = state->config.ddl_k_eps * state->config.ddl_k_eps / (float)hidden;
    float inverse = 1.0f / lm_sqrt(sum / (float)hidden + epsilon);
    for (uint32_t index = 0; index < hidden; ++index)
        state->attention[index] = sublayer_output[index] * inverse;
    float beta_value[1];
    float target_value[1];
    lm_matvec(&delta->beta, context, beta_value);
    lm_matvec(&delta->value, residual, target_value);
    float beta = 2.0f * lm_sigmoid(beta_value[0] + delta->beta_bias[0]);
    float target = lm_sigmoid(target_value[0] + delta->value_bias[0]) *
        state->config.ddl_v_scale;
    float scale = 1.0f / lm_sqrt((float)hidden);
    float projection = 0.0f;
    for (uint32_t index = 0; index < hidden; ++index)
        projection += state->attention[index] * residual[index];
    projection *= scale;
    float adjustment = beta * (target - projection) * scale;
    for (uint32_t index = 0; index < hidden; ++index)
        residual[index] += adjustment * state->attention[index];
}

static void run_block(struct lm_runtime *runtime, struct ms_state *state,
                      const struct ms_plain_block *block,
                      const struct ms_body_block *body, uint32_t loop,
                      const float *loop_embedding, uint32_t count) {
    const struct litemodel_header *header = runtime->header;
    uint32_t hidden = header->hidden_size;
    uint32_t intermediate = header->intermediate_size;
    prepare_qkv(runtime, state, block, body, loop, loop_embedding, count);
    for (uint32_t time = 0; time < count; ++time) {
        float *residual = &state->x[time * hidden];
        attend(runtime, state, time, count);
        lm_matvec(&block->attention_output, state->attention, state->sublayer_output);
        if (body) {
            for (uint32_t index = 0; index < hidden; ++index)
                state->normalized[index] = residual[index] + loop_embedding[index];
            lm_rms_norm(runtime, state->normalized, state->normalized,
                        block->attention_norm, hidden);
            apply_deep_delta(runtime, state, residual, state->sublayer_output,
                             state->normalized, &body->attention_delta);
        } else {
            for (uint32_t index = 0; index < hidden; ++index)
                residual[index] += state->sublayer_output[index];
        }

        if (body) {
            for (uint32_t index = 0; index < hidden; ++index)
                state->normalized[index] = residual[index] + loop_embedding[index];
            lm_rms_norm(runtime, state->normalized, state->normalized,
                        block->ffn_norm, hidden);
        } else {
            lm_rms_norm(runtime, state->normalized, residual,
                        block->ffn_norm, hidden);
        }
        lm_matvec(&block->gate_up, state->normalized, state->gate_up);
        for (uint32_t index = 0; index < intermediate; ++index)
            state->gate_up[index] = state->gate_up[index] *
                lm_sigmoid(state->gate_up[index]) * state->gate_up[intermediate + index];
        lm_matvec(&block->down, state->gate_up, state->sublayer_output);
        if (body) {
            apply_deep_delta(runtime, state, residual, state->sublayer_output,
                             state->normalized, &body->ffn_delta);
        } else {
            for (uint32_t index = 0; index < hidden; ++index)
                residual[index] += state->sublayer_output[index];
        }
    }
}

uint32_t lm_arch_minspark_forward(struct lm_runtime *runtime,
                                  const uint32_t *tokens, uint32_t count) {
    struct ms_state *state = runtime->architecture_state;
    const struct litemodel_header *header = runtime->header;
    if (!state || !count || count > runtime->context_capacity) return 0u;
    uint32_t hidden = header->hidden_size;
    uint32_t document = 0u;
    for (uint32_t time = 0; time < count; ++time) {
        if (tokens[time] >= header->vocab_size) return 0u;
        lm_matrix_row(&state->embedding, tokens[time], &state->x[time * hidden]);
        state->documents[time] = document;
        if (state->config.document_eos != 0xFFFFFFFFu &&
            tokens[time] == state->config.document_eos) ++document;
    }
    for (uint32_t index = 0; index < state->config.prelude_layers; ++index)
        run_block(runtime, state, &state->prelude[index], 0, 0u, 0, count);
    for (uint32_t loop = 0; loop < state->active_loops; ++loop) {
        lm_matrix_row(&state->loop_embedding, loop, state->loop_embedding_values);
        for (uint32_t index = 0; index < state->config.body_blocks; ++index)
            run_block(runtime, state, &state->body[index].block,
                      &state->body[index], loop, state->loop_embedding_values, count);
    }
    for (uint32_t index = 0; index < state->config.coda_layers; ++index)
        run_block(runtime, state, &state->coda[index], 0, 0u, 0, count);
    float *last = &state->x[(count - 1u) * hidden];
    lm_rms_norm(runtime, state->normalized, last, state->final_norm, hidden);
    lm_matvec(&state->embedding, state->normalized, state->logits);
    uint32_t best = 0u;
    for (uint32_t index = 1; index < header->vocab_size; ++index)
        if (state->logits[index] > state->logits[best]) best = index;
    return best;
}

int lm_arch_minspark_set_effort(struct lm_runtime *runtime, uint32_t loops) {
    struct ms_state *state = runtime->architecture_state;
    if (!state || runtime->header->architecture != LITEMODEL_ARCH_MINSPARK ||
        loops < 1u || loops > state->config.max_loops) return 0;
    state->active_loops = loops;
    return 1;
}

uint32_t lm_arch_minspark_get_effort(const struct lm_runtime *runtime) {
    const struct ms_state *state = runtime->architecture_state;
    if (!state || runtime->header->architecture != LITEMODEL_ARCH_MINSPARK) return 0u;
    return state->active_loops;
}
