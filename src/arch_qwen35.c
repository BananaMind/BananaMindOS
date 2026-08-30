#include <stdint.h>

#include "litemodel_runtime.h"

/* Qwen 3.5 text inference only. Vision tensors are intentionally omitted by
   the converter; the language stack alternates recurrent gated-delta layers
   with conventional gated full-attention layers. */
struct __attribute__((packed)) qwen35_file_config {
    uint32_t full_attention_mask;
    uint32_t conv_kernel;
    uint32_t key_heads;
    uint32_t value_heads;
    uint32_t key_head_dim;
    uint32_t value_head_dim;
    uint32_t rotary_dim;
    float rope_ratio;
};

struct qwen35_linear {
    struct lm_matrix qkv;
    struct lm_matrix z;
    struct lm_matrix beta;
    struct lm_matrix decay;
    struct lm_matrix convolution;
    const float *dt_bias;
    const float *a_log;
    const float *norm;
    struct lm_matrix output;
};

struct qwen35_attention {
    struct lm_matrix query_gate;
    struct lm_matrix key;
    struct lm_matrix value;
    struct lm_matrix output;
    const float *q_norm;
    const float *k_norm;
};

struct qwen35_layer {
    const float *input_norm;
    const float *post_attention_norm;
    struct qwen35_linear linear;
    struct qwen35_attention attention;
    struct lm_matrix gate;
    struct lm_matrix up;
    struct lm_matrix down;
};

struct qwen35_state {
    struct qwen35_file_config config;
    struct lm_matrix embedding;
    struct qwen35_layer layers[LM_MAX_LAYERS];
    const float *final_norm;
    float *key_cache;
    float *value_cache;
    float *conv_history;
    float *recurrent;
    float *x;
    float *normalized;
    float *temporary;
    float *projection;
    float *z;
    float *query;
    float *key;
    float *value;
    float *attention;
    float *gate;
    float *up;
    float *scores;
    float *logits;
    float *beta;
    float *decay;
    float kernel_values[8];
};

static struct qwen35_state qwen35;

static void zero_bytes(void *memory_, uint32_t count) {
    uint8_t *memory = memory_;
    while (count--) *memory++ = 0u;
}

static void fast_sincos(float angle, float *sine, float *cosine) {
    const float pi = 3.14159265359f, tau = 6.28318530718f;
    while (angle > pi) angle -= tau;
    while (angle < -pi) angle += tau;
    float squared = angle * angle;
    *sine = angle * (1.0f + squared *
        (-0.16666667f + squared * (0.00833333f + squared * -0.00019841f)));
    *cosine = 1.0f + squared *
        (-0.5f + squared * (0.04166667f + squared * -0.00138889f));
}

static void apply_partial_rope(float *values, uint32_t heads, uint32_t head_dim,
                               uint32_t rotary_dim, uint32_t position, float ratio) {
    uint32_t half = rotary_dim / 2u;
    for (uint32_t head = 0u; head < heads; ++head) {
        float frequency = 1.0f;
        for (uint32_t index = 0u; index < half; ++index) {
            float sine, cosine;
            fast_sincos((float)position * frequency, &sine, &cosine);
            uint32_t first = head * head_dim + index;
            uint32_t second = first + half;
            float a = values[first], b = values[second];
            values[first] = a * cosine - b * sine;
            values[second] = b * cosine + a * sine;
            frequency *= ratio;
        }
    }
}

static uint32_t cache_index(const struct lm_runtime *runtime, uint32_t layer,
                            uint32_t position, uint32_t head, uint32_t dimension) {
    return ((((layer * runtime->context_capacity + position) *
              runtime->header->num_kv_heads + head) *
             runtime->header->head_dim) + dimension);
}

static uint32_t conv_index(const struct qwen35_state *state,
                           uint32_t layer, uint32_t channel, uint32_t slot) {
    uint32_t key_dim = state->config.key_heads * state->config.key_head_dim;
    uint32_t value_dim = state->config.value_heads * state->config.value_head_dim;
    uint32_t conv_dim = key_dim * 2u + value_dim;
    return (layer * conv_dim + channel) * (state->config.conv_kernel - 1u) + slot;
}

static uint32_t recurrent_index(const struct qwen35_state *state, uint32_t layer,
                                uint32_t head, uint32_t key, uint32_t value) {
    return (((layer * state->config.value_heads + head) * state->config.key_head_dim + key) *
            state->config.value_head_dim + value);
}

int lm_arch_qwen35_load(struct lm_runtime *runtime, const uint8_t *config,
                        uint32_t config_size, const uint8_t *weights,
                        const uint8_t *weights_end) {
    if (config_size != sizeof(struct qwen35_file_config)) return 0;
    zero_bytes(&qwen35, sizeof(qwen35));
    qwen35.config = *(const struct qwen35_file_config *)config;
    const struct litemodel_header *header = runtime->header;
    uint32_t hidden = header->hidden_size;
    uint32_t key_dim = qwen35.config.key_heads * qwen35.config.key_head_dim;
    uint32_t value_dim = qwen35.config.value_heads * qwen35.config.value_head_dim;
    uint32_t conv_dim = key_dim * 2u + value_dim;
    uint32_t full_query = header->num_heads * header->head_dim;
    uint32_t full_kv = header->num_kv_heads * header->head_dim;
    if (qwen35.config.conv_kernel < 2u || qwen35.config.conv_kernel > 8u ||
        !qwen35.config.key_heads || !qwen35.config.value_heads ||
        qwen35.config.value_heads % qwen35.config.key_heads ||
        qwen35.config.rotary_dim > header->head_dim ||
        (qwen35.config.rotary_dim & 1u))
        return 0;
    const uint8_t *position = lm_parse_matrix(
        weights, weights_end, header->vocab_size, hidden,
        header->quant_bits, &qwen35.embedding
    );
    for (uint32_t index = 0u; position && index < header->num_layers; ++index) {
        struct qwen35_layer *layer = &qwen35.layers[index];
        position = lm_parse_vector(position, weights_end, hidden, &layer->input_norm);
        if (qwen35.config.full_attention_mask & (1u << index)) {
            if (position) position = lm_parse_matrix(position, weights_end, full_query * 2u,
                                                      hidden, header->quant_bits,
                                                      &layer->attention.query_gate);
            if (position) position = lm_parse_matrix(position, weights_end, full_kv, hidden,
                                                      header->quant_bits, &layer->attention.key);
            if (position) position = lm_parse_matrix(position, weights_end, full_kv, hidden,
                                                      header->quant_bits, &layer->attention.value);
            if (position) position = lm_parse_matrix(position, weights_end, hidden, full_query,
                                                      header->quant_bits, &layer->attention.output);
            if (position) position = lm_parse_vector(position, weights_end, header->head_dim,
                                                      &layer->attention.q_norm);
            if (position) position = lm_parse_vector(position, weights_end, header->head_dim,
                                                      &layer->attention.k_norm);
        } else {
            if (position) position = lm_parse_matrix(position, weights_end, conv_dim, hidden,
                                                      header->quant_bits, &layer->linear.qkv);
            if (position) position = lm_parse_matrix(position, weights_end, value_dim, hidden,
                                                      header->quant_bits, &layer->linear.z);
            if (position) position = lm_parse_matrix(position, weights_end,
                                                      qwen35.config.value_heads, hidden,
                                                      header->quant_bits, &layer->linear.beta);
            if (position) position = lm_parse_matrix(position, weights_end,
                                                      qwen35.config.value_heads, hidden,
                                                      header->quant_bits, &layer->linear.decay);
            if (position) position = lm_parse_matrix(position, weights_end, conv_dim,
                                                      qwen35.config.conv_kernel,
                                                      header->quant_bits,
                                                      &layer->linear.convolution);
            if (position) position = lm_parse_vector(position, weights_end,
                                                      qwen35.config.value_heads,
                                                      &layer->linear.dt_bias);
            if (position) position = lm_parse_vector(position, weights_end,
                                                      qwen35.config.value_heads,
                                                      &layer->linear.a_log);
            if (position) position = lm_parse_vector(position, weights_end,
                                                      qwen35.config.value_head_dim,
                                                      &layer->linear.norm);
            if (position) position = lm_parse_matrix(position, weights_end, hidden, value_dim,
                                                      header->quant_bits, &layer->linear.output);
        }
        if (position) position = lm_parse_vector(position, weights_end, hidden,
                                                  &layer->post_attention_norm);
        if (position) position = lm_parse_matrix(position, weights_end,
                                                  header->intermediate_size, hidden,
                                                  header->quant_bits, &layer->gate);
        if (position) position = lm_parse_matrix(position, weights_end,
                                                  header->intermediate_size, hidden,
                                                  header->quant_bits, &layer->up);
        if (position) position = lm_parse_matrix(position, weights_end, hidden,
                                                  header->intermediate_size,
                                                  header->quant_bits, &layer->down);
    }
    if (position) position = lm_parse_vector(position, weights_end, hidden, &qwen35.final_norm);
    if (!position || position != weights_end) return 0;
    runtime->embedding = qwen35.embedding;
    runtime->final_norm = qwen35.final_norm;
    runtime->architecture_state = &qwen35;
    return 1;
}

int lm_arch_qwen35_allocate(struct lm_runtime *runtime, struct lm_arena *arena) {
    struct qwen35_state *state = runtime->architecture_state;
    const struct litemodel_header *header = runtime->header;
    uint32_t hidden = header->hidden_size;
    uint32_t full_query = header->num_heads * header->head_dim;
    uint32_t full_kv = header->num_kv_heads * header->head_dim;
    uint32_t key_dim = state->config.key_heads * state->config.key_head_dim;
    uint32_t value_dim = state->config.value_heads * state->config.value_head_dim;
    uint32_t conv_dim = key_dim * 2u + value_dim;
    uint32_t cache = header->num_layers * runtime->context_capacity * full_kv;
    uint32_t conv_history = header->num_layers * conv_dim * (state->config.conv_kernel - 1u);
    uint32_t recurrent = header->num_layers * state->config.value_heads *
        state->config.key_head_dim * state->config.value_head_dim;
    uint32_t largest_projection = conv_dim > full_query * 2u ? conv_dim : full_query * 2u;
    state->key_cache = lm_arena_alloc(arena, cache * 4u, 16u);
    state->value_cache = lm_arena_alloc(arena, cache * 4u, 16u);
    state->conv_history = lm_arena_alloc(arena, conv_history * 4u, 16u);
    state->recurrent = lm_arena_alloc(arena, recurrent * 4u, 16u);
    state->x = lm_arena_alloc(arena, hidden * 4u, 16u);
    state->normalized = lm_arena_alloc(arena, hidden * 4u, 16u);
    state->temporary = lm_arena_alloc(arena, hidden * 4u, 16u);
    state->projection = lm_arena_alloc(arena, largest_projection * 4u, 16u);
    state->z = lm_arena_alloc(arena, value_dim * 4u, 16u);
    state->query = lm_arena_alloc(arena, full_query * 4u, 16u);
    state->key = lm_arena_alloc(arena, (key_dim > full_kv ? key_dim : full_kv) * 4u, 16u);
    state->value = lm_arena_alloc(arena, (value_dim > full_kv ? value_dim : full_kv) * 4u, 16u);
    state->attention = lm_arena_alloc(arena, (value_dim > full_query ? value_dim : full_query) * 4u, 16u);
    state->gate = lm_arena_alloc(arena, header->intermediate_size * 4u, 16u);
    state->up = lm_arena_alloc(arena, header->intermediate_size * 4u, 16u);
    state->scores = lm_arena_alloc(arena, runtime->context_capacity * 4u, 16u);
    state->logits = lm_arena_alloc(arena, header->vocab_size * 4u, 16u);
    state->beta = lm_arena_alloc(arena, state->config.value_heads * 4u, 16u);
    state->decay = lm_arena_alloc(arena, state->config.value_heads * 4u, 16u);
    runtime->logits = state->logits;
    return state->key_cache && state->value_cache && state->conv_history &&
        state->recurrent && state->x && state->normalized && state->temporary &&
        state->projection && state->z && state->query && state->key && state->value &&
        state->attention && state->gate && state->up && state->scores && state->logits &&
        state->beta && state->decay;
}

void lm_arch_qwen35_reset(struct lm_runtime *runtime) {
    struct qwen35_state *state = runtime->architecture_state;
    if (!state || !state->conv_history || !state->recurrent) return;
    uint32_t key_dim = state->config.key_heads * state->config.key_head_dim;
    uint32_t value_dim = state->config.value_heads * state->config.value_head_dim;
    uint32_t conv_dim = key_dim * 2u + value_dim;
    uint32_t conv_count = runtime->header->num_layers * conv_dim *
        (state->config.conv_kernel - 1u);
    uint32_t recurrent_count = runtime->header->num_layers * state->config.value_heads *
        state->config.key_head_dim * state->config.value_head_dim;
    zero_bytes(state->conv_history, conv_count * 4u);
    zero_bytes(state->recurrent, recurrent_count * 4u);
}

static void l2_normalize(float *values, uint32_t size) {
    float sum = 0.0f;
    for (uint32_t index = 0u; index < size; ++index) sum += values[index] * values[index];
    float inverse = 1.0f / lm_sqrt(sum + 0.000001f);
    for (uint32_t index = 0u; index < size; ++index) values[index] *= inverse;
}

static void run_linear(struct lm_runtime *runtime, struct qwen35_state *state,
                       struct qwen35_layer *layer, uint32_t layer_index) {
    uint32_t key_heads = state->config.key_heads;
    uint32_t value_heads = state->config.value_heads;
    uint32_t key_head_dim = state->config.key_head_dim;
    uint32_t value_head_dim = state->config.value_head_dim;
    uint32_t key_dim = key_heads * key_head_dim;
    uint32_t value_dim = value_heads * value_head_dim;
    uint32_t conv_dim = key_dim * 2u + value_dim;
    uint32_t kernel = state->config.conv_kernel;
    lm_matvec(&layer->linear.qkv, state->normalized, state->projection);
    lm_matvec(&layer->linear.z, state->normalized, state->z);
    lm_matvec(&layer->linear.beta, state->normalized, state->beta);
    lm_matvec(&layer->linear.decay, state->normalized, state->decay);
    for (uint32_t channel = 0u; channel < conv_dim; ++channel) {
        lm_matrix_row(&layer->linear.convolution, channel, state->kernel_values);
        float convolved = state->kernel_values[kernel - 1u] * state->projection[channel];
        for (uint32_t slot = 0u; slot + 1u < kernel; ++slot)
            convolved += state->kernel_values[slot] *
                state->conv_history[conv_index(state, layer_index, channel, slot)];
        for (uint32_t slot = 0u; slot + 2u < kernel; ++slot)
            state->conv_history[conv_index(state, layer_index, channel, slot)] =
                state->conv_history[conv_index(state, layer_index, channel, slot + 1u)];
        state->conv_history[conv_index(state, layer_index, channel, kernel - 2u)] =
            state->projection[channel];
        state->projection[channel] = convolved * lm_sigmoid(convolved);
    }
    for (uint32_t head = 0u; head < key_heads; ++head) {
        l2_normalize(state->projection + head * key_head_dim, key_head_dim);
        l2_normalize(state->projection + key_dim + head * key_head_dim, key_head_dim);
    }
    for (uint32_t head = 0u; head < value_heads; ++head) {
        state->beta[head] = lm_sigmoid(state->beta[head]);
        float g = -lm_exp(layer->linear.a_log[head]) *
            lm_softplus(state->decay[head] + layer->linear.dt_bias[head]);
        state->decay[head] = lm_exp(g);
        uint32_t key_head = head / (value_heads / key_heads);
        float *query = state->projection + key_head * key_head_dim;
        float *key = state->projection + key_dim + key_head * key_head_dim;
        float *value = state->projection + key_dim * 2u + head * value_head_dim;
        float *output = state->attention + head * value_head_dim;
        for (uint32_t value_index = 0u; value_index < value_head_dim; ++value_index) {
            float predicted = 0.0f;
            for (uint32_t key_index = 0u; key_index < key_head_dim; ++key_index) {
                uint32_t cell = recurrent_index(state, layer_index, head,
                                                key_index, value_index);
                state->recurrent[cell] *= state->decay[head];
                predicted += state->recurrent[cell] * key[key_index];
            }
            state->value[value_index] = (value[value_index] - predicted) * state->beta[head];
        }
        for (uint32_t key_index = 0u; key_index < key_head_dim; ++key_index)
            for (uint32_t value_index = 0u; value_index < value_head_dim; ++value_index)
                state->recurrent[recurrent_index(state, layer_index, head,
                                                  key_index, value_index)] +=
                    key[key_index] * state->value[value_index];
        for (uint32_t value_index = 0u; value_index < value_head_dim; ++value_index) {
            float result = 0.0f;
            for (uint32_t key_index = 0u; key_index < key_head_dim; ++key_index)
                result += state->recurrent[recurrent_index(state, layer_index, head,
                                                            key_index, value_index)] *
                    query[key_index];
            output[value_index] = result;
        }
        lm_rms_norm(runtime, output, output, layer->linear.norm, value_head_dim);
        for (uint32_t value_index = 0u; value_index < value_head_dim; ++value_index) {
            float gate = state->z[head * value_head_dim + value_index];
            output[value_index] *= gate * lm_sigmoid(gate);
        }
    }
    lm_matvec(&layer->linear.output, state->attention, state->temporary);
}

static void run_full_attention(struct lm_runtime *runtime, struct qwen35_state *state,
                               struct qwen35_layer *layer, uint32_t layer_index,
                               uint32_t position) {
    const struct litemodel_header *header = runtime->header;
    uint32_t heads = header->num_heads;
    uint32_t kv_heads = header->num_kv_heads;
    uint32_t head_dim = header->head_dim;
    uint32_t query_dim = heads * head_dim;
    lm_matvec(&layer->attention.query_gate, state->normalized, state->projection);
    for (uint32_t head = 0u; head < heads; ++head)
        for (uint32_t dimension = 0u; dimension < head_dim; ++dimension) {
            state->query[head * head_dim + dimension] =
                state->projection[head * head_dim * 2u + dimension];
            state->z[head * head_dim + dimension] =
                state->projection[head * head_dim * 2u + head_dim + dimension];
        }
    lm_matvec(&layer->attention.key, state->normalized, state->key);
    lm_matvec(&layer->attention.value, state->normalized, state->value);
    for (uint32_t head = 0u; head < heads; ++head)
        lm_rms_norm_offset(runtime, state->query + head * head_dim,
                           state->query + head * head_dim,
                           layer->attention.q_norm, head_dim, 1.0f);
    for (uint32_t head = 0u; head < kv_heads; ++head)
        lm_rms_norm_offset(runtime, state->key + head * head_dim,
                           state->key + head * head_dim,
                           layer->attention.k_norm, head_dim, 1.0f);
    apply_partial_rope(state->query, heads, head_dim, state->config.rotary_dim,
                       position, state->config.rope_ratio);
    apply_partial_rope(state->key, kv_heads, head_dim, state->config.rotary_dim,
                       position, state->config.rope_ratio);
    for (uint32_t head = 0u; head < kv_heads; ++head)
        for (uint32_t dimension = 0u; dimension < head_dim; ++dimension) {
            uint32_t cache = cache_index(runtime, layer_index, position, head, dimension);
            state->key_cache[cache] = state->key[head * head_dim + dimension];
            state->value_cache[cache] = state->value[head * head_dim + dimension];
        }
    uint32_t repeats = heads / kv_heads;
    float scale = 1.0f / lm_sqrt((float)head_dim);
    for (uint32_t head = 0u; head < heads; ++head) {
        uint32_t kv_head = head / repeats;
        float maximum = -1000000.0f;
        for (uint32_t time = 0u; time <= position; ++time) {
            float score = 0.0f;
            for (uint32_t dimension = 0u; dimension < head_dim; ++dimension)
                score += state->query[head * head_dim + dimension] *
                    state->key_cache[cache_index(runtime, layer_index, time,
                                                  kv_head, dimension)];
            state->scores[time] = score * scale;
            if (state->scores[time] > maximum) maximum = state->scores[time];
        }
        float sum = 0.0f;
        for (uint32_t time = 0u; time <= position; ++time) {
            state->scores[time] = lm_exp(state->scores[time] - maximum);
            sum += state->scores[time];
        }
        for (uint32_t dimension = 0u; dimension < head_dim; ++dimension) {
            float value = 0.0f;
            for (uint32_t time = 0u; time <= position; ++time)
                value += state->scores[time] *
                    state->value_cache[cache_index(runtime, layer_index, time,
                                                    kv_head, dimension)];
            uint32_t index = head * head_dim + dimension;
            state->attention[index] = value / sum * lm_sigmoid(state->z[index]);
        }
    }
    lm_matvec(&layer->attention.output, state->attention, state->temporary);
    (void)query_dim;
}

uint32_t lm_arch_qwen35_forward(struct lm_runtime *runtime, uint32_t token,
                                uint32_t position) {
    struct qwen35_state *state = runtime->architecture_state;
    const struct litemodel_header *header = runtime->header;
    if (!state || token >= header->vocab_size || position >= runtime->context_capacity) return 0u;
    uint32_t hidden = header->hidden_size;
    lm_matrix_row(&state->embedding, token, state->x);
    for (uint32_t layer_index = 0u; layer_index < header->num_layers; ++layer_index) {
        struct qwen35_layer *layer = &state->layers[layer_index];
        lm_rms_norm_offset(runtime, state->normalized, state->x,
                           layer->input_norm, hidden, 1.0f);
        if (state->config.full_attention_mask & (1u << layer_index))
            run_full_attention(runtime, state, layer, layer_index, position);
        else
            run_linear(runtime, state, layer, layer_index);
        for (uint32_t index = 0u; index < hidden; ++index) state->x[index] += state->temporary[index];
        lm_rms_norm_offset(runtime, state->normalized, state->x,
                           layer->post_attention_norm, hidden, 1.0f);
        lm_matvec(&layer->gate, state->normalized, state->gate);
        lm_matvec(&layer->up, state->normalized, state->up);
        for (uint32_t index = 0u; index < header->intermediate_size; ++index)
            state->gate[index] = state->gate[index] * lm_sigmoid(state->gate[index]) * state->up[index];
        lm_matvec(&layer->down, state->gate, state->temporary);
        for (uint32_t index = 0u; index < hidden; ++index) state->x[index] += state->temporary[index];
    }
    lm_rms_norm_offset(runtime, state->normalized, state->x,
                       state->final_norm, hidden, 1.0f);
    lm_matvec(&state->embedding, state->normalized, state->logits);
    uint32_t best = 0u;
    for (uint32_t index = 1u; index < header->vocab_size; ++index)
        if (state->logits[index] > state->logits[best]) best = index;
    return best;
}
