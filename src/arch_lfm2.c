#include <stdint.h>

#include "litemodel_runtime.h"

/* LiquidAI LFM2 stays isolated from the ordinary transformer parser because
   its token mixer alternates depthwise short convolutions and attention. */
struct __attribute__((packed)) lfm2_file_config {
    uint32_t attention_mask;
    uint32_t conv_kernel;
    float rope_ratio;
};

struct lfm2_layer {
    const float *operator_norm;
    const float *q_norm;
    const float *k_norm;
    const float *ffn_norm;
    struct lm_matrix query;
    struct lm_matrix key;
    struct lm_matrix value;
    struct lm_matrix mixer_output;
    struct lm_matrix conv_input;
    struct lm_matrix conv_kernel;
    struct lm_matrix gate;
    struct lm_matrix up;
    struct lm_matrix down;
};

struct lfm2_state {
    struct lfm2_file_config config;
    struct lm_matrix embedding;
    struct lfm2_layer layers[LM_MAX_LAYERS];
    const float *final_norm;
    float *key_cache;
    float *value_cache;
    float *conv_history;
    float *x;
    float *normalized;
    float *temporary;
    float *projection;
    float *query;
    float *key;
    float *value;
    float *attention;
    float *gate;
    float *up;
    float *scores;
    float *logits;
    float kernel_values[8];
};

static struct lfm2_state lfm2;

static void zero_bytes(void *memory_, uint32_t count) {
    uint8_t *memory = memory_;
    while (count--) *memory++ = 0u;
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

static void apply_rope(float *values, uint32_t heads, uint32_t head_dim,
                       uint32_t position, float ratio) {
    uint32_t half = head_dim / 2u;
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

static uint32_t conv_index(const struct lm_runtime *runtime,
                           const struct lfm2_state *state, uint32_t layer,
                           uint32_t channel, uint32_t slot) {
    uint32_t history = state->config.conv_kernel - 1u;
    return (layer * runtime->header->hidden_size + channel) * history + slot;
}

int lm_arch_lfm2_load(struct lm_runtime *runtime, const uint8_t *config,
                      uint32_t config_size, const uint8_t *weights,
                      const uint8_t *weights_end) {
    if (config_size != sizeof(struct lfm2_file_config)) return 0;
    zero_bytes(&lfm2, sizeof(lfm2));
    lfm2.config = *(const struct lfm2_file_config *)config;
    const struct litemodel_header *header = runtime->header;
    if (lfm2.config.conv_kernel < 2u || lfm2.config.conv_kernel > 8u ||
        header->num_heads * header->head_dim != header->hidden_size)
        return 0;
    uint32_t hidden = header->hidden_size;
    uint32_t kv = header->num_kv_heads * header->head_dim;
    const uint8_t *position = lm_parse_matrix(
        weights, weights_end, header->vocab_size, hidden,
        header->quant_bits, &lfm2.embedding
    );
    for (uint32_t index = 0u; position && index < header->num_layers; ++index) {
        struct lfm2_layer *layer = &lfm2.layers[index];
        position = lm_parse_vector(position, weights_end, hidden, &layer->operator_norm);
        if (lfm2.config.attention_mask & (1u << index)) {
            if (position) position = lm_parse_matrix(position, weights_end, hidden, hidden,
                                                      header->quant_bits, &layer->query);
            if (position) position = lm_parse_matrix(position, weights_end, kv, hidden,
                                                      header->quant_bits, &layer->key);
            if (position) position = lm_parse_matrix(position, weights_end, kv, hidden,
                                                      header->quant_bits, &layer->value);
            if (position) position = lm_parse_matrix(position, weights_end, hidden, hidden,
                                                      header->quant_bits, &layer->mixer_output);
            if (position) position = lm_parse_vector(position, weights_end,
                                                      header->head_dim, &layer->q_norm);
            if (position) position = lm_parse_vector(position, weights_end,
                                                      header->head_dim, &layer->k_norm);
        } else {
            if (position) position = lm_parse_matrix(position, weights_end, hidden * 3u,
                                                      hidden, header->quant_bits,
                                                      &layer->conv_input);
            if (position) position = lm_parse_matrix(position, weights_end, hidden,
                                                      lfm2.config.conv_kernel,
                                                      header->quant_bits, &layer->conv_kernel);
            if (position) position = lm_parse_matrix(position, weights_end, hidden, hidden,
                                                      header->quant_bits, &layer->mixer_output);
        }
        if (position) position = lm_parse_vector(position, weights_end, hidden,
                                                  &layer->ffn_norm);
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
    if (position) position = lm_parse_vector(position, weights_end, hidden, &lfm2.final_norm);
    if (!position || position != weights_end) return 0;
    runtime->embedding = lfm2.embedding;
    runtime->final_norm = lfm2.final_norm;
    runtime->architecture_state = &lfm2;
    return 1;
}

int lm_arch_lfm2_allocate(struct lm_runtime *runtime, struct lm_arena *arena) {
    struct lfm2_state *state = runtime->architecture_state;
    const struct litemodel_header *header = runtime->header;
    uint32_t hidden = header->hidden_size;
    uint32_t kv = header->num_kv_heads * header->head_dim;
    uint32_t cache = header->num_layers * runtime->context_capacity * kv;
    uint32_t history = header->num_layers * hidden * (state->config.conv_kernel - 1u);
    state->key_cache = lm_arena_alloc(arena, cache * 4u, 16u);
    state->value_cache = lm_arena_alloc(arena, cache * 4u, 16u);
    state->conv_history = lm_arena_alloc(arena, history * 4u, 16u);
    state->x = lm_arena_alloc(arena, hidden * 4u, 16u);
    state->normalized = lm_arena_alloc(arena, hidden * 4u, 16u);
    state->temporary = lm_arena_alloc(arena, hidden * 4u, 16u);
    state->projection = lm_arena_alloc(arena, hidden * 3u * 4u, 16u);
    state->query = lm_arena_alloc(arena, hidden * 4u, 16u);
    state->key = lm_arena_alloc(arena, kv * 4u, 16u);
    state->value = lm_arena_alloc(arena, kv * 4u, 16u);
    state->attention = lm_arena_alloc(arena, hidden * 4u, 16u);
    state->gate = lm_arena_alloc(arena, header->intermediate_size * 4u, 16u);
    state->up = lm_arena_alloc(arena, header->intermediate_size * 4u, 16u);
    state->scores = lm_arena_alloc(arena, runtime->context_capacity * 4u, 16u);
    state->logits = lm_arena_alloc(arena, header->vocab_size * 4u, 16u);
    runtime->logits = state->logits;
    return state->key_cache && state->value_cache && state->conv_history &&
        state->x && state->normalized && state->temporary && state->projection &&
        state->query && state->key && state->value && state->attention &&
        state->gate && state->up && state->scores && state->logits;
}

void lm_arch_lfm2_reset(struct lm_runtime *runtime) {
    struct lfm2_state *state = runtime->architecture_state;
    if (!state || !state->conv_history) return;
    uint32_t count = runtime->header->num_layers * runtime->header->hidden_size *
        (state->config.conv_kernel - 1u);
    zero_bytes(state->conv_history, count * 4u);
}

static void run_attention(struct lm_runtime *runtime, struct lfm2_state *state,
                          struct lfm2_layer *layer, uint32_t layer_index,
                          uint32_t position) {
    const struct litemodel_header *header = runtime->header;
    uint32_t head_dim = header->head_dim;
    uint32_t kv_heads = header->num_kv_heads;
    lm_matvec(&layer->query, state->normalized, state->query);
    lm_matvec(&layer->key, state->normalized, state->key);
    lm_matvec(&layer->value, state->normalized, state->value);
    for (uint32_t head = 0u; head < header->num_heads; ++head)
        lm_rms_norm(runtime, state->query + head * head_dim,
                    state->query + head * head_dim, layer->q_norm, head_dim);
    for (uint32_t head = 0u; head < kv_heads; ++head)
        lm_rms_norm(runtime, state->key + head * head_dim,
                    state->key + head * head_dim, layer->k_norm, head_dim);
    apply_rope(state->query, header->num_heads, head_dim, position, state->config.rope_ratio);
    apply_rope(state->key, kv_heads, head_dim, position, state->config.rope_ratio);
    for (uint32_t head = 0u; head < kv_heads; ++head)
        for (uint32_t dimension = 0u; dimension < head_dim; ++dimension) {
            uint32_t cache = cache_index(runtime, layer_index, position, head, dimension);
            state->key_cache[cache] = state->key[head * head_dim + dimension];
            state->value_cache[cache] = state->value[head * head_dim + dimension];
        }
    uint32_t repeats = header->num_heads / kv_heads;
    float scale = 1.0f / lm_sqrt((float)head_dim);
    for (uint32_t head = 0u; head < header->num_heads; ++head) {
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
            state->attention[head * head_dim + dimension] = value / sum;
        }
    }
    lm_matvec(&layer->mixer_output, state->attention, state->temporary);
}

static void run_convolution(struct lm_runtime *runtime, struct lfm2_state *state,
                            struct lfm2_layer *layer, uint32_t layer_index) {
    uint32_t hidden = runtime->header->hidden_size;
    uint32_t kernel = state->config.conv_kernel;
    lm_matvec(&layer->conv_input, state->normalized, state->projection);
    for (uint32_t channel = 0u; channel < hidden; ++channel) {
        float product = state->projection[channel] * state->projection[hidden * 2u + channel];
        lm_matrix_row(&layer->conv_kernel, channel, state->kernel_values);
        float convolved = state->kernel_values[kernel - 1u] * product;
        for (uint32_t slot = 0u; slot + 1u < kernel; ++slot)
            convolved += state->kernel_values[slot] *
                state->conv_history[conv_index(runtime, state, layer_index, channel, slot)];
        for (uint32_t slot = 0u; slot + 2u < kernel; ++slot)
            state->conv_history[conv_index(runtime, state, layer_index, channel, slot)] =
                state->conv_history[conv_index(runtime, state, layer_index, channel, slot + 1u)];
        state->conv_history[conv_index(runtime, state, layer_index, channel, kernel - 2u)] = product;
        state->projection[channel] = state->projection[hidden + channel] * convolved;
    }
    lm_matvec(&layer->mixer_output, state->projection, state->temporary);
}

uint32_t lm_arch_lfm2_forward(struct lm_runtime *runtime, uint32_t token,
                              uint32_t position) {
    struct lfm2_state *state = runtime->architecture_state;
    const struct litemodel_header *header = runtime->header;
    if (!state || token >= header->vocab_size || position >= runtime->context_capacity) return 0u;
    uint32_t hidden = header->hidden_size;
    lm_matrix_row(&state->embedding, token, state->x);
    for (uint32_t layer_index = 0u; layer_index < header->num_layers; ++layer_index) {
        struct lfm2_layer *layer = &state->layers[layer_index];
        lm_rms_norm(runtime, state->normalized, state->x, layer->operator_norm, hidden);
        if (state->config.attention_mask & (1u << layer_index))
            run_attention(runtime, state, layer, layer_index, position);
        else
            run_convolution(runtime, state, layer, layer_index);
        for (uint32_t index = 0u; index < hidden; ++index) state->x[index] += state->temporary[index];
        lm_rms_norm(runtime, state->normalized, state->x, layer->ffn_norm, hidden);
        lm_matvec(&layer->gate, state->normalized, state->gate);
        lm_matvec(&layer->up, state->normalized, state->up);
        for (uint32_t index = 0u; index < header->intermediate_size; ++index)
            state->gate[index] = state->gate[index] * lm_sigmoid(state->gate[index]) * state->up[index];
        lm_matvec(&layer->down, state->gate, state->temporary);
        for (uint32_t index = 0u; index < hidden; ++index) state->x[index] += state->temporary[index];
    }
    lm_rms_norm(runtime, state->normalized, state->x, state->final_norm, hidden);
    lm_matvec(&state->embedding, state->normalized, state->logits);
    uint32_t best = 0u;
    for (uint32_t index = 1u; index < header->vocab_size; ++index)
        if (state->logits[index] > state->logits[best]) best = index;
    return best;
}
