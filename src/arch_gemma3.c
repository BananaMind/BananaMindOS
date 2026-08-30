#include <stdint.h>

#include "litemodel_runtime.h"

/* Gemma 3 has four normalization points per block, 1+weight RMSNorm,
   oversized query projections, and alternating local/global attention. */
struct __attribute__((packed)) gemma3_file_config {
    uint32_t full_attention_mask;
    uint32_t sliding_window;
    float full_rope_ratio;
    float sliding_rope_ratio;
    float attention_scale;
};

struct gemma3_layer {
    const float *input_norm;
    const float *q_norm;
    const float *k_norm;
    const float *post_attention_norm;
    const float *pre_ffn_norm;
    const float *post_ffn_norm;
    struct lm_matrix query;
    struct lm_matrix key;
    struct lm_matrix value;
    struct lm_matrix attention_output;
    struct lm_matrix gate;
    struct lm_matrix up;
    struct lm_matrix down;
};

struct gemma3_state {
    struct gemma3_file_config config;
    struct lm_matrix embedding;
    struct gemma3_layer layers[LM_MAX_LAYERS];
    const float *final_norm;
    float *key_cache;
    float *value_cache;
    float *x;
    float *normalized;
    float *temporary;
    float *query;
    float *key;
    float *value;
    float *attention;
    float *gate;
    float *up;
    float *scores;
    float *logits;
};

static struct gemma3_state gemma3;

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

int lm_arch_gemma3_load(struct lm_runtime *runtime, const uint8_t *config,
                        uint32_t config_size, const uint8_t *weights,
                        const uint8_t *weights_end) {
    if (config_size != sizeof(struct gemma3_file_config)) return 0;
    zero_bytes(&gemma3, sizeof(gemma3));
    gemma3.config = *(const struct gemma3_file_config *)config;
    const struct litemodel_header *header = runtime->header;
    uint32_t hidden = header->hidden_size;
    uint32_t query = header->num_heads * header->head_dim;
    uint32_t kv = header->num_kv_heads * header->head_dim;
    const uint8_t *position = lm_parse_matrix(
        weights, weights_end, header->vocab_size, hidden,
        header->quant_bits, &gemma3.embedding
    );
    for (uint32_t index = 0u; position && index < header->num_layers; ++index) {
        struct gemma3_layer *layer = &gemma3.layers[index];
        position = lm_parse_vector(position, weights_end, hidden, &layer->input_norm);
        if (position) position = lm_parse_matrix(position, weights_end, query, hidden,
                                                  header->quant_bits, &layer->query);
        if (position) position = lm_parse_matrix(position, weights_end, kv, hidden,
                                                  header->quant_bits, &layer->key);
        if (position) position = lm_parse_matrix(position, weights_end, kv, hidden,
                                                  header->quant_bits, &layer->value);
        if (position) position = lm_parse_matrix(position, weights_end, hidden, query,
                                                  header->quant_bits,
                                                  &layer->attention_output);
        if (position) position = lm_parse_vector(position, weights_end, header->head_dim,
                                                  &layer->q_norm);
        if (position) position = lm_parse_vector(position, weights_end, header->head_dim,
                                                  &layer->k_norm);
        if (position) position = lm_parse_vector(position, weights_end, hidden,
                                                  &layer->post_attention_norm);
        if (position) position = lm_parse_vector(position, weights_end, hidden,
                                                  &layer->pre_ffn_norm);
        if (position) position = lm_parse_matrix(position, weights_end,
                                                  header->intermediate_size, hidden,
                                                  header->quant_bits, &layer->gate);
        if (position) position = lm_parse_matrix(position, weights_end,
                                                  header->intermediate_size, hidden,
                                                  header->quant_bits, &layer->up);
        if (position) position = lm_parse_matrix(position, weights_end, hidden,
                                                  header->intermediate_size,
                                                  header->quant_bits, &layer->down);
        if (position) position = lm_parse_vector(position, weights_end, hidden,
                                                  &layer->post_ffn_norm);
    }
    if (position) position = lm_parse_vector(position, weights_end, hidden, &gemma3.final_norm);
    if (!position || position != weights_end) return 0;
    runtime->embedding = gemma3.embedding;
    runtime->final_norm = gemma3.final_norm;
    runtime->architecture_state = &gemma3;
    return 1;
}

int lm_arch_gemma3_allocate(struct lm_runtime *runtime, struct lm_arena *arena) {
    struct gemma3_state *state = runtime->architecture_state;
    const struct litemodel_header *header = runtime->header;
    uint32_t hidden = header->hidden_size;
    uint32_t query = header->num_heads * header->head_dim;
    uint32_t kv = header->num_kv_heads * header->head_dim;
    uint32_t cache = header->num_layers * runtime->context_capacity * kv;
    state->key_cache = lm_arena_alloc(arena, cache * 4u, 16u);
    state->value_cache = lm_arena_alloc(arena, cache * 4u, 16u);
    state->x = lm_arena_alloc(arena, hidden * 4u, 16u);
    state->normalized = lm_arena_alloc(arena, hidden * 4u, 16u);
    state->temporary = lm_arena_alloc(arena, hidden * 4u, 16u);
    state->query = lm_arena_alloc(arena, query * 4u, 16u);
    state->key = lm_arena_alloc(arena, kv * 4u, 16u);
    state->value = lm_arena_alloc(arena, kv * 4u, 16u);
    state->attention = lm_arena_alloc(arena, query * 4u, 16u);
    state->gate = lm_arena_alloc(arena, header->intermediate_size * 4u, 16u);
    state->up = lm_arena_alloc(arena, header->intermediate_size * 4u, 16u);
    state->scores = lm_arena_alloc(arena, runtime->context_capacity * 4u, 16u);
    state->logits = lm_arena_alloc(arena, header->vocab_size * 4u, 16u);
    runtime->logits = state->logits;
    return state->key_cache && state->value_cache && state->x && state->normalized &&
        state->temporary && state->query && state->key && state->value &&
        state->attention && state->gate && state->up && state->scores && state->logits;
}

static float gelu_tanh(float value) {
    float inside = 0.7978845608f * (value + 0.044715f * value * value * value);
    return 0.5f * value * (1.0f + lm_tanh(inside));
}

uint32_t lm_arch_gemma3_forward(struct lm_runtime *runtime, uint32_t token,
                                uint32_t position) {
    struct gemma3_state *state = runtime->architecture_state;
    const struct litemodel_header *header = runtime->header;
    if (!state || token >= header->vocab_size || position >= runtime->context_capacity) return 0u;
    uint32_t hidden = header->hidden_size;
    uint32_t head_dim = header->head_dim;
    uint32_t heads = header->num_heads;
    uint32_t kv_heads = header->num_kv_heads;
    lm_matrix_row(&state->embedding, token, state->x);
    float embedding_scale = lm_sqrt((float)hidden);
    for (uint32_t index = 0u; index < hidden; ++index) state->x[index] *= embedding_scale;
    for (uint32_t layer_index = 0u; layer_index < header->num_layers; ++layer_index) {
        struct gemma3_layer *layer = &state->layers[layer_index];
        lm_rms_norm_offset(runtime, state->normalized, state->x,
                           layer->input_norm, hidden, 1.0f);
        lm_matvec(&layer->query, state->normalized, state->query);
        lm_matvec(&layer->key, state->normalized, state->key);
        lm_matvec(&layer->value, state->normalized, state->value);
        for (uint32_t head = 0u; head < heads; ++head)
            lm_rms_norm_offset(runtime, state->query + head * head_dim,
                               state->query + head * head_dim,
                               layer->q_norm, head_dim, 1.0f);
        for (uint32_t head = 0u; head < kv_heads; ++head)
            lm_rms_norm_offset(runtime, state->key + head * head_dim,
                               state->key + head * head_dim,
                               layer->k_norm, head_dim, 1.0f);
        uint8_t full = (state->config.full_attention_mask & (1u << layer_index)) != 0u;
        float ratio = full ? state->config.full_rope_ratio : state->config.sliding_rope_ratio;
        apply_rope(state->query, heads, head_dim, position, ratio);
        apply_rope(state->key, kv_heads, head_dim, position, ratio);
        for (uint32_t head = 0u; head < kv_heads; ++head)
            for (uint32_t dimension = 0u; dimension < head_dim; ++dimension) {
                uint32_t cache = cache_index(runtime, layer_index, position, head, dimension);
                state->key_cache[cache] = state->key[head * head_dim + dimension];
                state->value_cache[cache] = state->value[head * head_dim + dimension];
            }
        uint32_t repeats = heads / kv_heads;
        uint32_t start = 0u;
        if (!full && position + 1u > state->config.sliding_window)
            start = position + 1u - state->config.sliding_window;
        for (uint32_t head = 0u; head < heads; ++head) {
            uint32_t kv_head = head / repeats;
            float maximum = -1000000.0f;
            for (uint32_t time = start; time <= position; ++time) {
                float score = 0.0f;
                for (uint32_t dimension = 0u; dimension < head_dim; ++dimension)
                    score += state->query[head * head_dim + dimension] *
                        state->key_cache[cache_index(runtime, layer_index, time,
                                                      kv_head, dimension)];
                state->scores[time] = score * state->config.attention_scale;
                if (state->scores[time] > maximum) maximum = state->scores[time];
            }
            float sum = 0.0f;
            for (uint32_t time = start; time <= position; ++time) {
                state->scores[time] = lm_exp(state->scores[time] - maximum);
                sum += state->scores[time];
            }
            for (uint32_t dimension = 0u; dimension < head_dim; ++dimension) {
                float value = 0.0f;
                for (uint32_t time = start; time <= position; ++time)
                    value += state->scores[time] *
                        state->value_cache[cache_index(runtime, layer_index, time,
                                                        kv_head, dimension)];
                state->attention[head * head_dim + dimension] = value / sum;
            }
        }
        lm_matvec(&layer->attention_output, state->attention, state->temporary);
        lm_rms_norm_offset(runtime, state->temporary, state->temporary,
                           layer->post_attention_norm, hidden, 1.0f);
        for (uint32_t index = 0u; index < hidden; ++index) state->x[index] += state->temporary[index];
        lm_rms_norm_offset(runtime, state->normalized, state->x,
                           layer->pre_ffn_norm, hidden, 1.0f);
        lm_matvec(&layer->gate, state->normalized, state->gate);
        lm_matvec(&layer->up, state->normalized, state->up);
        for (uint32_t index = 0u; index < header->intermediate_size; ++index)
            state->gate[index] = gelu_tanh(state->gate[index]) * state->up[index];
        lm_matvec(&layer->down, state->gate, state->temporary);
        lm_rms_norm_offset(runtime, state->temporary, state->temporary,
                           layer->post_ffn_norm, hidden, 1.0f);
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
