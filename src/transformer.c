#include <stdint.h>

#include "litemodel_runtime.h"

static float rope_ratio(const struct litemodel_header *header) {
    uint32_t dimension = header->head_dim;
    int large_theta = header->rope_theta > 50000.0f;
    if (dimension == 32u) return large_theta ? 0.4869675252f : 0.5623413252f;
    if (dimension == 48u) return large_theta ? 0.6189658189f : 0.6812920691f;
    if (dimension == 64u) return large_theta ? 0.6978305849f : 0.7498942093f;
    if (dimension == 80u) return large_theta ? 0.7498942093f : 0.7943282347f;
    return large_theta ? 0.6978305849f : 0.7498942093f;
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

static void apply_rope(struct lm_runtime *runtime, float *values,
                       uint32_t heads, uint32_t position) {
    uint32_t head_dim = runtime->header->head_dim;
    float ratio = rope_ratio(runtime->header);
    for (uint32_t head = 0; head < heads; ++head) {
        float frequency = 1.0f;
        if (runtime->transformer.half_split_rope) {
            uint32_t half = head_dim / 2u;
            for (uint32_t index = 0; index < half; ++index) {
                float sine, cosine;
                fast_sincos((float)position * frequency, &sine, &cosine);
                uint32_t first = head * head_dim + index;
                uint32_t second = first + half;
                float a = values[first];
                float b = values[second];
                values[first] = a * cosine - b * sine;
                values[second] = b * cosine + a * sine;
                frequency *= ratio;
            }
        } else {
            for (uint32_t index = 0; index < head_dim; index += 2u) {
                float sine, cosine;
                fast_sincos((float)position * frequency, &sine, &cosine);
                uint32_t first = head * head_dim + index;
                float a = values[first];
                float b = values[first + 1u];
                values[first] = a * cosine - b * sine;
                values[first + 1u] = a * sine + b * cosine;
                frequency *= ratio;
            }
        }
    }
}

int lm_parse_transformer(struct lm_runtime *runtime, const uint8_t *weights,
                         const uint8_t *end, const struct lm_transformer_spec *spec) {
    const struct litemodel_header *header = runtime->header;
    uint32_t hidden = header->hidden_size;
    uint32_t kv = header->num_kv_heads * header->head_dim;
    const uint8_t *position = lm_parse_matrix(
        weights, end, header->vocab_size, hidden, header->quant_bits,
        &runtime->embedding
    );
    runtime->transformer = *spec;
    for (uint32_t index = 0; position && index < header->num_layers; ++index) {
        struct lm_transformer_layer *layer = &runtime->layers[index];
        position = lm_parse_vector(position, end, hidden, &layer->attention_norm);
        if (position) position = lm_parse_matrix(
            position, end, hidden, hidden, header->quant_bits, &layer->query
        );
        if (position) position = lm_parse_matrix(
            position, end, kv, hidden, header->quant_bits, &layer->key
        );
        if (position) position = lm_parse_matrix(
            position, end, kv, hidden, header->quant_bits, &layer->value
        );
        if (position) position = lm_parse_matrix(
            position, end, hidden, hidden, header->quant_bits, &layer->attention_output
        );
        if (spec->qk_norm) {
            if (position) position = lm_parse_vector(
                position, end, header->head_dim, &layer->q_norm
            );
            if (position) position = lm_parse_vector(
                position, end, header->head_dim, &layer->k_norm
            );
        }
        if (spec->refresh_layers & (1u << index)) {
            struct lm_refresh *refresh = &layer->refresh;
            if (position) position = lm_parse_vector(
                position, end, hidden, &refresh->attention_norm
            );
            if (position) position = lm_parse_vector(
                position, end, hidden, &refresh->embedding_norm
            );
            if (position) position = lm_parse_vector(
                position, end, hidden, &refresh->output_norm
            );
            if (position) position = lm_parse_matrix(
                position, end, hidden, hidden, header->quant_bits, &refresh->gate
            );
            if (position) position = lm_parse_matrix(
                position, end, hidden, hidden, header->quant_bits, &refresh->value
            );
            if (position) position = lm_parse_matrix(
                position, end, hidden, hidden, header->quant_bits, &refresh->output
            );
            if (position) position = lm_parse_matrix(
                position, end, hidden, spec->refresh_kernel,
                header->quant_bits, &refresh->kernel
            );
            if (position) position = lm_parse_vector(position, end, 1u, &refresh->alpha);
        }
        if (position) position = lm_parse_vector(position, end, hidden, &layer->ffn_norm);
        if (position) position = lm_parse_matrix(
            position, end, header->intermediate_size, hidden,
            header->quant_bits, &layer->gate
        );
        if (position) position = lm_parse_matrix(
            position, end, header->intermediate_size, hidden,
            header->quant_bits, &layer->up
        );
        if (position) position = lm_parse_matrix(
            position, end, hidden, header->intermediate_size,
            header->quant_bits, &layer->down
        );
    }
    if (position) position = lm_parse_vector(position, end, hidden, &runtime->final_norm);
    return position && position == end;
}

static uint32_t cache_index(const struct lm_runtime *runtime, uint32_t layer,
                            uint32_t position, uint32_t head, uint32_t dimension) {
    return ((((layer * runtime->context_capacity + position) * runtime->header->num_kv_heads + head) *
             runtime->header->head_dim) + dimension);
}

static uint32_t refresh_index(const struct lm_runtime *runtime, uint32_t layer,
                              uint32_t channel, uint32_t slot) {
    uint32_t history = runtime->transformer.refresh_kernel - 1u;
    return (layer * runtime->header->hidden_size + channel) * history + slot;
}

static void apply_xsa(struct lm_runtime *runtime) {
    const struct litemodel_header *header = runtime->header;
    uint32_t repeats = header->num_heads / header->num_kv_heads;
    for (uint32_t head = 0; head < header->num_heads; ++head) {
        uint32_t kv_head = head / repeats;
        float denominator = 0.0f;
        float projection = 0.0f;
        for (uint32_t dimension = 0; dimension < header->head_dim; ++dimension) {
            float current_value = runtime->value[kv_head * header->head_dim + dimension];
            denominator += current_value * current_value;
            projection += runtime->attention[head * header->head_dim + dimension] * current_value;
        }
        if (denominator < 0.000001f) denominator = 0.000001f;
        projection /= denominator;
        for (uint32_t dimension = 0; dimension < header->head_dim; ++dimension) {
            float current_value = runtime->value[kv_head * header->head_dim + dimension];
            runtime->attention[head * header->head_dim + dimension] -= projection * current_value;
        }
    }
}

static void apply_refresh(struct lm_runtime *runtime, uint32_t layer_index,
                          const float *attention_output) {
    const struct litemodel_header *header = runtime->header;
    struct lm_refresh *refresh = &runtime->layers[layer_index].refresh;
    uint32_t hidden = header->hidden_size;
    uint32_t kernel = runtime->transformer.refresh_kernel;
    lm_rms_norm(runtime, runtime->normalized, attention_output,
                refresh->attention_norm, hidden);
    lm_matvec(&refresh->gate, runtime->normalized, runtime->refresh_gate);
    for (uint32_t channel = 0; channel < hidden; ++channel) {
        lm_matrix_row(&refresh->kernel, channel, runtime->up);
        float convolved = runtime->up[kernel - 1u] * runtime->normalized[channel];
        for (uint32_t slot = 0; slot + 1u < kernel; ++slot)
            convolved += runtime->up[slot] *
                runtime->refresh_history[refresh_index(runtime, layer_index, channel, slot)];
        runtime->refresh_gate[channel] += convolved;
        for (uint32_t slot = 0; slot + 2u < kernel; ++slot)
            runtime->refresh_history[refresh_index(runtime, layer_index, channel, slot)] =
                runtime->refresh_history[refresh_index(runtime, layer_index, channel, slot + 1u)];
        runtime->refresh_history[refresh_index(runtime, layer_index, channel, kernel - 2u)] =
            runtime->normalized[channel];
    }
    lm_rms_norm(runtime, runtime->temporary, runtime->original_embedding,
                refresh->embedding_norm, hidden);
    lm_matvec(&refresh->value, runtime->temporary, runtime->refresh_value);
    for (uint32_t index = 0; index < hidden; ++index)
        runtime->refresh_gate[index] = runtime->refresh_gate[index] *
            lm_sigmoid(runtime->refresh_gate[index]) * runtime->refresh_value[index];
    lm_matvec(&refresh->output, runtime->refresh_gate, runtime->temporary);
    lm_rms_norm(runtime, runtime->normalized, runtime->temporary,
                refresh->output_norm, hidden);
    for (uint32_t index = 0; index < hidden; ++index)
        runtime->x[index] += refresh->alpha[0] * runtime->normalized[index];
}

uint16_t lm_transformer_forward(struct lm_runtime *runtime, uint16_t token,
                                uint32_t position) {
    const struct litemodel_header *header = runtime->header;
    if (position >= runtime->context_capacity || token >= header->vocab_size) return 0;
    uint32_t hidden = header->hidden_size;
    uint32_t head_dim = header->head_dim;
    uint32_t heads = header->num_heads;
    uint32_t kv_heads = header->num_kv_heads;
    uint32_t intermediate = header->intermediate_size;
    lm_matrix_row(&runtime->embedding, token, runtime->x);
    if (runtime->transformer.embedding_scale) {
        float scale = lm_sqrt((float)hidden);
        for (uint32_t index = 0; index < hidden; ++index) runtime->x[index] *= scale;
    }
    for (uint32_t index = 0; index < hidden; ++index)
        runtime->original_embedding[index] = runtime->x[index];

    for (uint32_t layer_index = 0; layer_index < header->num_layers; ++layer_index) {
        struct lm_transformer_layer *layer = &runtime->layers[layer_index];
        lm_rms_norm(runtime, runtime->normalized, runtime->x,
                    layer->attention_norm, hidden);
        lm_matvec(&layer->query, runtime->normalized, runtime->query);
        lm_matvec(&layer->key, runtime->normalized, runtime->key);
        lm_matvec(&layer->value, runtime->normalized, runtime->value);
        if (runtime->transformer.qk_norm) {
            for (uint32_t head = 0; head < heads; ++head)
                lm_rms_norm(runtime, &runtime->query[head * head_dim],
                            &runtime->query[head * head_dim], layer->q_norm, head_dim);
            for (uint32_t head = 0; head < kv_heads; ++head)
                lm_rms_norm(runtime, &runtime->key[head * head_dim],
                            &runtime->key[head * head_dim], layer->k_norm, head_dim);
        }
        apply_rope(runtime, runtime->query, heads, position);
        apply_rope(runtime, runtime->key, kv_heads, position);
        for (uint32_t head = 0; head < kv_heads; ++head)
            for (uint32_t dimension = 0; dimension < head_dim; ++dimension) {
                uint32_t cache = cache_index(runtime, layer_index, position, head, dimension);
                runtime->key_cache[cache] = runtime->key[head * head_dim + dimension];
                runtime->value_cache[cache] = runtime->value[head * head_dim + dimension];
            }
        float scale = 1.0f / lm_sqrt((float)head_dim);
        uint32_t repeats = heads / kv_heads;
        for (uint32_t head = 0; head < heads; ++head) {
            uint32_t kv_head = head / repeats;
            float maximum = -1000000.0f;
            for (uint32_t time = 0; time <= position; ++time) {
                float score = 0.0f;
                for (uint32_t dimension = 0; dimension < head_dim; ++dimension)
                    score += runtime->query[head * head_dim + dimension] *
                        runtime->key_cache[cache_index(runtime, layer_index, time,
                                                       kv_head, dimension)];
                runtime->scores[time] = score * scale;
                if (runtime->scores[time] > maximum) maximum = runtime->scores[time];
            }
            float sum = 0.0f;
            for (uint32_t time = 0; time <= position; ++time) {
                runtime->scores[time] = lm_exp(runtime->scores[time] - maximum);
                sum += runtime->scores[time];
            }
            for (uint32_t dimension = 0; dimension < head_dim; ++dimension) {
                float value = 0.0f;
                for (uint32_t time = 0; time <= position; ++time)
                    value += runtime->scores[time] *
                        runtime->value_cache[cache_index(runtime, layer_index, time,
                                                         kv_head, dimension)];
                runtime->attention[head * head_dim + dimension] = value / sum;
            }
        }
        if (runtime->transformer.xsa) apply_xsa(runtime);
        lm_matvec(&layer->attention_output, runtime->attention, runtime->temporary);
        for (uint32_t index = 0; index < hidden; ++index)
            runtime->x[index] += runtime->temporary[index];
        if (runtime->transformer.refresh_layers & (1u << layer_index))
            apply_refresh(runtime, layer_index, runtime->temporary);
        lm_rms_norm(runtime, runtime->normalized, runtime->x, layer->ffn_norm, hidden);
        lm_matvec(&layer->gate, runtime->normalized, runtime->gate);
        lm_matvec(&layer->up, runtime->normalized, runtime->up);
        for (uint32_t index = 0; index < intermediate; ++index)
            runtime->gate[index] = runtime->gate[index] *
                lm_sigmoid(runtime->gate[index]) * runtime->up[index];
        lm_matvec(&layer->down, runtime->gate, runtime->temporary);
        for (uint32_t index = 0; index < hidden; ++index)
            runtime->x[index] += runtime->temporary[index];
    }
    lm_rms_norm(runtime, runtime->normalized, runtime->x, runtime->final_norm, hidden);
    lm_matvec(&runtime->embedding, runtime->normalized, runtime->logits);
    uint16_t best = 0;
    for (uint32_t index = 1; index < header->vocab_size; ++index)
        if (runtime->logits[index] > runtime->logits[best]) best = (uint16_t)index;
    return best;
}

void lm_transformer_reset(struct lm_runtime *runtime) {
    if (!runtime->refresh_history || !runtime->transformer.refresh_layers) return;
    uint32_t count = runtime->header->num_layers * runtime->header->hidden_size *
        (runtime->transformer.refresh_kernel - 1u);
    for (uint32_t index = 0; index < count; ++index) runtime->refresh_history[index] = 0.0f;
}
