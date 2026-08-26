#ifndef LITEMODEL_RUNTIME_H
#define LITEMODEL_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#include "litemodel.h"
#include "cpu.h"

#define LM_CONTEXT 64u
#define LM_MAX_CONTEXT 256u
#define LM_MAX_LAYERS 32u
#define LM_REFRESH_KERNEL_MAX 9u

struct lm_arena {
    uint8_t *next;
    uint8_t *end;
};

struct lm_matrix {
    const uint8_t *data;
    uint32_t rows;
    uint32_t cols;
    uint32_t stride;
    uint32_t bits;
};

struct lm_refresh {
    const float *attention_norm;
    const float *embedding_norm;
    const float *output_norm;
    const float *alpha;
    struct lm_matrix gate;
    struct lm_matrix value;
    struct lm_matrix output;
    struct lm_matrix kernel;
};

struct lm_transformer_layer {
    const float *attention_norm;
    const float *q_norm;
    const float *k_norm;
    const float *ffn_norm;
    struct lm_matrix query;
    struct lm_matrix key;
    struct lm_matrix value;
    struct lm_matrix attention_output;
    struct lm_matrix gate;
    struct lm_matrix up;
    struct lm_matrix down;
    struct lm_refresh refresh;
};

struct lm_transformer_spec {
    uint8_t qk_norm;
    uint8_t embedding_scale;
    uint8_t xsa;
    uint8_t half_split_rope;
    uint32_t refresh_layers;
    uint32_t refresh_kernel;
};

struct lm_merge_slot {
    uint32_t key;
    uint16_t result;
    uint16_t rank_plus_one;
};

struct lm_runtime {
    const struct litemodel_header *header;
    const uint8_t *file_base;
    uint32_t file_size;
    uint32_t context_capacity;
    struct lm_matrix embedding;
    struct lm_transformer_layer layers[LM_MAX_LAYERS];
    struct lm_transformer_spec transformer;
    const float *final_norm;

    const struct litemodel_token *tokens;
    const uint8_t *token_data;
    const struct litemodel_merge *merges;
    struct lm_merge_slot *merge_table;
    uint32_t merge_slots;
    uint16_t byte_token[256];

    float *key_cache;
    float *value_cache;
    float *refresh_history;
    float *x;
    float *normalized;
    float *temporary;
    float *original_embedding;
    float *query;
    float *key;
    float *value;
    float *attention;
    float *gate;
    float *up;
    float *refresh_gate;
    float *refresh_value;
    float *scores;
    float *logits;
    uint16_t *bpe_tokens;
    uint16_t *prompt_tokens;
    void *architecture_state;
};

void *lm_arena_alloc(struct lm_arena *arena, uint32_t size, uint32_t alignment);
int lm_range_ok(uint32_t offset, uint32_t size, uint32_t file_size);
const uint8_t *lm_parse_vector(const uint8_t *p, const uint8_t *end,
                               uint32_t count, const float **result);
const uint8_t *lm_parse_matrix(const uint8_t *p, const uint8_t *end,
                               uint32_t rows, uint32_t cols, uint32_t bits,
                               struct lm_matrix *result);
void lm_matrix_row(const struct lm_matrix *matrix, uint32_t row, float *output);
void lm_matvec(const struct lm_matrix *matrix, const float *input, float *output);
void lm_matvec_configure(enum cpu_math_backend backend);
void lm_rms_norm(struct lm_runtime *runtime, float *output, const float *input,
                 const float *weight, uint32_t size);
float lm_sqrt(float value);
float lm_exp(float value);
float lm_sigmoid(float value);

int lm_parse_transformer(struct lm_runtime *runtime, const uint8_t *weights,
                         const uint8_t *end, const struct lm_transformer_spec *spec);
uint16_t lm_transformer_forward(struct lm_runtime *runtime, uint16_t token,
                                uint32_t position);
void lm_transformer_reset(struct lm_runtime *runtime);

int lm_arch_banana_load(struct lm_runtime *runtime, const uint8_t *config,
                        uint32_t config_size, const uint8_t *weights,
                        const uint8_t *weights_end);
int lm_arch_llama_load(struct lm_runtime *runtime, const uint8_t *config,
                       uint32_t config_size, const uint8_t *weights,
                       const uint8_t *weights_end);
int lm_arch_gptx_load(struct lm_runtime *runtime, const uint8_t *config,
                      uint32_t config_size, const uint8_t *weights,
                      const uint8_t *weights_end);
int lm_arch_rose_load(struct lm_runtime *runtime, const uint8_t *config,
                      uint32_t config_size, const uint8_t *weights,
                      const uint8_t *weights_end);
int lm_arch_minspark_load(struct lm_runtime *runtime, const uint8_t *config,
                          uint32_t config_size, const uint8_t *weights,
                          const uint8_t *weights_end);
int lm_arch_minspark_allocate(struct lm_runtime *runtime, struct lm_arena *arena);
uint16_t lm_arch_minspark_forward(struct lm_runtime *runtime,
                                  const uint16_t *tokens, uint32_t count);
int lm_arch_minspark_set_effort(struct lm_runtime *runtime, uint32_t loops);
uint32_t lm_arch_minspark_get_effort(const struct lm_runtime *runtime);

int litemodel_load(struct lm_runtime *runtime, const void *base, uint32_t size,
                   struct lm_arena *arena);
int litemodel_load_with_context(struct lm_runtime *runtime, const void *base,
                                uint32_t size, struct lm_arena *arena,
                                uint32_t context_tokens);
void litemodel_reset(struct lm_runtime *runtime);
uint32_t litemodel_tokenize(struct lm_runtime *runtime, const uint8_t *input,
                            uint32_t length, uint16_t *output, uint32_t capacity);
uint16_t litemodel_forward(struct lm_runtime *runtime, uint16_t token,
                           uint32_t position);
uint16_t litemodel_forward_sequence(struct lm_runtime *runtime,
                                    const uint16_t *tokens, uint32_t count);
const struct litemodel_token *litemodel_token(const struct lm_runtime *runtime,
                                               uint16_t id);
uint16_t litemodel_sample(const struct lm_runtime *runtime,
                          uint32_t temperature_tenths, uint32_t random_value);

#endif
