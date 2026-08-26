#include <stdint.h>

#include "litemodel_runtime.h"

int lm_arch_llama_load(struct lm_runtime *runtime, const uint8_t *config,
                       uint32_t config_size, const uint8_t *weights,
                       const uint8_t *weights_end) {
    if (config_size != 4u || *(const uint32_t *)config != 1u) return 0;
    const struct lm_transformer_spec spec = {0u, 0u, 0u, 1u, 0u, 0u};
    return lm_parse_transformer(runtime, weights, weights_end, &spec);
}
