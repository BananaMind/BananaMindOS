#include <stdint.h>

#include "litemodel_runtime.h"

struct __attribute__((packed)) banana_config {
    uint32_t kind;
    uint32_t qk_norm;
    uint32_t embedding_scale;
    uint32_t refresh_layers;
    uint32_t refresh_kernel;
};

int lm_arch_banana_load(struct lm_runtime *runtime, const uint8_t *config_,
                        uint32_t config_size, const uint8_t *weights,
                        const uint8_t *weights_end) {
    if (config_size != sizeof(struct banana_config)) return 0;
    const struct banana_config *config = (const struct banana_config *)config_;
    if (config->kind < 1u || config->kind > 3u || config->qk_norm > 1u ||
        config->embedding_scale > 1u ||
        (config->refresh_layers &&
         (config->refresh_kernel < 2u || config->refresh_kernel > LM_REFRESH_KERNEL_MAX)))
        return 0;
    struct lm_transformer_spec spec = {
        (uint8_t)config->qk_norm,
        (uint8_t)config->embedding_scale,
        0u,
        0u,
        config->refresh_layers,
        config->refresh_kernel,
    };
    return lm_parse_transformer(runtime, weights, weights_end, &spec);
}
