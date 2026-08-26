#include <stdint.h>

#include "litemodel_runtime.h"

struct __attribute__((packed)) rose_config {
    uint32_t half_split_rope;
    uint32_t qk_norm;
    uint32_t refresh_layers;
    uint32_t refresh_kernel;
};

int lm_arch_rose_load(struct lm_runtime *runtime, const uint8_t *config_,
                      uint32_t config_size, const uint8_t *weights,
                      const uint8_t *weights_end) {
    if (config_size != sizeof(struct rose_config)) return 0;
    const struct rose_config *config = (const struct rose_config *)config_;
    if (config->half_split_rope != 1u || config->qk_norm != 1u ||
        !config->refresh_layers || config->refresh_kernel < 2u ||
        config->refresh_kernel > LM_REFRESH_KERNEL_MAX)
        return 0;
    struct lm_transformer_spec spec = {
        1u,
        0u,
        0u,
        1u,
        config->refresh_layers,
        config->refresh_kernel,
    };
    return lm_parse_transformer(runtime, weights, weights_end, &spec);
}
