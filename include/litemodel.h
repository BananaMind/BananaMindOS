#ifndef LITEMODEL_H
#define LITEMODEL_H

#include <stdint.h>

#define LITEMODEL_MAGIC "LITEMDL\032"
#define LITEMODEL_VERSION 1u

enum litemodel_architecture {
    LITEMODEL_ARCH_BANANA = 1,
    LITEMODEL_ARCH_LLAMA = 2,
    LITEMODEL_ARCH_GPTX2 = 3,
    LITEMODEL_ARCH_ROSE_X1 = 4,
    LITEMODEL_ARCH_MINSPARK = 5,
};

enum litemodel_flags {
    LITEMODEL_FLAG_CHAT = 1u << 0,
    LITEMODEL_FLAG_TIED_EMBEDDING = 1u << 1,
};

/*
 * Architecture-neutral, little-endian container header.  Architecture data
 * is deliberately kept in its own blob so model runtimes do not leak into
 * this shared format definition.
 */
struct __attribute__((packed)) litemodel_header {
    char magic[8];
    uint32_t version;
    uint32_t header_size;
    uint32_t file_size;
    uint32_t architecture;
    uint32_t quant_bits;
    uint32_t flags;
    uint32_t vocab_size;
    uint32_t hidden_size;
    uint32_t num_layers;
    uint32_t num_heads;
    uint32_t num_kv_heads;
    uint32_t head_dim;
    uint32_t intermediate_size;
    uint32_t max_seq_len;
    float rope_theta;
    float norm_eps;
    uint32_t weights_offset;
    uint32_t weights_size;
    uint32_t token_index_offset;
    uint32_t token_data_offset;
    uint32_t token_data_size;
    uint32_t merges_offset;
    uint32_t merges_count;
    uint32_t bos_id;
    uint32_t eos_id;
    uint32_t pad_id;
    uint32_t unk_id;
    uint32_t payload_crc32;
    uint32_t architecture_data_offset;
    uint32_t architecture_data_size;
};

struct __attribute__((packed)) litemodel_token {
    uint32_t offset;
    uint16_t length;
    uint16_t reserved;
};

struct __attribute__((packed)) litemodel_merge {
    uint16_t left;
    uint16_t right;
    uint16_t result;
    uint16_t rank;
};

#endif
