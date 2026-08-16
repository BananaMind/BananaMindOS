#ifndef BM2N_H
#define BM2N_H

#include <stdint.h>

#define BM2N_MAGIC "BM2NQ\r\n\032"
#define BM2N_VERSION 1u
#define BM2N_CONTEXT 64u

/* Little-endian, deliberately fixed-width and pointer-free on disk. */
struct __attribute__((packed)) bm2n_header {
    char magic[8];
    uint32_t version;
    uint32_t header_size;
    uint32_t file_size;
    uint32_t bits;
    uint32_t vocab_size;
    uint32_t hidden_size;
    uint32_t num_layers;
    uint32_t num_heads;
    uint32_t num_kv_heads;
    uint32_t head_dim;
    uint32_t intermediate_size;
    uint32_t max_seq_len;
    float rope_theta;
    float rms_eps;
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
};

struct __attribute__((packed)) bm2n_token {
    uint32_t offset;
    uint16_t length;
    uint16_t reserved;
};

struct __attribute__((packed)) bm2n_merge {
    uint16_t left;
    uint16_t right;
    uint16_t result;
    uint16_t rank;
};

#endif
