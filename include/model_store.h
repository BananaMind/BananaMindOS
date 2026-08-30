#ifndef MODEL_STORE_H
#define MODEL_STORE_H

#include <stdint.h>

#define MODEL_STORE_MAX_MODELS 48u

struct model_store_entry {
    char id[20];
    char filename[32];
    char name[48];
    char variant[20];
    char description[72];
    uint32_t recommended_ram_mb;
    uint32_t file_extent;
    uint32_t file_size;
    uint8_t chat;
    uint8_t legacy;
};

struct model_store {
    uint16_t io_base;
    uint8_t drive;
    uint32_t count;
    struct model_store_entry entries[MODEL_STORE_MAX_MODELS];
};

struct model_store_file {
    uint32_t extent;
    uint32_t size;
};

typedef void (*model_store_progress)(uint32_t completed, uint32_t total);

int model_store_open(struct model_store *store);
int model_store_load(const struct model_store *store,
                     const struct model_store_entry *entry,
                     void *destination, uint32_t capacity,
                     model_store_progress progress);
int model_store_find_file(const struct model_store *store, const char *path,
                          struct model_store_file *file);
int model_store_read_file(const struct model_store *store,
                          const struct model_store_file *file,
                          uint32_t offset, void *destination, uint32_t count);

#endif
