#include <stdint.h>

#include "model_store.h"

#define ATAPI_SECTOR_SIZE 2048u
#define CATALOG_LIMIT 16384u

struct iso_file {
    uint32_t extent;
    uint32_t size;
    uint8_t directory;
};

static uint8_t sector_buffer[ATAPI_SECTOR_SIZE];
static char catalog_buffer[CATALOG_LIMIT + 1u];

static inline void outb(uint16_t port, uint8_t value) {
    __asm__ volatile ("outb %0, %1" : : "a"(value), "Nd"(port));
}

static inline uint8_t inb(uint16_t port) {
    uint8_t value;
    __asm__ volatile ("inb %1, %0" : "=a"(value) : "Nd"(port));
    return value;
}

static inline void outw(uint16_t port, uint16_t value) {
    __asm__ volatile ("outw %0, %1" : : "a"(value), "Nd"(port));
}

static inline uint16_t inw(uint16_t port) {
    uint16_t value;
    __asm__ volatile ("inw %1, %0" : "=a"(value) : "Nd"(port));
    return value;
}

static uint32_t little32(const uint8_t *data) {
    return (uint32_t)data[0] | ((uint32_t)data[1] << 8) |
        ((uint32_t)data[2] << 16) | ((uint32_t)data[3] << 24);
}

static void copy_bytes(void *destination_, const void *source_, uint32_t count) {
    uint8_t *destination = destination_;
    const uint8_t *source = source_;
    while (count--) *destination++ = *source++;
}

static void zero_bytes(void *destination_, uint32_t count) {
    uint8_t *destination = destination_;
    while (count--) *destination++ = 0;
}

static int wait_not_busy(uint16_t io) {
    for (uint32_t timeout = 0; timeout < 10000000u; ++timeout) {
        uint8_t status = inb(io + 7u);
        if (!(status & 0x80u)) return !(status & 0x01u);
    }
    return 0;
}

static int wait_data(uint16_t io) {
    for (uint32_t timeout = 0; timeout < 10000000u; ++timeout) {
        uint8_t status = inb(io + 7u);
        if (status & 0x01u) return 0;
        if (!(status & 0x80u) && (status & 0x08u)) return 1;
    }
    return 0;
}

static void select_drive(uint16_t io, uint8_t drive) {
    outb(io + 6u, (uint8_t)(0xA0u | (drive ? 0x10u : 0u)));
    uint16_t control = (uint16_t)(io + 0x206u);
    (void)inb(control); (void)inb(control); (void)inb(control); (void)inb(control);
}

static int atapi_read(uint16_t io, uint8_t drive, uint32_t lba, uint8_t *destination) {
    select_drive(io, drive);
    if (!wait_not_busy(io)) return 0;
    outb(io + 1u, 0u);
    outb(io + 4u, (uint8_t)ATAPI_SECTOR_SIZE);
    outb(io + 5u, (uint8_t)(ATAPI_SECTOR_SIZE >> 8));
    outb(io + 7u, 0xA0u);
    if (!wait_data(io)) return 0;

    uint8_t packet[12] = {0};
    packet[0] = 0xA8u;
    packet[2] = (uint8_t)(lba >> 24);
    packet[3] = (uint8_t)(lba >> 16);
    packet[4] = (uint8_t)(lba >> 8);
    packet[5] = (uint8_t)lba;
    packet[9] = 1u;
    for (uint32_t index = 0; index < 6u; ++index)
        outw(io, (uint16_t)packet[index * 2u] | ((uint16_t)packet[index * 2u + 1u] << 8));
    if (!wait_data(io)) return 0;
    uint32_t byte_count = (uint32_t)inb(io + 4u) | ((uint32_t)inb(io + 5u) << 8);
    if (!byte_count || byte_count > ATAPI_SECTOR_SIZE || (byte_count & 1u)) return 0;
    for (uint32_t index = 0; index < byte_count / 2u; ++index) {
        uint16_t word = inw(io);
        destination[index * 2u] = (uint8_t)word;
        destination[index * 2u + 1u] = (uint8_t)(word >> 8);
    }
    for (uint32_t index = byte_count; index < ATAPI_SECTOR_SIZE; ++index)
        destination[index] = 0;
    return wait_not_busy(io);
}

static int character_equal(char left, char right) {
    if (left >= 'a' && left <= 'z') left = (char)(left - 'a' + 'A');
    if (right >= 'a' && right <= 'z') right = (char)(right - 'a' + 'A');
    return left == right;
}

static int iso_name_equal(const uint8_t *iso_name, uint32_t iso_length,
                          const char *wanted, uint32_t wanted_length) {
    uint32_t clean_length = iso_length;
    for (uint32_t index = 0; index < iso_length; ++index)
        if (iso_name[index] == ';') { clean_length = index; break; }
    while (clean_length && iso_name[clean_length - 1u] == '.') --clean_length;
    if (clean_length != wanted_length) return 0;
    for (uint32_t index = 0; index < clean_length; ++index)
        if (!character_equal((char)iso_name[index], wanted[index])) return 0;
    return 1;
}

static int directory_find(const struct model_store *store, const struct iso_file *directory,
                          const char *name, uint32_t name_length, struct iso_file *result) {
    uint32_t offset = 0;
    uint32_t loaded_sector = 0xFFFFFFFFu;
    while (offset < directory->size) {
        uint32_t sector_index = offset / ATAPI_SECTOR_SIZE;
        uint32_t within = offset % ATAPI_SECTOR_SIZE;
        if (sector_index != loaded_sector) {
            if (!atapi_read(store->io_base, store->drive,
                            directory->extent + sector_index, sector_buffer)) return 0;
            loaded_sector = sector_index;
        }
        uint8_t record_length = sector_buffer[within];
        if (!record_length) {
            offset = (sector_index + 1u) * ATAPI_SECTOR_SIZE;
            continue;
        }
        if (within + record_length > ATAPI_SECTOR_SIZE || record_length < 34u) return 0;
        const uint8_t *record = &sector_buffer[within];
        uint8_t iso_length = record[32];
        if (33u + iso_length <= record_length &&
            iso_name_equal(record + 33u, iso_length, name, name_length)) {
            result->extent = little32(record + 2u);
            result->size = little32(record + 10u);
            result->directory = (uint8_t)((record[25] & 2u) != 0u);
            return 1;
        }
        offset += record_length;
    }
    return 0;
}

static int iso_find(const struct model_store *store, const char *path, struct iso_file *result) {
    if (!atapi_read(store->io_base, store->drive, 16u, sector_buffer)) return 0;
    if (sector_buffer[0] != 1u || sector_buffer[1] != 'C' || sector_buffer[2] != 'D' ||
        sector_buffer[3] != '0' || sector_buffer[4] != '0' || sector_buffer[5] != '1')
        return 0;
    const uint8_t *root_record = sector_buffer + 156u;
    struct iso_file current = {
        little32(root_record + 2u), little32(root_record + 10u), 1u
    };
    while (*path == '/') ++path;
    while (*path) {
        const char *start = path;
        while (*path && *path != '/') ++path;
        uint32_t length = (uint32_t)(path - start);
        if (!length || !current.directory ||
            !directory_find(store, &current, start, length, &current)) return 0;
        while (*path == '/') ++path;
    }
    *result = current;
    return 1;
}

static uint32_t string_length(const char *text) {
    uint32_t length = 0;
    while (text[length]) ++length;
    return length;
}

static void copy_field(char *destination, uint32_t capacity,
                       const char *begin, const char *end) {
    uint32_t written = 0;
    while (begin < end && written + 1u < capacity) destination[written++] = *begin++;
    destination[written] = 0;
}

static uint32_t parse_number(const char *begin, const char *end) {
    uint32_t value = 0;
    while (begin < end && *begin >= '0' && *begin <= '9')
        value = value * 10u + (uint32_t)(*begin++ - '0');
    return value;
}

static int field_true(const char *begin, const char *end, const char *word) {
    uint32_t length = (uint32_t)(end - begin);
    if (string_length(word) != length) return 0;
    for (uint32_t index = 0; index < length; ++index)
        if (!character_equal(begin[index], word[index])) return 0;
    return 1;
}

static int parse_catalog(struct model_store *store, uint32_t size) {
    uint32_t position = 0;
    while (position < size && store->count < MODEL_STORE_MAX_MODELS) {
        uint32_t line_start = position;
        while (position < size && catalog_buffer[position] != '\n' &&
               catalog_buffer[position] != '\r') ++position;
        uint32_t line_end = position;
        while (position < size &&
               (catalog_buffer[position] == '\n' || catalog_buffer[position] == '\r')) ++position;
        if (line_end == line_start || catalog_buffer[line_start] == '#') continue;
        const char *fields[8];
        const char *ends[8];
        uint32_t field_count = 0;
        const char *cursor = catalog_buffer + line_start;
        const char *end = catalog_buffer + line_end;
        while (cursor <= end && field_count < 8u) {
            fields[field_count] = cursor;
            while (cursor < end && *cursor != '|') ++cursor;
            ends[field_count++] = cursor;
            if (cursor < end) ++cursor;
            else break;
        }
        if (field_count != 8u) continue;
        struct model_store_entry *entry = &store->entries[store->count];
        zero_bytes(entry, sizeof(*entry));
        copy_field(entry->id, sizeof(entry->id), fields[0], ends[0]);
        copy_field(entry->filename, sizeof(entry->filename), fields[1], ends[1]);
        copy_field(entry->name, sizeof(entry->name), fields[2], ends[2]);
        copy_field(entry->variant, sizeof(entry->variant), fields[3], ends[3]);
        entry->recommended_ram_mb = parse_number(fields[4], ends[4]);
        entry->chat = (uint8_t)field_true(fields[5], ends[5], "chat");
        entry->legacy = (uint8_t)field_true(fields[6], ends[6], "legacy");
        copy_field(entry->description, sizeof(entry->description), fields[7], ends[7]);
        char path[64] = "/BOOT/MODELS/";
        uint32_t prefix = string_length(path);
        uint32_t filename = string_length(entry->filename);
        if (prefix + filename + 1u > sizeof(path)) continue;
        copy_bytes(path + prefix, entry->filename, filename + 1u);
        struct iso_file file;
        if (!iso_find(store, path, &file) || file.directory) continue;
        entry->file_extent = file.extent;
        entry->file_size = file.size;
        ++store->count;
    }
    return store->count != 0u;
}

int model_store_open(struct model_store *store) {
    zero_bytes(store, sizeof(*store));
    const uint16_t ports[] = {0x170u, 0x1F0u};
    for (uint32_t port = 0; port < 2u; ++port) {
        for (uint32_t drive = 0; drive < 2u; ++drive) {
            store->io_base = ports[port];
            store->drive = (uint8_t)drive;
            struct iso_file catalog;
            if (!iso_find(store, "/BOOT/MODELS/CATALOG.CFG", &catalog) ||
                catalog.directory || !catalog.size || catalog.size > CATALOG_LIMIT) continue;
            uint32_t copied = 0;
            while (copied < catalog.size) {
                if (!atapi_read(store->io_base, store->drive,
                                catalog.extent + copied / ATAPI_SECTOR_SIZE,
                                sector_buffer)) break;
                uint32_t chunk = catalog.size - copied;
                if (chunk > ATAPI_SECTOR_SIZE) chunk = ATAPI_SECTOR_SIZE;
                copy_bytes(catalog_buffer + copied, sector_buffer, chunk);
                copied += chunk;
            }
            if (copied != catalog.size) continue;
            catalog_buffer[catalog.size] = 0;
            if (parse_catalog(store, catalog.size)) return 1;
        }
    }
    return 0;
}

int model_store_load(const struct model_store *store,
                     const struct model_store_entry *entry,
                     void *destination_, uint32_t capacity,
                     model_store_progress progress) {
    if (!entry->file_size || entry->file_size > capacity) return 0;
    uint8_t *destination = destination_;
    uint32_t copied = 0;
    while (copied < entry->file_size) {
        uint32_t chunk = entry->file_size - copied;
        uint8_t *target = destination + copied;
        if (chunk >= ATAPI_SECTOR_SIZE) {
            if (!atapi_read(store->io_base, store->drive,
                            entry->file_extent + copied / ATAPI_SECTOR_SIZE, target)) return 0;
            chunk = ATAPI_SECTOR_SIZE;
        } else {
            if (!atapi_read(store->io_base, store->drive,
                            entry->file_extent + copied / ATAPI_SECTOR_SIZE,
                            sector_buffer)) return 0;
            copy_bytes(target, sector_buffer, chunk);
        }
        copied += chunk;
        if (progress) progress(copied, entry->file_size);
    }
    return 1;
}
