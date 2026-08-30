#include <stdint.h>

#include "qoi.h"

struct qoi_rgba {
    uint8_t r, g, b, a;
};

static uint32_t big32(const uint8_t *data) {
    return ((uint32_t)data[0] << 24) | ((uint32_t)data[1] << 16) |
        ((uint32_t)data[2] << 8) | (uint32_t)data[3];
}

static uint32_t qoi_hash(struct qoi_rgba value) {
    return ((uint32_t)value.r * 3u + (uint32_t)value.g * 5u +
            (uint32_t)value.b * 7u + (uint32_t)value.a * 11u) & 63u;
}

int qoi_decode(qoi_read_fn read, void *read_context,
               qoi_pixel_fn pixel, void *pixel_context,
               struct qoi_info *info) {
    uint8_t header[14];
    struct qoi_rgba index[64] = {{0u, 0u, 0u, 0u}};
    struct qoi_rgba current = {0u, 0u, 0u, 255u};
    uint32_t run = 0u;
    if (!read || !pixel || !info || !read(read_context, header, sizeof(header))) return 0;
    if (header[0] != 'q' || header[1] != 'o' || header[2] != 'i' || header[3] != 'f')
        return 0;
    info->width = big32(header + 4u);
    info->height = big32(header + 8u);
    info->channels = header[12];
    info->colorspace = header[13];
    if (!info->width || !info->height || info->width > 8192u || info->height > 8192u ||
        (info->channels != 3u && info->channels != 4u) || info->width > 0xFFFFFFFFu / info->height)
        return 0;

    uint32_t total = info->width * info->height;
    for (uint32_t position = 0u; position < total; ++position) {
        if (run) {
            --run;
        } else {
            uint8_t tag;
            if (!read(read_context, &tag, 1u)) return 0;
            if (tag == 0xFEu) {
                uint8_t rgb[3];
                if (!read(read_context, rgb, sizeof(rgb))) return 0;
                current.r = rgb[0]; current.g = rgb[1]; current.b = rgb[2];
            } else if (tag == 0xFFu) {
                uint8_t rgba[4];
                if (!read(read_context, rgba, sizeof(rgba))) return 0;
                current.r = rgba[0]; current.g = rgba[1];
                current.b = rgba[2]; current.a = rgba[3];
            } else if ((tag & 0xC0u) == 0x00u) {
                current = index[tag & 63u];
            } else if ((tag & 0xC0u) == 0x40u) {
                current.r = (uint8_t)(current.r + ((tag >> 4) & 3u) - 2u);
                current.g = (uint8_t)(current.g + ((tag >> 2) & 3u) - 2u);
                current.b = (uint8_t)(current.b + (tag & 3u) - 2u);
            } else if ((tag & 0xC0u) == 0x80u) {
                uint8_t second;
                if (!read(read_context, &second, 1u)) return 0;
                int32_t dg = (int32_t)(tag & 0x3Fu) - 32;
                current.r = (uint8_t)((int32_t)current.r + dg + (int32_t)(second >> 4) - 8);
                current.g = (uint8_t)((int32_t)current.g + dg);
                current.b = (uint8_t)((int32_t)current.b + dg + (int32_t)(second & 15u) - 8);
            } else {
                run = tag & 0x3Fu;
            }
            index[qoi_hash(current)] = current;
        }
        pixel(pixel_context, position % info->width, position / info->width,
              current.r, current.g, current.b, current.a);
    }
    return 1;
}
