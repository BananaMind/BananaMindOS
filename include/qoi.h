#ifndef QOI_H
#define QOI_H

#include <stdint.h>

struct qoi_info {
    uint32_t width;
    uint32_t height;
    uint8_t channels;
    uint8_t colorspace;
};

typedef int (*qoi_read_fn)(void *context, uint8_t *destination, uint32_t count);
typedef void (*qoi_pixel_fn)(void *context, uint32_t x, uint32_t y,
                             uint8_t red, uint8_t green, uint8_t blue,
                             uint8_t alpha);

/* Decodes a standard Quite OK Image stream without libc or allocation. */
int qoi_decode(qoi_read_fn read, void *read_context,
               qoi_pixel_fn pixel, void *pixel_context,
               struct qoi_info *info);

#endif
