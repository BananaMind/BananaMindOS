#ifndef MOUSE_H
#define MOUSE_H

#include <stdint.h>

struct mouse_state {
    int32_t x;
    int32_t y;
    uint8_t buttons;
    uint8_t clicked;
    uint8_t changed;
};

int mouse_init(struct mouse_state *state, uint32_t width, uint32_t height);
void mouse_poll(struct mouse_state *state, uint32_t width, uint32_t height);
int mouse_take_keyboard_byte(uint8_t *value);

#endif
