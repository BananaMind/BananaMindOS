#include <stdint.h>

#include "mouse.h"

static uint8_t packet[3];
static uint8_t cycle;

static inline void outb(uint16_t port, uint8_t value) {
    __asm__ volatile ("outb %0, %1" : : "a"(value), "Nd"(port));
}

static inline uint8_t inb(uint16_t port) {
    uint8_t value;
    __asm__ volatile ("inb %1, %0" : "=a"(value) : "Nd"(port));
    return value;
}

static int wait_input_empty(void) {
    for (uint32_t timeout = 0; timeout < 1000000u; ++timeout)
        if (!(inb(0x64u) & 2u)) return 1;
    return 0;
}

static int wait_output_full(void) {
    for (uint32_t timeout = 0; timeout < 1000000u; ++timeout)
        if (inb(0x64u) & 1u) return 1;
    return 0;
}

static int mouse_command(uint8_t command) {
    if (!wait_input_empty()) return 0;
    outb(0x64u, 0xD4u);
    if (!wait_input_empty()) return 0;
    outb(0x60u, command);
    if (!wait_output_full()) return 0;
    return inb(0x60u) == 0xFAu;
}

int mouse_init(struct mouse_state *state, uint32_t width, uint32_t height) {
    state->x = (int32_t)(width / 2u);
    state->y = (int32_t)(height / 2u);
    state->buttons = state->clicked = state->changed = 0;
    cycle = 0;
    if (!wait_input_empty()) return 0;
    outb(0x64u, 0xA8u);
    if (!wait_input_empty()) return 0;
    outb(0x64u, 0x20u);
    if (!wait_output_full()) return 0;
    uint8_t controller = inb(0x60u);
#ifdef UEFI_APP
    /* The EFI frontend polls port 0x60 directly and needs no IRQ12 handler. */
    controller &= (uint8_t)~2u;
#else
    controller |= 2u;
#endif
    controller &= (uint8_t)~0x20u;
    if (!wait_input_empty()) return 0;
    outb(0x64u, 0x60u);
    if (!wait_input_empty()) return 0;
    outb(0x60u, controller);
    return mouse_command(0xF6u) && mouse_command(0xF4u);
}

void mouse_poll(struct mouse_state *state, uint32_t width, uint32_t height) {
    state->clicked = 0;
    state->changed = 0;
    /* Drain the controller queue so motion cannot build up behind the UI. */
    for (uint32_t bytes = 0u; bytes < 48u; ++bytes) {
        uint8_t status = inb(0x64u);
        if (!(status & 1u) || !(status & 0x20u)) break;
        uint8_t data = inb(0x60u);
        if (cycle == 0u && !(data & 8u)) continue;
        packet[cycle++] = data;
        if (cycle != 3u) continue;
        cycle = 0u;
        if (packet[0] & 0xC0u) continue;
        int32_t old_x = state->x;
        int32_t old_y = state->y;
        uint8_t old_buttons = state->buttons;
        state->x += (int8_t)packet[1];
        state->y -= (int8_t)packet[2];
        if (state->x < 0) state->x = 0;
        if (state->y < 0) state->y = 0;
        if (state->x >= (int32_t)width) state->x = (int32_t)width - 1;
        if (state->y >= (int32_t)height) state->y = (int32_t)height - 1;
        state->buttons = packet[0] & 7u;
        if ((state->buttons & 1u) && !(old_buttons & 1u)) state->clicked = 1u;
        if (state->x != old_x || state->y != old_y || state->buttons != old_buttons)
            state->changed = 1u;
    }
}
