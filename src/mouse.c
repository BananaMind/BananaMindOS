#include <stdint.h>

#include "mouse.h"

static uint8_t packet[3];
static uint8_t cycle;
static uint8_t vmware_untagged;
static uint8_t ambiguous[3];
static uint8_t ambiguous_count;
static uint8_t keyboard_queue[32];
static uint8_t keyboard_head;
static uint8_t keyboard_tail;

static int cpuid_available(void) {
#if defined(__x86_64__)
    return 1;
#else
    uint32_t original, changed;
    __asm__ volatile ("pushfl\n\tpopl %0" : "=r"(original));
    uint32_t requested = original ^ (1u << 21);
    __asm__ volatile ("pushl %0\n\tpopfl" : : "r"(requested) : "cc");
    __asm__ volatile ("pushfl\n\tpopl %0" : "=r"(changed));
    __asm__ volatile ("pushl %0\n\tpopfl" : : "r"(original) : "cc");
    return ((changed ^ original) & (1u << 21)) != 0u;
#endif
}

static int running_on_vmware(void) {
    if (!cpuid_available()) return 0;
    uint32_t maximum, b, c, d;
    __asm__ volatile ("cpuid"
                      : "=a"(maximum), "=b"(b), "=c"(c), "=d"(d)
                      : "a"(0x40000000u), "c"(0u));
    return maximum >= 0x40000000u &&
        b == 0x61774D56u && c == 0x4D566572u && d == 0x65726177u;
}

static void queue_keyboard(uint8_t value) {
    if (vmware_untagged) {
        uint8_t scan = value & 0x7Fu;
        /* Workstation occasionally exposes mouse delta bytes without the
           i8042 AUX flag. Values 0x2A/0x36 followed by 0x35 are interpreted
           as Shift + slash by a keyboard decoder, producing the observed
           endless '?' stream. Never promote those ambiguous bytes to keys;
           the mouse packet path still consumes them normally. */
        if (scan == 42u || scan == 54u || scan == 53u) return;
    }
    uint8_t next = (uint8_t)((keyboard_tail + 1u) & 31u);
    if (next == keyboard_head) return;
    keyboard_queue[keyboard_tail] = value;
    keyboard_tail = next;
}

int mouse_take_keyboard_byte(uint8_t *value) {
    /* In VMware every controller byte must pass through mouse_poll(), because
       Workstation does not consistently set the i8042 AUX bit.  Return -1
       when its routed keyboard queue is empty so keyboard_char() does not
       bypass the classifier and consume a mouse delta directly from 0x60. */
    if (keyboard_head == keyboard_tail) return vmware_untagged ? -1 : 0;
    *value = keyboard_queue[keyboard_head];
    keyboard_head = (uint8_t)((keyboard_head + 1u) & 31u);
    return 1;
}

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

static int mouse_response(uint8_t *response) {
    for (uint32_t timeout = 0; timeout < 1000000u; ++timeout) {
        uint8_t status = inb(0x64u);
        if (!(status & 1u)) continue;
        uint8_t value = inb(0x60u);
        if (!(status & 0x20u)) continue;
        *response = value;
        return 1;
    }
    return 0;
}

static int mouse_command(uint8_t command) {
    for (uint32_t attempt = 0u; attempt < 3u; ++attempt) {
        if (!wait_input_empty()) return 0;
        outb(0x64u, 0xD4u);
        if (!wait_input_empty()) return 0;
        outb(0x60u, command);
        uint8_t response;
        if (!mouse_response(&response)) return 0;
        if (response == 0xFAu) return 1;
        if (response != 0xFEu) return 0;
    }
    return 0;
}

int mouse_init(struct mouse_state *state, uint32_t width, uint32_t height) {
    state->x = (int32_t)(width / 2u);
    state->y = (int32_t)(height / 2u);
    state->buttons = state->clicked = state->changed = 0;
    cycle = 0;
    ambiguous_count = 0u;
    keyboard_head = keyboard_tail = 0u;
    vmware_untagged = (uint8_t)running_on_vmware();
    /* Discard stale firmware/GRUB bytes before waiting for mouse ACKs. */
    for (uint32_t stale = 0u; stale < 32u && (inb(0x64u) & 1u); ++stale)
        (void)inb(0x60u);
    if (!wait_input_empty()) goto failed;
    outb(0x64u, 0xA8u);
    if (!wait_input_empty()) goto failed;
    outb(0x64u, 0x20u);
    uint8_t controller = 0u;
    uint8_t received_controller = 0u;
    for (uint32_t timeout = 0u; timeout < 1000000u; ++timeout) {
        uint8_t status = inb(0x64u);
        if (status & 1u) {
            controller = inb(0x60u);
            received_controller = 1u;
            break;
        }
    }
    if (!received_controller) goto failed;
    /* Both frontends poll port 0x60 directly and need no IRQ12 handler. */
    controller &= (uint8_t)~2u;
    controller &= (uint8_t)~0x20u;
    if (!wait_input_empty()) goto failed;
    outb(0x64u, 0x60u);
    if (!wait_input_empty()) goto failed;
    outb(0x60u, controller);
    if (!mouse_command(0xF6u) || !mouse_command(0xF4u)) goto failed;
    return 1;

failed:
    /* Workstation may reject legacy setup commands while continuing to
       deliver its virtual PS/2 stream. Keep the VMware router active so the
       pointer works and untagged packets cannot enter keyboard input. */
    if (vmware_untagged) return 1;
    /* Do not leave a half-configured mouse streaming bytes into the shared
       keyboard controller. Keyboard navigation remains usable. */
    if (wait_input_empty()) outb(0x64u, 0xA7u);
    for (uint32_t stale = 0u; stale < 48u && (inb(0x64u) & 1u); ++stale)
        (void)inb(0x60u);
    return 0;
}

static void feed_mouse_byte(struct mouse_state *state, uint32_t width,
                            uint32_t height, uint8_t data) {
    if (cycle == 0u && !(data & 8u)) return;
    packet[cycle++] = data;
    if (cycle != 3u) return;
    cycle = 0u;
    if (packet[0] & 0xC0u) return;
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
    if (state->x != old_x || state->y != old_y ||
        state->buttons != old_buttons)
        state->changed = 1u;
}

static void shift_ambiguous(void) {
    for (uint8_t index = 1u; index < ambiguous_count; ++index)
        ambiguous[index - 1u] = ambiguous[index];
    --ambiguous_count;
}

static uint8_t route_ambiguous(struct mouse_state *state, uint32_t width,
                               uint32_t height) {
    uint8_t discarded = 0u;
    while (ambiguous_count == 3u) {
        uint8_t first = ambiguous[0];
        uint8_t x_negative = (uint8_t)((ambiguous[1] & 0x80u) != 0u);
        uint8_t y_negative = (uint8_t)((ambiguous[2] & 0x80u) != 0u);
        if ((first & 8u) && !(first & 0xC0u) &&
            x_negative == (uint8_t)((first & 0x10u) != 0u) &&
            y_negative == (uint8_t)((first & 0x20u) != 0u)) {
            feed_mouse_byte(state, width, height, ambiguous[0]);
            feed_mouse_byte(state, width, height, ambiguous[1]);
            feed_mouse_byte(state, width, height, ambiguous[2]);
            ambiguous_count = 0u;
            return discarded;
        }
        /* Three immediately adjacent bytes that do not form a valid packet
           are a misaligned VMware mouse burst. Drop one byte and search for
           the next packet boundary; returning it as a key caused the '?'
           flood. Human key make/break events normally arrive separately and
           are handled by the one/two-byte timeout path below. */
        shift_ambiguous();
        discarded = 1u;
    }
    return discarded;
}

void mouse_poll(struct mouse_state *state, uint32_t width, uint32_t height) {
    state->clicked = 0;
    state->changed = 0;
    uint8_t resynchronizing = 0u;
    /* Drain the controller queue so motion cannot build up behind the UI. */
    for (uint32_t bytes = 0u; bytes < 48u; ++bytes) {
        uint8_t status = inb(0x64u);
        if (!(status & 1u)) break;
        if (!(status & 0x20u) && !vmware_untagged) break;
        uint8_t data = inb(0x60u);
        if (status & 0x20u) {
            while (ambiguous_count) {
                queue_keyboard(ambiguous[0]);
                shift_ambiguous();
            }
            feed_mouse_byte(state, width, height, data);
        } else {
            if (ambiguous_count < 3u) ambiguous[ambiguous_count++] = data;
            resynchronizing |= route_ambiguous(state, width, height);
        }
    }

    /* VMware can omit the AUX status bit. Give a possible packet a short
       window to receive all three bytes, then return unmatched bytes to the
       keyboard decoder instead of turning mouse deltas into prompt text. */
    if (vmware_untagged && ambiguous_count) {
        for (uint32_t wait = 0u; wait < 20000u && ambiguous_count; ++wait) {
            uint8_t status = inb(0x64u);
            if (!(status & 1u)) continue;
            uint8_t data = inb(0x60u);
            if (status & 0x20u) {
                while (ambiguous_count) {
                    queue_keyboard(ambiguous[0]);
                    shift_ambiguous();
                }
                feed_mouse_byte(state, width, height, data);
                break;
            }
            if (ambiguous_count < 3u) ambiguous[ambiguous_count++] = data;
            resynchronizing |= route_ambiguous(state, width, height);
        }
        while (ambiguous_count) {
            if (resynchronizing) {
                shift_ambiguous();
            } else {
                queue_keyboard(ambiguous[0]);
                shift_ambiguous();
            }
        }
    }
}
