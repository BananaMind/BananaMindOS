#include <stdint.h>
#include <stddef.h>
#include "bm2n.h"
#include "font8x8.h"

#define MULTIBOOT_BOOTLOADER_MAGIC 0x2BADB002u
#define MULTIBOOT_INFO_MEMORY      0x00000001u
#define MULTIBOOT_INFO_MODS        0x00000008u
#define MULTIBOOT_INFO_FRAMEBUFFER 0x00001000u
#define VGA_WIDTH 80
#define VGA_HEIGHT 25
#define MAX_LAYERS 14
#define MAX_HIDDEN 384
#define MAX_FF 1024
#define MAX_HEADS 6
#define MAX_KV_HEADS 2
#define MAX_HEAD_DIM 64
#define MAX_VOCAB 8192
#define REFRESH_KERNEL 9
#define MAX_INPUT_BYTES 512
#define MAX_BPE_TOKENS 512
#define MERGE_SLOTS 16384

struct __attribute__((packed)) multiboot_info {
    uint32_t flags, mem_lower, mem_upper, boot_device, cmdline;
    uint32_t mods_count, mods_addr;
    uint32_t symbols[4];
    uint32_t mmap_length, mmap_addr, drives_length, drives_addr;
    uint32_t config_table, boot_loader_name, apm_table;
    uint32_t vbe_control_info, vbe_mode_info;
    uint16_t vbe_mode, vbe_interface_seg, vbe_interface_off, vbe_interface_len;
    uint64_t framebuffer_addr;
    uint32_t framebuffer_pitch, framebuffer_width, framebuffer_height;
    uint8_t framebuffer_bpp, framebuffer_type;
    uint16_t framebuffer_reserved;
    uint8_t red_field_position, red_mask_size;
    uint8_t green_field_position, green_mask_size;
    uint8_t blue_field_position, blue_mask_size;
};

struct multiboot_module {
    uint32_t start, end, string, reserved;
};

struct matrix {
    const uint8_t *data;
    uint32_t rows, cols, stride, bits;
};

struct layer {
    const float *ln1;
    struct matrix q, k, v, o;
    const float *qnorm, *knorm, *ln2;
    const float *refresh_att_norm, *refresh_emb_norm, *refresh_out_norm;
    const float *refresh_alpha;
    struct matrix refresh_gate, refresh_value, refresh_out, refresh_kernel;
    struct matrix gate, up, down;
};

enum model_kind { MODEL_BANANA2, MODEL_MICRO2, MODEL_MICRO_V1 };

struct model {
    const struct bm2n_header *header;
    struct matrix embedding;
    struct layer layers[MAX_LAYERS];
    const float *final_norm;
    const struct bm2n_token *tokens;
    const uint8_t *token_data;
    const struct bm2n_merge *merges;
    enum model_kind kind;
};

struct merge_slot {
    uint32_t key;
    uint16_t result;
    uint16_t rank_plus_one;
};

static volatile uint16_t *const vga = (volatile uint16_t *)0xB8000;
static uint32_t cursor_row, cursor_col;
static uint8_t color = 0x60;
static uint8_t gui_enabled;
static uint8_t chat_mode;
static volatile uint8_t *framebuffer;
static uint32_t fb_pitch, fb_width, fb_height;
static uint8_t fb_bpp, fb_bytes;
static uint8_t fb_red_pos, fb_green_pos, fb_blue_pos;
static uint32_t gui_x, gui_y, gui_x0, gui_y0, gui_x1, gui_y1;
static uint32_t gui_background = 2;
static struct model net;
static struct merge_slot merge_table[MERGE_SLOTS];

/* The fixed buffers are the reason BM2N_CONTEXT is capped at 64. */
static float *key_cache;
static float *value_cache;
static float *refresh_history;
static float x[MAX_HIDDEN], xb[MAX_HIDDEN], tmp[MAX_HIDDEN], embedbuf[MAX_HIDDEN];
static float qbuf[MAX_HIDDEN], kbuf[MAX_KV_HEADS * MAX_HEAD_DIM];
static float vbuf[MAX_KV_HEADS * MAX_HEAD_DIM], att[MAX_HIDDEN];
static float gatebuf[MAX_FF], upbuf[MAX_FF];
static float refresh_gatebuf[MAX_HIDDEN], refresh_valuebuf[MAX_HIDDEN];
static float scores[BM2N_CONTEXT], logits[MAX_VOCAB];
static uint16_t bpe_tokens[MAX_BPE_TOKENS], prompt_tokens[BM2N_CONTEXT];
static uint16_t byte_token[256];
static uint8_t chat_input[MAX_INPUT_BYTES];

#define CACHE_INDEX(layer, position, head, dim) \
    (((((layer) * BM2N_CONTEXT + (position)) * net.header->num_kv_heads + (head)) * net.header->head_dim) + (dim))
#define KEY_AT(layer, position, head, dim) key_cache[CACHE_INDEX(layer, position, head, dim)]
#define VALUE_AT(layer, position, head, dim) value_cache[CACHE_INDEX(layer, position, head, dim)]
#define REFRESH_AT(layer, channel, slot) \
    refresh_history[(((layer) * net.header->hidden_size + (channel)) * (REFRESH_KERNEL - 1u)) + (slot)]

static inline void outb(uint16_t port, uint8_t value) {
    __asm__ volatile ("outb %0, %1" : : "a"(value), "Nd"(port));
}

static inline uint8_t inb(uint16_t port) {
    uint8_t value;
    __asm__ volatile ("inb %1, %0" : "=a"(value) : "Nd"(port));
    return value;
}

static void serial_init(void) {
    outb(0x3F8 + 1, 0x00);
    outb(0x3F8 + 3, 0x80);
    outb(0x3F8 + 0, 0x01);
    outb(0x3F8 + 1, 0x00);
    outb(0x3F8 + 3, 0x03);
    outb(0x3F8 + 2, 0xC7);
    outb(0x3F8 + 4, 0x0B);
}

static void fpu_init(void) {
    uint32_t cr0;
    __asm__ volatile ("mov %%cr0, %0" : "=r"(cr0));
    cr0 &= ~((1u << 2) | (1u << 3)); /* clear EM and TS */
    cr0 |= (1u << 1) | (1u << 5);    /* set MP and NE */
    __asm__ volatile ("mov %0, %%cr0\n\tfninit" : : "r"(cr0));
}

enum ui_color { UI_YELLOW, UI_INK, UI_PANEL, UI_WHITE, UI_RED, UI_GREEN, UI_SHADOW };

static const uint32_t ui_rgb[] = {
    0xF4C430u, 0x30260Fu, 0xFFF3B0u, 0xFFFFFFu,
    0xC62828u, 0x237A36u, 0xC89B16u
};

static void palette_init(void) {
    outb(0x3C8, 0);
    for (uint32_t i = 0; i < sizeof(ui_rgb) / sizeof(ui_rgb[0]); ++i) {
        uint32_t rgb = ui_rgb[i];
        outb(0x3C9, (uint8_t)(((rgb >> 16) & 255u) >> 2));
        outb(0x3C9, (uint8_t)(((rgb >> 8) & 255u) >> 2));
        outb(0x3C9, (uint8_t)((rgb & 255u) >> 2));
    }
}

static uint32_t direct_color(uint32_t index) {
    uint32_t rgb = ui_rgb[index];
    uint32_t red = (rgb >> 16) & 255u, green = (rgb >> 8) & 255u, blue = rgb & 255u;
    return (red << fb_red_pos) | (green << fb_green_pos) | (blue << fb_blue_pos);
}

static void pixel(uint32_t px, uint32_t py, uint32_t palette_color) {
    if (px >= fb_width || py >= fb_height) return;
    volatile uint8_t *address = framebuffer + py * fb_pitch + px * fb_bytes;
    if (fb_bpp == 8) {
        address[0] = (uint8_t)palette_color;
        return;
    }
    uint32_t value = direct_color(palette_color);
    address[0] = (uint8_t)value;
    address[1] = (uint8_t)(value >> 8);
    if (fb_bytes > 2) address[2] = (uint8_t)(value >> 16);
    if (fb_bytes > 3) address[3] = (uint8_t)(value >> 24);
}

static void fill_rect(uint32_t left, uint32_t top, uint32_t width, uint32_t height,
                      uint32_t palette_color) {
    uint32_t right = left + width, bottom = top + height;
    if (right > fb_width) right = fb_width;
    if (bottom > fb_height) bottom = fb_height;
    for (uint32_t py = top; py < bottom; ++py)
        for (uint32_t px = left; px < right; ++px) pixel(px, py, palette_color);
}

static void draw_glyph(uint32_t left, uint32_t top, char character,
                       uint32_t foreground, uint32_t background, uint32_t scale) {
    uint8_t code = (uint8_t)character;
    if (code < 32 || code > 127) code = '?';
    const uint8_t *glyph = font8x8[code - 32];
    for (uint32_t row = 0; row < 8; ++row)
        for (uint32_t col = 0; col < 8; ++col) {
            uint32_t ink = glyph[row] & (0x80u >> col) ? foreground : background;
            fill_rect(left + col * scale, top + row * scale, scale, scale, ink);
        }
}

static void draw_label(uint32_t left, uint32_t top, const char *text, uint32_t scale) {
    while (*text) {
        draw_glyph(left, top, *text++, UI_INK, UI_YELLOW, scale);
        left += 8u * scale;
    }
}

static void gui_set_region(uint32_t left, uint32_t top, uint32_t right,
                           uint32_t bottom, uint32_t background) {
    gui_x0 = left; gui_y0 = top; gui_x1 = right; gui_y1 = bottom;
    gui_x = gui_x0; gui_y = gui_y0; gui_background = background;
}

static void gui_scroll(void) {
    uint32_t row_bytes = (gui_x1 - gui_x0) * fb_bytes;
    for (uint32_t py = gui_y0; py + 8 < gui_y1; ++py) {
        volatile uint8_t *to = framebuffer + py * fb_pitch + gui_x0 * fb_bytes;
        volatile uint8_t *from = framebuffer + (py + 8) * fb_pitch + gui_x0 * fb_bytes;
        for (uint32_t byte = 0; byte < row_bytes; ++byte) to[byte] = from[byte];
    }
    fill_rect(gui_x0, gui_y1 - 8, gui_x1 - gui_x0, 8, gui_background);
    gui_y = gui_y1 - 8;
}

static void gui_putc(char c) {
    if (c == '\n') {
        gui_x = gui_x0; gui_y += 8;
        if (gui_y + 8 > gui_y1) gui_scroll();
        return;
    }
    if (c == '\b') {
        if (gui_x >= gui_x0 + 8) gui_x -= 8;
        fill_rect(gui_x, gui_y, 8, 8, gui_background);
        return;
    }
    draw_glyph(gui_x, gui_y, c, UI_INK, gui_background, 1);
    gui_x += 8;
    if (gui_x + 8 > gui_x1) {
        gui_x = gui_x0; gui_y += 8;
        if (gui_y + 8 > gui_y1) gui_scroll();
    }
}

static uint32_t input_top(void) { return fb_height - 64u; }

static void gui_output_begin(void) {
    uint32_t top = 44, bottom = input_top() - 12u;
    fill_rect(16, top, fb_width - 32u, bottom - top, UI_PANEL);
    draw_glyph(20, top + 4, 'R', UI_INK, UI_PANEL, 1);
    draw_glyph(28, top + 4, 'E', UI_INK, UI_PANEL, 1);
    draw_glyph(36, top + 4, 'S', UI_INK, UI_PANEL, 1);
    draw_glyph(44, top + 4, 'P', UI_INK, UI_PANEL, 1);
    draw_glyph(52, top + 4, 'O', UI_INK, UI_PANEL, 1);
    draw_glyph(60, top + 4, 'N', UI_INK, UI_PANEL, 1);
    draw_glyph(68, top + 4, 'S', UI_INK, UI_PANEL, 1);
    draw_glyph(76, top + 4, 'E', UI_INK, UI_PANEL, 1);
    gui_set_region(20, top + 16, fb_width - 20u, bottom - 4u, UI_PANEL);
}

static void gui_input_begin(void) {
    uint32_t top = input_top();
    fill_rect(16, top, fb_width - 32u, fb_height - top - 16u, UI_WHITE);
    const char *label = "PROMPT >";
    uint32_t left = 20;
    while (*label) { draw_glyph(left, top + 5, *label++, UI_INK, UI_WHITE, 1); left += 8; }
    gui_set_region(20, top + 20, fb_width - 20u, fb_height - 20u, UI_WHITE);
}

static void gui_draw_shell(void) {
    fill_rect(0, 0, fb_width, fb_height, UI_YELLOW);
    draw_label(16, 10, "BANANAMIND OS", 2);
    uint32_t top = 40, in_top = input_top();
    fill_rect(15, top + 3, fb_width - 24u, in_top - top - 5u, UI_SHADOW);
    fill_rect(12, top, fb_width - 24u, in_top - top - 5u, UI_INK);
    fill_rect(16, top + 4, fb_width - 32u, in_top - top - 13u, UI_PANEL);
    fill_rect(15, in_top + 3, fb_width - 24u, fb_height - in_top - 9u, UI_SHADOW);
    fill_rect(12, in_top, fb_width - 24u, fb_height - in_top - 12u, UI_INK);
    fill_rect(16, in_top + 4, fb_width - 32u, fb_height - in_top - 20u, UI_WHITE);
    gui_output_begin();
}

static void gui_show_tps(uint32_t tenths) {
    if (!gui_enabled) return;
    uint32_t left = fb_width - 112u;
    fill_rect(left, 8, 96, 20, UI_YELLOW);
    const char *label = "TPS ";
    while (*label) { draw_glyph(left, 12, *label++, UI_INK, UI_YELLOW, 1); left += 8; }
    if (tenths == 0xFFFFFFFFu) {
        draw_glyph(left, 12, '-', UI_INK, UI_YELLOW, 1); left += 8;
        draw_glyph(left, 12, '-', UI_INK, UI_YELLOW, 1); left += 8;
        draw_glyph(left, 12, '.', UI_INK, UI_YELLOW, 1); left += 8;
        draw_glyph(left, 12, '-', UI_INK, UI_YELLOW, 1);
        return;
    }
    if (tenths > 999u) tenths = 999u;
    uint32_t whole = tenths / 10u;
    if (whole >= 10u) { draw_glyph(left, 12, (char)('0' + whole / 10u), UI_INK, UI_YELLOW, 1); left += 8; }
    draw_glyph(left, 12, (char)('0' + whole % 10u), UI_INK, UI_YELLOW, 1); left += 8;
    draw_glyph(left, 12, '.', UI_INK, UI_YELLOW, 1); left += 8;
    draw_glyph(left, 12, (char)('0' + tenths % 10u), UI_INK, UI_YELLOW, 1);
}

static void gui_init(const struct multiboot_info *mb) {
    if (!(mb->flags & MULTIBOOT_INFO_FRAMEBUFFER) || (uint32_t)(mb->framebuffer_addr >> 32) ||
        mb->framebuffer_width < 320 || mb->framebuffer_height < 200) return;
    if (mb->framebuffer_type == 0 && mb->framebuffer_bpp == 8) {
        fb_red_pos = 16; fb_green_pos = 8; fb_blue_pos = 0;
    } else if (mb->framebuffer_type == 1 &&
               (mb->framebuffer_bpp == 15 || mb->framebuffer_bpp == 16 ||
                mb->framebuffer_bpp == 24 || mb->framebuffer_bpp == 32)) {
        fb_red_pos = mb->red_field_position;
        fb_green_pos = mb->green_field_position;
        fb_blue_pos = mb->blue_field_position;
    } else return;
    framebuffer = (volatile uint8_t *)(uint32_t)mb->framebuffer_addr;
    fb_pitch = mb->framebuffer_pitch; fb_width = mb->framebuffer_width;
    fb_height = mb->framebuffer_height; fb_bpp = mb->framebuffer_bpp;
    fb_bytes = (uint8_t)((fb_bpp + 7u) / 8u);
    gui_enabled = 1;
    if (fb_bpp == 8) palette_init();
}

static void serial_putc(char c) {
    while (!(inb(0x3F8 + 5) & 0x20)) { }
    outb(0x3F8, (uint8_t)c);
}

static void scroll(void) {
    if (cursor_row < VGA_HEIGHT) return;
    for (uint32_t row = 1; row < VGA_HEIGHT; ++row)
        for (uint32_t col = 0; col < VGA_WIDTH; ++col)
            vga[(row - 1) * VGA_WIDTH + col] = vga[row * VGA_WIDTH + col];
    for (uint32_t col = 0; col < VGA_WIDTH; ++col)
        vga[(VGA_HEIGHT - 1) * VGA_WIDTH + col] = (uint16_t)color << 8 | ' ';
    cursor_row = VGA_HEIGHT - 1;
}

static void putc(char c) {
    if (c == '\n') {
        serial_putc('\r');
        serial_putc('\n');
        if (gui_enabled) { gui_putc(c); return; }
        cursor_col = 0;
        ++cursor_row;
        scroll();
        return;
    }
    if (c == '\b') {
        if (gui_enabled) {
            gui_putc(c);
            serial_putc('\b'); serial_putc(' '); serial_putc('\b');
            return;
        }
        if (cursor_col) --cursor_col;
        vga[cursor_row * VGA_WIDTH + cursor_col] = (uint16_t)color << 8 | ' ';
        serial_putc('\b'); serial_putc(' '); serial_putc('\b');
        return;
    }
    serial_putc(c);
    if (gui_enabled) { gui_putc(c); return; }
    vga[cursor_row * VGA_WIDTH + cursor_col] = (uint16_t)color << 8 | (uint8_t)c;
    if (++cursor_col == VGA_WIDTH) { cursor_col = 0; ++cursor_row; scroll(); }
}

static void puts(const char *s) { while (*s) putc(*s++); }

static void putu(uint32_t value) {
    char buf[11]; uint32_t n = 0;
    if (!value) { putc('0'); return; }
    while (value) { buf[n++] = (char)('0' + value % 10); value /= 10; }
    while (n) putc(buf[--n]);
}

static int memeq(const void *a_, const void *b_, uint32_t size) {
    const uint8_t *a = a_, *b = b_;
    while (size--) if (*a++ != *b++) return 0;
    return 1;
}

static int begins_with_chat(const char *text) {
    return text && text[0] == 'c' && text[1] == 'h' && text[2] == 'a' && text[3] == 't';
}

static uint32_t chat_wrap(const uint8_t *input, uint32_t length) {
    static const char prefix[] = "<|user|>\n";
    static const char suffix[] = "\n<|assistant|>\n";
    uint32_t written = 0;
    for (uint32_t i = 0; prefix[i] && written < sizeof(chat_input); ++i)
        chat_input[written++] = (uint8_t)prefix[i];
    uint32_t room = sizeof(chat_input) - written - (sizeof(suffix) - 1u);
    if (length > room) length = room;
    for (uint32_t i = 0; i < length; ++i) chat_input[written++] = input[i];
    for (uint32_t i = 0; suffix[i] && written < sizeof(chat_input); ++i)
        chat_input[written++] = (uint8_t)suffix[i];
    return written;
}

static int range_ok(uint32_t offset, uint32_t size, uint32_t file_size) {
    return offset <= file_size && size <= file_size - offset;
}

static const uint8_t *parse_vector(const uint8_t *p, const uint8_t *end,
                                   uint32_t count, const float **result) {
    uint32_t bytes = count * 4u;
    if ((uint32_t)(end - p) < bytes) return 0;
    *result = (const float *)p;
    return p + bytes;
}

static const uint8_t *parse_matrix(const uint8_t *p, const uint8_t *end,
                                   uint32_t rows, uint32_t cols, uint32_t bits,
                                   struct matrix *result) {
    uint32_t stride;
    if (bits == 32u) stride = cols * 4u;
    else if (bits == 16u) stride = cols * 2u;
    else stride = 4u + (cols * bits + 7u) / 8u;
    uint32_t bytes = rows * stride;
    if ((uint32_t)(end - p) < bytes) return 0;
    result->data = p; result->rows = rows; result->cols = cols;
    result->stride = stride; result->bits = bits;
    return p + bytes;
}

static uint32_t merge_hash(uint32_t key) { return (key * 2654435761u) & (MERGE_SLOTS - 1); }

static int build_merge_table(void) {
    for (uint32_t i = 0; i < MERGE_SLOTS; ++i) merge_table[i].key = 0;
    for (uint32_t i = 0; i < net.header->merges_count; ++i) {
        const struct bm2n_merge *m = &net.merges[i];
        uint32_t key = (((uint32_t)m->left << 16) | m->right) + 1u;
        uint32_t slot = merge_hash(key);
        while (merge_table[slot].key && merge_table[slot].key != key)
            slot = (slot + 1) & (MERGE_SLOTS - 1);
        merge_table[slot].key = key;
        merge_table[slot].result = m->result;
        merge_table[slot].rank_plus_one = (uint16_t)(m->rank + 1u);
    }
    for (uint32_t b = 0; b < 256; ++b) byte_token[b] = net.header->unk_id;
    for (uint32_t id = 0; id < net.header->vocab_size; ++id) {
        const struct bm2n_token *t = &net.tokens[id];
        if (t->length == 1) byte_token[net.token_data[t->offset]] = (uint16_t)id;
    }
    return 1;
}

static int load_model(const uint8_t *base, uint32_t module_size) {
    if (module_size < sizeof(struct bm2n_header)) return 0;
    const struct bm2n_header *h = (const struct bm2n_header *)base;
    if (!memeq(h->magic, BM2N_MAGIC, 8) || h->version != BM2N_VERSION ||
        h->header_size != sizeof(*h) || h->file_size > module_size ||
        (h->bits != 2 && h->bits != 4 && h->bits != 8 && h->bits != 16 && h->bits != 32) ||
        h->vocab_size > MAX_VOCAB || h->hidden_size > MAX_HIDDEN ||
        h->num_layers > MAX_LAYERS || h->num_heads > MAX_HEADS ||
        h->num_kv_heads > MAX_KV_HEADS || h->head_dim > MAX_HEAD_DIM ||
        h->intermediate_size > MAX_FF) return 0;
    if (h->vocab_size == 2048u && h->hidden_size == 128u && h->num_layers == 9u)
        net.kind = MODEL_MICRO2;
    else if (h->vocab_size == 1536u && h->hidden_size == 128u && h->num_layers == 4u)
        net.kind = MODEL_MICRO_V1;
    else if (h->vocab_size == 8192u &&
             ((h->hidden_size == 256u && h->num_layers == 10u) ||
              (h->hidden_size == 384u && h->num_layers == 14u)))
        net.kind = MODEL_BANANA2;
    else return 0;
    if (!range_ok(h->weights_offset, h->weights_size, h->file_size) ||
        !range_ok(h->token_index_offset, h->vocab_size * sizeof(struct bm2n_token), h->file_size) ||
        !range_ok(h->token_data_offset, h->token_data_size, h->file_size) ||
        !range_ok(h->merges_offset, h->merges_count * sizeof(struct bm2n_merge), h->file_size) ||
        h->merges_count >= 65535u) return 0;

    net.header = h;
    const uint8_t *p = base + h->weights_offset;
    const uint8_t *end = p + h->weights_size;
    uint32_t hidden = h->hidden_size, kv = h->num_kv_heads * h->head_dim;
    p = parse_matrix(p, end, h->vocab_size, hidden, h->bits, &net.embedding);
    for (uint32_t i = 0; p && i < h->num_layers; ++i) {
        struct layer *l = &net.layers[i];
        p = parse_vector(p, end, hidden, &l->ln1);
        if (p) p = parse_matrix(p, end, hidden, hidden, h->bits, &l->q);
        if (p) p = parse_matrix(p, end, kv, hidden, h->bits, &l->k);
        if (p) p = parse_matrix(p, end, kv, hidden, h->bits, &l->v);
        if (p) p = parse_matrix(p, end, hidden, hidden, h->bits, &l->o);
        if (net.kind != MODEL_MICRO_V1) {
            if (p) p = parse_vector(p, end, h->head_dim, &l->qnorm);
            if (p) p = parse_vector(p, end, h->head_dim, &l->knorm);
        }
        if (net.kind == MODEL_MICRO2) {
            if (p) p = parse_vector(p, end, hidden, &l->refresh_att_norm);
            if (p) p = parse_vector(p, end, hidden, &l->refresh_emb_norm);
            if (p) p = parse_vector(p, end, hidden, &l->refresh_out_norm);
            if (p) p = parse_matrix(p, end, hidden, hidden, h->bits, &l->refresh_gate);
            if (p) p = parse_matrix(p, end, hidden, hidden, h->bits, &l->refresh_value);
            if (p) p = parse_matrix(p, end, hidden, hidden, h->bits, &l->refresh_out);
            if (p) p = parse_matrix(p, end, hidden, REFRESH_KERNEL, h->bits, &l->refresh_kernel);
            if (p) p = parse_vector(p, end, 1u, &l->refresh_alpha);
        }
        if (p) p = parse_vector(p, end, hidden, &l->ln2);
        if (p) p = parse_matrix(p, end, h->intermediate_size, hidden, h->bits, &l->gate);
        if (p) p = parse_matrix(p, end, h->intermediate_size, hidden, h->bits, &l->up);
        if (p) p = parse_matrix(p, end, hidden, h->intermediate_size, h->bits, &l->down);
    }
    if (p) p = parse_vector(p, end, hidden, &net.final_norm);
    if (!p || p != end) return 0;
    net.tokens = (const struct bm2n_token *)(base + h->token_index_offset);
    net.token_data = base + h->token_data_offset;
    net.merges = (const struct bm2n_merge *)(base + h->merges_offset);
    for (uint32_t id = 0; id < h->vocab_size; ++id)
        if (net.tokens[id].offset > h->token_data_size ||
            net.tokens[id].length > h->token_data_size - net.tokens[id].offset) return 0;
    return build_merge_table();
}

static int8_t signed_nibble(uint8_t value) {
    value &= 15u;
    return value & 8u ? (int8_t)(value | 0xF0u) : (int8_t)value;
}

static int weight_at(const uint8_t *packed, uint32_t index, uint32_t bits) {
    if (bits == 8) return (int8_t)packed[index];
    if (bits == 4) return signed_nibble(packed[index >> 1] >> ((index & 1u) * 4u));
    return (int)((packed[index >> 2] >> ((index & 3u) * 2u)) & 3u) - 1;
}

static float half_to_float(uint16_t half) {
    uint32_t sign = (uint32_t)(half & 0x8000u) << 16;
    uint32_t exponent = (half >> 10) & 31u;
    uint32_t mantissa = half & 1023u;
    union { uint32_t u; float f; } value;
    if (exponent == 0u) {
        if (mantissa == 0u) value.u = sign;
        else {
            int shift = 0;
            while (!(mantissa & 1024u)) { mantissa <<= 1; ++shift; }
            mantissa &= 1023u;
            value.u = sign | ((uint32_t)(113 - shift) << 23) | (mantissa << 13);
        }
    } else if (exponent == 31u) value.u = sign | 0x7F800000u | (mantissa << 13);
    else value.u = sign | ((exponent + 112u) << 23) | (mantissa << 13);
    return value.f;
}

static void matrix_row(const struct matrix *m, uint32_t row, float *output) {
    const uint8_t *record = m->data + row * m->stride;
    if (m->bits == 32u) {
        const float *values = (const float *)record;
        for (uint32_t i = 0; i < m->cols; ++i) output[i] = values[i];
        return;
    }
    if (m->bits == 16u) {
        const uint16_t *values = (const uint16_t *)record;
        for (uint32_t i = 0; i < m->cols; ++i) output[i] = half_to_float(values[i]);
        return;
    }
    float scale = *(const float *)record;
    const uint8_t *packed = record + 4;
    for (uint32_t i = 0; i < m->cols; ++i) output[i] = (float)weight_at(packed, i, m->bits) * scale;
}

static void matvec(const struct matrix *m, const float *input, float *output) {
    for (uint32_t row = 0; row < m->rows; ++row) {
        const uint8_t *record = m->data + row * m->stride;
        if (m->bits == 32u) {
            const float *values = (const float *)record;
            float sum = 0.0f;
            for (uint32_t col = 0; col < m->cols; ++col) sum += values[col] * input[col];
            output[row] = sum;
            continue;
        }
        if (m->bits == 16u) {
            const uint16_t *values = (const uint16_t *)record;
            float sum = 0.0f;
            for (uint32_t col = 0; col < m->cols; ++col)
                sum += half_to_float(values[col]) * input[col];
            output[row] = sum;
            continue;
        }
        float scale = *(const float *)record;
        const uint8_t *packed = record + 4;
        float sum = 0.0f;
        for (uint32_t col = 0; col < m->cols; ++col)
            sum += (float)weight_at(packed, col, m->bits) * input[col];
        output[row] = sum * scale;
    }
}

static inline float square_root(float value) {
    __asm__ volatile ("fsqrt" : "+t"(value));
    return value;
}

static void rms_norm(float *output, const float *input, const float *weight, uint32_t size) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < size; ++i) sum += input[i] * input[i];
    float inv = 1.0f / square_root(sum / (float)size + net.header->rms_eps);
    for (uint32_t i = 0; i < size; ++i) output[i] = input[i] * inv * weight[i];
}

static float fast_exp(float value) {
    if (value < -16.0f) return 0.0f;
    if (value > 16.0f) value = 16.0f;
    float z = value * 1.4426950409f;
    int power = (int)z;
    if ((float)power > z) --power;
    float fraction = z - (float)power;
    float poly = 1.0f + fraction * (0.69314718f + fraction *
        (0.24022651f + fraction * (0.05550411f + fraction * 0.00961813f)));
    union { uint32_t u; float f; } scale;
    scale.u = (uint32_t)(power + 127) << 23;
    return poly * scale.f;
}

static void fast_sincos(float angle, float *sine, float *cosine) {
    const float pi = 3.14159265359f, tau = 6.28318530718f;
    while (angle > pi) angle -= tau;
    while (angle < -pi) angle += tau;
    float x2 = angle * angle;
    *sine = angle * (1.0f + x2 * (-0.16666667f + x2 * (0.00833333f + x2 * -0.00019841f)));
    *cosine = 1.0f + x2 * (-0.5f + x2 * (0.04166667f + x2 * -0.00138889f));
}

static void rope(float *values, uint32_t heads, uint32_t position) {
    uint32_t head_dim = net.header->head_dim;
    float frequency_ratio;
    if (head_dim == 64u) frequency_ratio = 0.6978305849f;
    else if (net.header->rope_theta > 50000.0f) frequency_ratio = 0.4869675252f;
    else frequency_ratio = 0.5623413252f;
    for (uint32_t head = 0; head < heads; ++head) {
        float frequency = 1.0f;
        for (uint32_t i = 0; i < head_dim; i += 2) {
            float sine, cosine;
            fast_sincos((float)position * frequency, &sine, &cosine);
            float a = values[head * head_dim + i], b = values[head * head_dim + i + 1];
            values[head * head_dim + i] = a * cosine - b * sine;
            values[head * head_dim + i + 1] = a * sine + b * cosine;
            frequency *= frequency_ratio;
        }
    }
}

static uint16_t forward(uint16_t token, uint32_t position) {
    const struct bm2n_header *h = net.header;
    uint32_t hidden = h->hidden_size, head_dim = h->head_dim;
    uint32_t heads = h->num_heads, kv_heads = h->num_kv_heads;
    uint32_t intermediate = h->intermediate_size;
    matrix_row(&net.embedding, token, x);
    if (net.kind != MODEL_MICRO_V1) {
        float scale = square_root((float)hidden);
        for (uint32_t i = 0; i < hidden; ++i) x[i] *= scale;
    }
    for (uint32_t i = 0; i < hidden; ++i) embedbuf[i] = x[i];
    for (uint32_t li = 0; li < h->num_layers; ++li) {
        struct layer *l = &net.layers[li];
        rms_norm(xb, x, l->ln1, hidden);
        matvec(&l->q, xb, qbuf); matvec(&l->k, xb, kbuf); matvec(&l->v, xb, vbuf);
        if (net.kind != MODEL_MICRO_V1) {
            for (uint32_t head = 0; head < heads; ++head)
                rms_norm(&qbuf[head * head_dim], &qbuf[head * head_dim], l->qnorm, head_dim);
            for (uint32_t head = 0; head < kv_heads; ++head)
                rms_norm(&kbuf[head * head_dim], &kbuf[head * head_dim], l->knorm, head_dim);
        }
        rope(qbuf, heads, position); rope(kbuf, kv_heads, position);
        for (uint32_t kh = 0; kh < kv_heads; ++kh)
            for (uint32_t d = 0; d < head_dim; ++d) {
                KEY_AT(li, position, kh, d) = kbuf[kh * head_dim + d];
                VALUE_AT(li, position, kh, d) = vbuf[kh * head_dim + d];
            }
        float attention_scale = 1.0f / square_root((float)head_dim);
        for (uint32_t head = 0; head < heads; ++head) {
            uint32_t kh = head / (heads / kv_heads);
            float maximum = -1000000.0f;
            for (uint32_t t = 0; t <= position; ++t) {
                float score = 0.0f;
                for (uint32_t d = 0; d < head_dim; ++d)
                    score += qbuf[head * head_dim + d] * KEY_AT(li, t, kh, d);
                scores[t] = score * attention_scale;
                if (scores[t] > maximum) maximum = scores[t];
            }
            float sum = 0.0f;
            for (uint32_t t = 0; t <= position; ++t) { scores[t] = fast_exp(scores[t] - maximum); sum += scores[t]; }
            for (uint32_t d = 0; d < head_dim; ++d) {
                float value = 0.0f;
                for (uint32_t t = 0; t <= position; ++t)
                    value += scores[t] * VALUE_AT(li, t, kh, d);
                att[head * head_dim + d] = value / sum;
            }
        }
        matvec(&l->o, att, tmp);
        for (uint32_t i = 0; i < hidden; ++i) x[i] += tmp[i];
        if (net.kind == MODEL_MICRO2) {
            rms_norm(xb, tmp, l->refresh_att_norm, hidden);
            matvec(&l->refresh_gate, xb, refresh_gatebuf);
            for (uint32_t channel = 0; channel < hidden; ++channel) {
                matrix_row(&l->refresh_kernel, channel, upbuf);
                float convolved = upbuf[REFRESH_KERNEL - 1u] * xb[channel];
                for (uint32_t slot = 0; slot + 1u < REFRESH_KERNEL; ++slot)
                    convolved += upbuf[slot] * REFRESH_AT(li, channel, slot);
                refresh_gatebuf[channel] += convolved;
                for (uint32_t slot = 0; slot + 2u < REFRESH_KERNEL; ++slot)
                    REFRESH_AT(li, channel, slot) = REFRESH_AT(li, channel, slot + 1u);
                REFRESH_AT(li, channel, REFRESH_KERNEL - 2u) = xb[channel];
            }
            rms_norm(tmp, embedbuf, l->refresh_emb_norm, hidden);
            matvec(&l->refresh_value, tmp, refresh_valuebuf);
            for (uint32_t i = 0; i < hidden; ++i)
                refresh_gatebuf[i] = refresh_gatebuf[i] /
                    (1.0f + fast_exp(-refresh_gatebuf[i])) * refresh_valuebuf[i];
            matvec(&l->refresh_out, refresh_gatebuf, tmp);
            rms_norm(xb, tmp, l->refresh_out_norm, hidden);
            for (uint32_t i = 0; i < hidden; ++i) x[i] += l->refresh_alpha[0] * xb[i];
        }
        rms_norm(xb, x, l->ln2, hidden);
        matvec(&l->gate, xb, gatebuf); matvec(&l->up, xb, upbuf);
        for (uint32_t i = 0; i < intermediate; ++i) gatebuf[i] = gatebuf[i] / (1.0f + fast_exp(-gatebuf[i])) * upbuf[i];
        matvec(&l->down, gatebuf, tmp);
        for (uint32_t i = 0; i < hidden; ++i) x[i] += tmp[i];
    }
    rms_norm(xb, x, net.final_norm, hidden);
    matvec(&net.embedding, xb, logits);
    uint16_t best = 0;
    for (uint32_t i = 1; i < h->vocab_size; ++i) if (logits[i] > logits[best]) best = (uint16_t)i;
    return best;
}

static const struct merge_slot *find_merge(uint16_t left, uint16_t right) {
    uint32_t key = (((uint32_t)left << 16) | right) + 1u;
    uint32_t slot = merge_hash(key);
    while (merge_table[slot].key) {
        if (merge_table[slot].key == key) return &merge_table[slot];
        slot = (slot + 1) & (MERGE_SLOTS - 1);
    }
    return 0;
}

static uint32_t tokenize(const uint8_t *input, uint32_t length, uint16_t *output, uint32_t capacity) {
    if (length > MAX_BPE_TOKENS) length = MAX_BPE_TOKENS;
    for (uint32_t i = 0; i < length; ++i) bpe_tokens[i] = byte_token[input[i]];
    uint32_t count = length;
    for (;;) {
        uint32_t best_pos = count, best_rank = 0xFFFFFFFFu;
        uint16_t best_result = 0;
        for (uint32_t i = 0; i + 1 < count; ++i) {
            const struct merge_slot *m = find_merge(bpe_tokens[i], bpe_tokens[i + 1]);
            if (m && (uint32_t)(m->rank_plus_one - 1u) < best_rank) {
                best_rank = m->rank_plus_one - 1u; best_pos = i; best_result = m->result;
            }
        }
        if (best_pos == count) break;
        bpe_tokens[best_pos] = best_result;
        for (uint32_t i = best_pos + 1; i + 1 < count; ++i) bpe_tokens[i] = bpe_tokens[i + 1];
        --count;
    }
    uint32_t written = 0;
    if (written < capacity) output[written++] = (uint16_t)net.header->bos_id;
    uint32_t start = count > capacity - written ? count - (capacity - written) : 0;
    while (start < count && written < capacity) output[written++] = bpe_tokens[start++];
    return written;
}

static void print_token(uint16_t id) {
    if (id == net.header->bos_id || id == net.header->eos_id || id == net.header->pad_id) return;
    const struct bm2n_token *token = &net.tokens[id];
    for (uint32_t i = 0; i < token->length; ++i) {
        uint8_t c = net.token_data[token->offset + i];
        if (c == '\n' || c == '\t' || c >= 32) putc((char)c);
    }
}

static char keyboard_char(void) {
    static const char normal[] = "\0\0331234567890-=\b\tqwertyuiop[]\n\0asdfghjkl;'`\0\\zxcvbnm,./\0*\0 ";
    static const char shifted[] = "\0\033!@#$%^&*()_+\b\tQWERTYUIOP{}\n\0ASDFGHJKL:\"~\0|ZXCVBNM<>?\0*\0 ";
    static int shift;
    if (!(inb(0x64) & 1u)) return 0;
    uint8_t code = inb(0x60);
    if (code == 42 || code == 54) { shift = 1; return 0; }
    if (code == 170 || code == 182) { shift = 0; return 0; }
    if (code & 0x80u || code >= sizeof(normal) - 1) return 0;
    return shift ? shifted[code] : normal[code];
}

static char getc(void) {
    for (;;) {
        if (inb(0x3F8 + 5) & 1u) return (char)inb(0x3F8);
        char c = keyboard_char();
        if (c) return c;
    }
}

static uint8_t cmos_read(uint8_t index) {
    outb(0x70, (uint8_t)(0x80u | index));
    return inb(0x71);
}

static uint8_t from_bcd(uint8_t value) {
    return (uint8_t)((value & 15u) + (value >> 4) * 10u);
}

static uint32_t rtc_seconds(void) {
    uint8_t second, second_check, minute, hour, mode;
    do {
        while (cmos_read(0x0A) & 0x80u) { }
        second = cmos_read(0x00); minute = cmos_read(0x02); hour = cmos_read(0x04);
        second_check = cmos_read(0x00);
    } while (second != second_check);
    mode = cmos_read(0x0B);
    uint8_t pm = hour & 0x80u;
    hour &= 0x7Fu;
    if (!(mode & 0x04u)) { second = from_bcd(second); minute = from_bcd(minute); hour = from_bcd(hour); }
    if (!(mode & 0x02u)) {
        if (pm && hour != 12u) hour = (uint8_t)(hour + 12u);
        if (!pm && hour == 12u) hour = 0;
    }
    return (uint32_t)hour * 3600u + (uint32_t)minute * 60u + second;
}

static uint32_t readline(uint8_t *buffer, uint32_t capacity) {
    uint32_t length = 0;
    for (;;) {
        char c = getc();
        if (c == '\r') c = '\n';
        if (c == '\n') { putc('\n'); return length; }
        if ((c == '\b' || c == 127) && length) { --length; putc('\b'); continue; }
        if ((uint8_t)c >= 32 && length + 1 < capacity) { buffer[length++] = (uint8_t)c; putc(c); }
    }
}

static void clear(void) {
    if (gui_enabled) { gui_draw_shell(); gui_show_tps(0xFFFFFFFFu); return; }
    for (uint32_t i = 0; i < VGA_WIDTH * VGA_HEIGHT; ++i) vga[i] = (uint16_t)color << 8 | ' ';
    cursor_row = cursor_col = 0;
}

static void reboot(void) {
    while (inb(0x64) & 2u) { }
    outb(0x64, 0xFE);
    for (;;) __asm__ volatile ("hlt");
}

static void return_after_ram_error(const char *detail) {
    if (gui_enabled) gui_output_begin();
    color = 0x4F;
    puts("NOT ENOUGH RAM\n\n");
    puts(detail);
    puts("\n\n[ ENTER ]  OK - return to model selection");
    for (;;) { char choice = getc(); if (choice == '\n' || choice == '\r') break; }
    reboot();
}

static void low_ram_warning(uint32_t installed, uint32_t recommended) {
    if (gui_enabled) gui_output_begin();
    color = 0x60;
    puts("LOW MEMORY WARNING\n\nInstalled: "); putu(installed);
    puts(" MB    Recommended: "); putu(recommended); puts(" MB\n");
    puts("Generation may fail or have a shorter safe margin.\n\n");
    puts("[ C ] Continue     [ R ] Return to selection");
    for (;;) {
        char choice = getc();
        if (choice == 'c' || choice == 'C') return;
        if (choice == 'r' || choice == 'R') reboot();
    }
}

static void fatal(const char *message) {
    color = 0x0C; puts("\nERROR: "); puts(message); puts("\nSystem halted.\n");
    for (;;) __asm__ volatile ("hlt");
}

void kernel_main(uint32_t magic, const struct multiboot_info *mb) {
    serial_init(); fpu_init();
    if (magic == MULTIBOOT_BOOTLOADER_MAGIC) gui_init(mb);
    clear();
    puts("============================================================\n");
    puts("                    BananaMind OS\n");
    puts("          local intelligence for the Intel 486\n");
    puts("============================================================\n\n");
    if (magic != MULTIBOOT_BOOTLOADER_MAGIC) fatal("not loaded by a Multiboot bootloader");
    if (!(mb->flags & MULTIBOOT_INFO_MEMORY)) fatal("bootloader did not provide a memory size");
    if (!(mb->flags & MULTIBOOT_INFO_MODS) || mb->mods_count != 1) fatal("select one model module in the boot menu");
    const struct multiboot_module *module = (const struct multiboot_module *)mb->mods_addr;
    chat_mode = module->string && begins_with_chat((const char *)module->string);
    if (!load_model((const uint8_t *)module->start, module->end - module->start)) fatal("invalid or incompatible BM2NQ model");

    uint32_t total_kb = mb->mem_upper + 1024u;
    uint32_t total_mb = (total_kb + 1023u) / 1024u;
    uint32_t needed_mb;
    if (net.kind == MODEL_MICRO2)
        needed_mb = net.header->bits == 4u ? 4u : (net.header->bits == 8u ? 5u : 8u);
    else if (net.kind == MODEL_MICRO_V1)
        needed_mb = net.header->bits == 16u ? 4u : 6u;
    else if (net.header->hidden_size == 384u)
        needed_mb = net.header->bits == 2u ? 12u : 20u;
    else needed_mb = net.header->bits == 8u ? 14u : (net.header->bits == 4u ? 8u : 6u);
    puts("Model: ");
    if (net.kind == MODEL_MICRO2) puts("BananaMind-2-Micro");
    else if (net.kind == MODEL_MICRO_V1) puts("MicroBananaMind-v1");
    else if (net.header->hidden_size == 384u) puts("BananaMind-2-Mini");
    else { puts("BananaMind-2-Nano"); if (chat_mode) puts("-Chat"); }
    puts(net.header->bits <= 8u ? " Q" : " F"); putu(net.header->bits);
    puts(" | RAM: "); putu(total_mb); puts(" MB | context: 64\n");
    /* PC firmware reserves a small EBDA/ROM window; allow the usual 256 KiB. */
    if (total_kb + 256u < needed_mb * 1024u) {
        uint32_t shortfall = needed_mb * 1024u - (total_kb + 256u);
        if (shortfall <= 2048u) low_ram_warning(total_mb, needed_mb);
        else return_after_ram_error("This model is far below its minimum memory requirement.");
    }
    uintptr_t arena = (module->end + 15u) & ~15u;
    uint32_t cache_floats = net.header->num_layers * BM2N_CONTEXT *
        net.header->num_kv_heads * net.header->head_dim;
    uint32_t refresh_floats = net.kind == MODEL_MICRO2 ?
        net.header->num_layers * net.header->hidden_size * (REFRESH_KERNEL - 1u) : 0u;
    uintptr_t arena_end = arena + (uintptr_t)(cache_floats * 2u + refresh_floats) * sizeof(float);
    if (arena_end > (uintptr_t)(mb->mem_upper + 1024u) * 1024u)
        return_after_ram_error("There is no safe contiguous space for the model KV cache.");
    key_cache = (float *)arena;
    value_cache = key_cache + cache_floats;
    refresh_history = value_cache + cache_floats;
    color = 0x0A; puts("Model ready. Enter a prompt; BananaMind streams exactly 16 tokens.\n");
    color = 0x0F;
    static uint8_t input[MAX_INPUT_BYTES];
    for (;;) {
        if (gui_enabled) gui_input_begin();
        puts(gui_enabled ? "" : "\nbanana> ");
        uint32_t input_length = readline(input, sizeof(input));
        if (!input_length) continue;
        const uint8_t *model_input = input;
        uint32_t model_input_length = input_length;
        if (chat_mode) { model_input_length = chat_wrap(input, input_length); model_input = chat_input; }
        for (uint32_t i = 0; i < refresh_floats; ++i) refresh_history[i] = 0.0f;
        uint32_t count = tokenize(model_input, model_input_length, prompt_tokens, BM2N_CONTEXT - 16u);
        if (gui_enabled) gui_output_begin();
        puts(gui_enabled ? "Generating...\n\n" : "\n");
        uint16_t next = 0;
        for (uint32_t pos = 0; pos < count; ++pos) next = forward(prompt_tokens[pos], pos);
        if (gui_enabled) gui_output_begin();
        gui_show_tps(0xFFFFFFFFu);
        uint32_t generation_start = rtc_seconds();
        uint32_t generated_count = 0;
        for (uint32_t generated = 0; generated < 16u; ++generated) {
            uint32_t pos = count + generated;
            if (generated) next = forward(next, pos - 1u);
            print_token(next);
            ++generated_count;
            uint32_t now = rtc_seconds();
            uint32_t elapsed = now >= generation_start ? now - generation_start : now + 86400u - generation_start;
            if (elapsed) gui_show_tps(generated_count * 10u / elapsed);
            if (next == net.header->eos_id) break;
        }
        uint32_t generation_end = rtc_seconds();
        uint32_t elapsed = generation_end >= generation_start ? generation_end - generation_start
                                                               : generation_end + 86400u - generation_start;
        uint32_t tps_tenths = elapsed ? generated_count * 10u / elapsed : 999u;
        gui_show_tps(tps_tenths);
        puts("\n\nTPS: "); putu(tps_tenths / 10u); putc('.'); putu(tps_tenths % 10u); puts(" tokens/s\n");
    }
}
