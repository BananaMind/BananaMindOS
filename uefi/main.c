#include <efi.h>
#include <efilib.h>
#include <efipoint.h>
#include <stdint.h>
#include <stddef.h>

#include "font8x8.h"
#include "litemodel.h"
#include "litemodel_runtime.h"
#include "mouse.h"

#define MAX_MODELS 48u
#define CATALOG_BYTES 16384u
#define INPUT_BYTES 512u
#define CHAT_HISTORY_BYTES 4096u
#define CURSOR_W 12u
#define CURSOR_H 18u
#define MAX_POINTER_PROTOCOLS 8u

#define UI_DARK     0x111318u
#define UI_SURFACE  0x1B1F27u
#define UI_HOVER    0x272D38u
#define UI_BORDER   0x39414Fu
#define UI_WHITE    0xF2F4F7u
#define UI_MUTED    0x9AA4B2u
#define UI_YELLOW   0xFFD400u
#define UI_INK      0x111318u
#define UI_GREEN    0x52D273u
#define UI_RED      0xFF6577u

struct catalog_entry {
    char id[32];
    char filename[64];
    char name[64];
    char variant[16];
    char description[112];
    uint32_t ram_mb;
    uint8_t chat;
    uint8_t legacy;
};

static EFI_HANDLE image_handle;
static EFI_SYSTEM_TABLE *system_table;
static EFI_BOOT_SERVICES *boot_services;
static EFI_GRAPHICS_OUTPUT_PROTOCOL *gop;
static EFI_SIMPLE_POINTER_PROTOCOL *pointers[MAX_POINTER_PROTOCOLS];
static EFI_ABSOLUTE_POINTER_PROTOCOL *absolute_pointers[MAX_POINTER_PROTOCOLS];
static uint32_t pointer_count, absolute_pointer_count;
static EFI_FILE_HANDLE volume_root;
static uint32_t *framebuffer;
static uint32_t screen_width, screen_height, screen_stride;
static EFI_GRAPHICS_PIXEL_FORMAT pixel_format;
static EFI_PIXEL_BITMASK pixel_masks;
static struct catalog_entry models[MAX_MODELS];
static uint32_t model_count;
static char catalog[CATALOG_BYTES + 1u];
static struct lm_runtime runtime;
static void *model_memory;
static void *arena_memory;
static struct catalog_entry *active_model;
static uint8_t active_chat;
static uint8_t multi_turn_enabled = 1u;
static uint8_t kv_cache_enabled = 1u;
static uint32_t context_tokens_setting;
static uint32_t max_generation_tokens = 16u;
static uint32_t temperature_tenths;
static uint32_t random_state = 0x9E3779B9u;
static uint8_t chat_history[CHAT_HISTORY_BYTES];
static uint32_t chat_history_length;
static int32_t pointer_x, pointer_y;
static uint8_t pointer_down, cursor_saved;
static struct mouse_state ps2_mouse;
static uint8_t ps2_mouse_ready;
static uint32_t cursor_under[CURSOR_W * CURSOR_H];
static uint32_t output_x, output_y, output_left, output_top, output_right, output_bottom;

static EFI_GUID loaded_image_guid = EFI_LOADED_IMAGE_PROTOCOL_GUID;
static EFI_GUID filesystem_guid = EFI_SIMPLE_FILE_SYSTEM_PROTOCOL_GUID;
static EFI_GUID file_info_guid = EFI_FILE_INFO_ID;
static EFI_GUID gop_guid = EFI_GRAPHICS_OUTPUT_PROTOCOL_GUID;
static EFI_GUID pointer_guid = EFI_SIMPLE_POINTER_PROTOCOL_GUID;
static EFI_GUID absolute_pointer_guid = EFI_ABSOLUTE_POINTER_PROTOCOL_GUID;

static void serial_char(char value) {
    uint8_t ready;
    do __asm__ volatile ("inb %1, %0" : "=a"(ready) : "Nd"((uint16_t)0x3FD));
    while (!(ready & 0x20u));
    __asm__ volatile ("outb %0, %1" : : "a"((uint8_t)value), "Nd"((uint16_t)0x3F8));
}

static void serial_text(const char *text) {
    while (*text) {
        if (*text == '\n') serial_char('\r');
        serial_char(*text++);
    }
}

static void serial_u64(uint64_t value) {
    char digits[21];
    uint32_t count = 0u;
    if (!value) { serial_char('0'); return; }
    while (value) { digits[count++] = (char)('0' + value % 10u); value /= 10u; }
    while (count) serial_char(digits[--count]);
}

static uint32_t text_length(const char *text) {
    uint32_t length = 0u;
    while (text[length]) ++length;
    return length;
}

static int text_equal(const char *left, const char *right) {
    while (*left && *left == *right) { ++left; ++right; }
    return *left == *right;
}

static void copy_field(char *destination, uint32_t capacity,
                       const char *begin, const char *end) {
    uint32_t written = 0u;
    while (begin < end && written + 1u < capacity) destination[written++] = *begin++;
    destination[written] = 0;
}

static uint32_t parse_number(const char *text) {
    uint32_t value = 0u;
    while (*text >= '0' && *text <= '9') value = value * 10u + (uint32_t)(*text++ - '0');
    return value;
}

static void number_text(uint32_t value, char output[12]) {
    char reverse[11];
    uint32_t count = 0u, written = 0u;
    if (!value) reverse[count++] = '0';
    while (value) { reverse[count++] = (char)('0' + value % 10u); value /= 10u; }
    while (count) output[written++] = reverse[--count];
    output[written] = 0;
}

static uint32_t mask_shift(uint32_t mask) {
    uint32_t shift = 0u;
    if (!mask) return 0u;
    while (!(mask & 1u)) { mask >>= 1; ++shift; }
    return shift;
}

static uint32_t mask_bits(uint32_t mask) {
    uint32_t bits = 0u;
    while (mask) { bits += mask & 1u; mask >>= 1; }
    return bits;
}

static uint32_t channel_to_mask(uint32_t channel, uint32_t mask) {
    uint32_t bits = mask_bits(mask);
    if (!bits) return 0u;
    uint32_t maximum = bits >= 31u ? 0xFFFFFFFFu : ((1u << bits) - 1u);
    return ((channel * maximum + 127u) / 255u << mask_shift(mask)) & mask;
}

static uint32_t native_color(uint32_t rgb) {
    uint32_t red = (rgb >> 16) & 255u;
    uint32_t green = (rgb >> 8) & 255u;
    uint32_t blue = rgb & 255u;
    if (pixel_format == PixelBlueGreenRedReserved8BitPerColor) return rgb;
    if (pixel_format == PixelRedGreenBlueReserved8BitPerColor)
        return blue << 16 | green << 8 | red;
    return channel_to_mask(red, pixel_masks.RedMask) |
           channel_to_mask(green, pixel_masks.GreenMask) |
           channel_to_mask(blue, pixel_masks.BlueMask);
}

static void fill_rect(uint32_t left, uint32_t top, uint32_t width,
                      uint32_t height, uint32_t color) {
    if (left >= screen_width || top >= screen_height) return;
    if (width > screen_width - left) width = screen_width - left;
    if (height > screen_height - top) height = screen_height - top;
    uint32_t native = native_color(color);
    for (uint32_t y = top; y < top + height; ++y) {
        uint32_t *row = framebuffer + y * screen_stride + left;
        for (uint32_t x = 0; x < width; ++x) row[x] = native;
    }
}

static void draw_glyph(uint32_t left, uint32_t top, char character,
                       uint32_t foreground, uint32_t background) {
    uint8_t code = (uint8_t)character;
    if (code < 32u || code > 127u) code = '?';
    const uint8_t *glyph = font8x8[code - 32u];
    for (uint32_t row = 0; row < 8u; ++row)
        for (uint32_t column = 0; column < 8u; ++column)
            fill_rect(left + column, top + row, 1u, 1u,
                      glyph[row] & (0x80u >> column) ? foreground : background);
}

static void draw_text(uint32_t left, uint32_t top, const char *text,
                      uint32_t foreground, uint32_t background) {
    while (*text && left + 8u <= screen_width) {
        draw_glyph(left, top, *text++, foreground, background);
        left += 8u;
    }
}

static void cursor_restore(void) {
    if (!cursor_saved) return;
    for (uint32_t y = 0; y < CURSOR_H; ++y)
        for (uint32_t x = 0; x < CURSOR_W; ++x) {
            uint32_t px = (uint32_t)pointer_x + x, py = (uint32_t)pointer_y + y;
            if (px < screen_width && py < screen_height)
                framebuffer[py * screen_stride + px] = cursor_under[y * CURSOR_W + x];
        }
    cursor_saved = 0u;
}

static void cursor_draw(void) {
    cursor_restore();
    uint32_t white = native_color(UI_WHITE), dark = native_color(UI_INK);
    for (uint32_t y = 0; y < CURSOR_H; ++y)
        for (uint32_t x = 0; x < CURSOR_W; ++x) {
            uint32_t px = (uint32_t)pointer_x + x, py = (uint32_t)pointer_y + y;
            if (px >= screen_width || py >= screen_height) continue;
            uint32_t *pixel = framebuffer + py * screen_stride + px;
            cursor_under[y * CURSOR_W + x] = *pixel;
            if (x == 0u || y == x * 2u || (x == 1u && y > 1u) ||
                (y == 16u && x < 7u)) *pixel = dark;
            else if (x < 7u && y > x * 2u && y < 16u) *pixel = white;
        }
    cursor_saved = 1u;
}

static uint8_t inside(int32_t x, int32_t y, uint32_t left, uint32_t top,
                      uint32_t width, uint32_t height) {
    return x >= (int32_t)left && y >= (int32_t)top &&
           x < (int32_t)(left + width) && y < (int32_t)(top + height);
}

static void poll_pointer(void) {
    static uint8_t reported_absolute_event, reported_relative_event;
    if (ps2_mouse_ready) {
        mouse_poll(&ps2_mouse, screen_width, screen_height);
        if (ps2_mouse.changed) {
            cursor_restore();
            pointer_x = ps2_mouse.x; pointer_y = ps2_mouse.y;
            pointer_down = ps2_mouse.buttons & 1u ? 1u : 0u;
            cursor_draw();
            if (!reported_relative_event) {
                serial_text("BananaMind UEFI: direct PS/2 pointer events active\n");
                reported_relative_event = 1u;
            }
            return;
        }
    }
    for (uint32_t device = 0u; device < absolute_pointer_count; ++device) {
        EFI_ABSOLUTE_POINTER_PROTOCOL *absolute_pointer = absolute_pointers[device];
        EFI_ABSOLUTE_POINTER_STATE state;
        EFI_STATUS status = uefi_call_wrapper(absolute_pointer->GetState, 2,
                                              absolute_pointer, &state);
        if (!EFI_ERROR(status)) {
            EFI_ABSOLUTE_POINTER_MODE *mode = absolute_pointer->Mode;
            uint64_t range_x = mode->AbsoluteMaxX - mode->AbsoluteMinX;
            uint64_t range_y = mode->AbsoluteMaxY - mode->AbsoluteMinY;
            int32_t next_x = range_x ? (int32_t)((state.CurrentX - mode->AbsoluteMinX) *
                                      (screen_width - 1u) / range_x) : pointer_x;
            int32_t next_y = range_y ? (int32_t)((state.CurrentY - mode->AbsoluteMinY) *
                                      (screen_height - 1u) / range_y) : pointer_y;
            if (next_x != pointer_x || next_y != pointer_y) {
                cursor_restore(); pointer_x = next_x; pointer_y = next_y; cursor_draw();
                if (!reported_absolute_event) {
                    serial_text("BananaMind UEFI: absolute pointer events active\n");
                    reported_absolute_event = 1u;
                }
            }
            pointer_down = state.ActiveButtons & EFI_ABSP_TouchActive ? 1u : 0u;
            return;
        }
    }
    for (uint32_t device = 0u; device < pointer_count; ++device) {
        EFI_SIMPLE_POINTER_PROTOCOL *pointer = pointers[device];
        EFI_SIMPLE_POINTER_STATE state;
        EFI_STATUS status = uefi_call_wrapper(pointer->GetState, 2, pointer, &state);
        if (EFI_ERROR(status)) continue;
        int64_t dx = state.RelativeMovementX;
        int64_t dy = state.RelativeMovementY;
        uint64_t resolution_x = pointer->Mode->ResolutionX ? pointer->Mode->ResolutionX : 1u;
        uint64_t resolution_y = pointer->Mode->ResolutionY ? pointer->Mode->ResolutionY : 1u;
        int32_t move_x = (int32_t)(dx * (int64_t)screen_width * 4 / (int64_t)resolution_x);
        int32_t move_y = (int32_t)(dy * (int64_t)screen_height * 4 / (int64_t)resolution_y);
        if (dx && !move_x) move_x = dx > 0 ? 1 : -1;
        if (dy && !move_y) move_y = dy > 0 ? 1 : -1;
        if (move_x || move_y) {
            cursor_restore();
            pointer_x += move_x; pointer_y += move_y;
            if (pointer_x < 0) pointer_x = 0;
            if (pointer_y < 0) pointer_y = 0;
            if (pointer_x > (int32_t)screen_width - 1) pointer_x = (int32_t)screen_width - 1;
            if (pointer_y > (int32_t)screen_height - 1) pointer_y = (int32_t)screen_height - 1;
            cursor_draw();
            if (!reported_relative_event) {
                serial_text("BananaMind UEFI: relative pointer events active\n");
                reported_relative_event = 1u;
            }
        }
        pointer_down = state.LeftButton ? 1u : 0u;
        return;
    }
}

static void discover_pointers(void) {
    EFI_HANDLE *handles = 0;
    UINTN count = 0u;
    if (!EFI_ERROR(uefi_call_wrapper(boot_services->LocateHandleBuffer, 5, ByProtocol,
                                     &absolute_pointer_guid, 0, &count, &handles))) {
        for (UINTN index = 0u; index < count && absolute_pointer_count < MAX_POINTER_PROTOCOLS; ++index) {
            EFI_ABSOLUTE_POINTER_PROTOCOL *protocol = 0;
            if (!EFI_ERROR(uefi_call_wrapper(boot_services->HandleProtocol, 3, handles[index],
                                             &absolute_pointer_guid, (void **)&protocol)) && protocol) {
                absolute_pointers[absolute_pointer_count++] = protocol;
                uefi_call_wrapper(protocol->Reset, 2, protocol, FALSE);
            }
        }
        uefi_call_wrapper(boot_services->FreePool, 1, handles);
    }
    handles = 0; count = 0u;
    if (!EFI_ERROR(uefi_call_wrapper(boot_services->LocateHandleBuffer, 5, ByProtocol,
                                     &pointer_guid, 0, &count, &handles))) {
        for (UINTN index = 0u; index < count && pointer_count < MAX_POINTER_PROTOCOLS; ++index) {
            EFI_SIMPLE_POINTER_PROTOCOL *protocol = 0;
            if (!EFI_ERROR(uefi_call_wrapper(boot_services->HandleProtocol, 3, handles[index],
                                             &pointer_guid, (void **)&protocol)) && protocol) {
                pointers[pointer_count++] = protocol;
                uefi_call_wrapper(protocol->Reset, 2, protocol, FALSE);
            }
        }
        uefi_call_wrapper(boot_services->FreePool, 1, handles);
    }
    serial_text("BananaMind UEFI: pointer handles absolute=");
    serial_u64(absolute_pointer_count); serial_text(" relative=");
    serial_u64(pointer_count); serial_char('\n');
}

static int read_key(EFI_INPUT_KEY *key) {
    EFI_STATUS status = uefi_call_wrapper(system_table->ConIn->ReadKeyStroke, 2,
                                          system_table->ConIn, key);
    return !EFI_ERROR(status);
}

static void idle(void) {
    uefi_call_wrapper(boot_services->Stall, 1, 1000u);
}

static int make_path(const char *filename, CHAR16 path[96]) {
    static const char prefix[] = "\\MODELS\\";
    uint32_t written = 0u;
    for (uint32_t index = 0; prefix[index]; ++index) path[written++] = (CHAR16)prefix[index];
    for (uint32_t index = 0; filename[index] && written + 1u < 96u; ++index)
        path[written++] = (CHAR16)(uint8_t)filename[index];
    path[written] = 0;
    return filename[0] && written + 1u < 96u;
}

static EFI_STATUS open_file(EFI_FILE_HANDLE root, CHAR16 *path, EFI_FILE_HANDLE *file) {
    return uefi_call_wrapper(root->Open, 5, root, file, path, EFI_FILE_MODE_READ, 0u);
}

static EFI_STATUS file_size(EFI_FILE_HANDLE file, UINT64 *size) {
    uint8_t info_buffer[512];
    UINTN info_size = sizeof(info_buffer);
    EFI_STATUS status = uefi_call_wrapper(file->GetInfo, 4, file, &file_info_guid,
                                          &info_size, info_buffer);
    if (!EFI_ERROR(status)) *size = ((EFI_FILE_INFO *)info_buffer)->FileSize;
    return status;
}

static int read_catalog_from(EFI_FILE_HANDLE root) {
    EFI_FILE_HANDLE file;
    CHAR16 path[] = L"\\MODELS\\CATALOG.CFG";
    if (EFI_ERROR(open_file(root, path, &file))) return 0;
    UINT64 size64 = 0u;
    if (EFI_ERROR(file_size(file, &size64)) || !size64 || size64 > CATALOG_BYTES) {
        uefi_call_wrapper(file->Close, 1, file); return 0;
    }
    UINTN size = (UINTN)size64;
    EFI_STATUS status = uefi_call_wrapper(file->Read, 3, file, &size, catalog);
    uefi_call_wrapper(file->Close, 1, file);
    if (EFI_ERROR(status) || size != (UINTN)size64) return 0;
    catalog[size] = 0;
    return 1;
}

static int locate_model_volume(void) {
    EFI_LOADED_IMAGE_PROTOCOL *loaded = 0;
    EFI_SIMPLE_FILE_SYSTEM_PROTOCOL *filesystem = 0;
    if (!EFI_ERROR(uefi_call_wrapper(boot_services->HandleProtocol, 3, image_handle,
                                     &loaded_image_guid, (void **)&loaded)) && loaded &&
        !EFI_ERROR(uefi_call_wrapper(boot_services->HandleProtocol, 3, loaded->DeviceHandle,
                                     &filesystem_guid, (void **)&filesystem)) &&
        !EFI_ERROR(uefi_call_wrapper(filesystem->OpenVolume, 2, filesystem, &volume_root)) &&
        read_catalog_from(volume_root)) return 1;

    EFI_HANDLE *handles = 0;
    UINTN count = 0u;
    if (EFI_ERROR(uefi_call_wrapper(boot_services->LocateHandleBuffer, 5, ByProtocol,
                                    &filesystem_guid, 0, &count, &handles))) return 0;
    for (UINTN index = 0; index < count; ++index) {
        EFI_FILE_HANDLE root = 0;
        filesystem = 0;
        if (EFI_ERROR(uefi_call_wrapper(boot_services->HandleProtocol, 3, handles[index],
                                        &filesystem_guid, (void **)&filesystem)) ||
            EFI_ERROR(uefi_call_wrapper(filesystem->OpenVolume, 2, filesystem, &root))) continue;
        if (read_catalog_from(root)) { volume_root = root; break; }
        uefi_call_wrapper(root->Close, 1, root);
    }
    uefi_call_wrapper(boot_services->FreePool, 1, handles);
    return volume_root != 0;
}

static uint32_t split_fields(char *line, char *fields[8]) {
    uint32_t count = 0u;
    fields[count++] = line;
    while (*line && count < 8u) {
        if (*line == '|') { *line = 0; fields[count++] = line + 1; }
        ++line;
    }
    return count;
}

static int parse_catalog(void) {
    char *line = catalog;
    while (*line && model_count < MAX_MODELS) {
        char *end = line;
        while (*end && *end != '\n' && *end != '\r') ++end;
        char saved = *end; *end = 0;
        if (*line && *line != '#') {
            char *fields[8];
            if (split_fields(line, fields) == 8u) {
                struct catalog_entry *entry = &models[model_count++];
                copy_field(entry->id, sizeof(entry->id), fields[0], fields[0] + text_length(fields[0]));
                copy_field(entry->filename, sizeof(entry->filename), fields[1], fields[1] + text_length(fields[1]));
                copy_field(entry->name, sizeof(entry->name), fields[2], fields[2] + text_length(fields[2]));
                copy_field(entry->variant, sizeof(entry->variant), fields[3], fields[3] + text_length(fields[3]));
                entry->ram_mb = parse_number(fields[4]);
                entry->chat = text_equal(fields[5], "chat") ? 1u : 0u;
                entry->legacy = text_equal(fields[6], "legacy") ? 1u : 0u;
                copy_field(entry->description, sizeof(entry->description), fields[7], fields[7] + text_length(fields[7]));
            }
        }
        *end = saved;
        while (*end == '\n' || *end == '\r') ++end;
        line = end;
    }
    return model_count != 0u;
}

static int initialize_graphics(void) {
    if (EFI_ERROR(uefi_call_wrapper(boot_services->LocateProtocol, 3, &gop_guid, 0,
                                    (void **)&gop)) || !gop || !gop->Mode) return 0;
    uint32_t selected = gop->Mode->Mode;
    uint64_t selected_area = 0u;
    for (uint32_t mode = 0; mode < gop->Mode->MaxMode; ++mode) {
        EFI_GRAPHICS_OUTPUT_MODE_INFORMATION *info = 0;
        UINTN info_size = 0u;
        if (EFI_ERROR(uefi_call_wrapper(gop->QueryMode, 4, gop, mode, &info_size, &info))) continue;
        uint64_t area = (uint64_t)info->HorizontalResolution * info->VerticalResolution;
        if (info->HorizontalResolution >= 800u && info->VerticalResolution >= 600u &&
            info->HorizontalResolution <= 1600u && info->VerticalResolution <= 1000u &&
            info->PixelFormat != PixelBltOnly && area > selected_area) {
            selected = mode; selected_area = area;
        }
        uefi_call_wrapper(boot_services->FreePool, 1, info);
    }
    if (selected != gop->Mode->Mode &&
        EFI_ERROR(uefi_call_wrapper(gop->SetMode, 2, gop, selected))) return 0;
    if (!gop->Mode->Info || gop->Mode->Info->PixelFormat == PixelBltOnly) return 0;
    screen_width = gop->Mode->Info->HorizontalResolution;
    screen_height = gop->Mode->Info->VerticalResolution;
    screen_stride = gop->Mode->Info->PixelsPerScanLine;
    pixel_format = gop->Mode->Info->PixelFormat;
    pixel_masks = gop->Mode->Info->PixelInformation;
    framebuffer = (uint32_t *)(uintptr_t)gop->Mode->FrameBufferBase;
    pointer_x = (int32_t)(screen_width / 2u);
    pointer_y = (int32_t)(screen_height / 2u);
    discover_pointers();
    ps2_mouse_ready = (uint8_t)mouse_init(&ps2_mouse, screen_width, screen_height);
    if (ps2_mouse_ready) serial_text("BananaMind UEFI: direct PS/2 fallback ready\n");
    if (!absolute_pointer_count && !pointer_count)
        serial_text("BananaMind UEFI: no firmware pointer; keyboard navigation active\n");
    return screen_width >= 800u && screen_height >= 600u;
}

static uint32_t total_memory_mb(void) {
    static EFI_MEMORY_DESCRIPTOR map[1024];
    UINTN size = sizeof(map), key = 0u, descriptor_size = 0u;
    UINT32 version = 0u;
    if (EFI_ERROR(uefi_call_wrapper(boot_services->GetMemoryMap, 5, &size, map, &key,
                                    &descriptor_size, &version)) || !descriptor_size) return 0u;
    uint64_t pages = 0u;
    for (UINTN offset = 0; offset + descriptor_size <= size; offset += descriptor_size) {
        EFI_MEMORY_DESCRIPTOR *entry = (EFI_MEMORY_DESCRIPTOR *)((uint8_t *)map + offset);
        if (entry->Type != EfiReservedMemoryType && entry->Type != EfiMemoryMappedIO &&
            entry->Type != EfiMemoryMappedIOPortSpace && entry->Type != EfiUnusableMemory)
            pages += entry->NumberOfPages;
    }
    return (uint32_t)(pages / 256u);
}

static uint32_t list_width(void) { return screen_width * 55u / 100u; }
static uint32_t visible_cards(void) {
    uint32_t visible = (screen_height - 160u) / 54u;
    return visible ? visible : 1u;
}

static void draw_card(uint32_t index, uint32_t selected, uint32_t first) {
    if (index < first || index >= first + visible_cards()) return;
    uint32_t top = 76u + (index - first) * 54u;
    uint32_t background = index == selected ? UI_HOVER : UI_SURFACE;
    fill_rect(20u, top, list_width(), 48u, background);
    if (index == selected) fill_rect(20u, top, 4u, 48u, UI_YELLOW);
    draw_text(36u, top + 10u, models[index].name, UI_WHITE, background);
    draw_text(36u, top + 28u, models[index].variant, UI_YELLOW, background);
    char ram[12]; number_text(models[index].ram_mb, ram);
    uint32_t right = 20u + list_width() - (text_length(ram) + 7u) * 8u;
    draw_text(right, top + 28u, ram, UI_MUTED, background);
    draw_text(right + text_length(ram) * 8u, top + 28u, " MB RAM", UI_MUTED, background);
}

static void draw_details(uint32_t selected, uint32_t memory_mb) {
    uint32_t left = 42u + list_width();
    uint32_t width = screen_width - left - 20u;
    struct catalog_entry *entry = &models[selected];
    fill_rect(left, 76u, width, screen_height - 152u, UI_SURFACE);
    fill_rect(left, 76u, width, 3u, UI_YELLOW);
    draw_text(left + 16u, 94u, "SELECTED MODEL", UI_MUTED, UI_SURFACE);
    draw_text(left + 16u, 118u, entry->name, UI_WHITE, UI_SURFACE);
    draw_text(left + 16u, 142u, entry->variant, UI_YELLOW, UI_SURFACE);
    draw_text(left + 16u, 172u, entry->description, UI_MUTED, UI_SURFACE);
    draw_text(left + 16u, 212u, entry->chat ? "CHAT MODEL" : "BASE MODEL", UI_WHITE, UI_SURFACE);
    draw_text(left + 16u, 238u, "UEFI X64", UI_MUTED, UI_SURFACE);
    char available[12], required[12];
    number_text(memory_mb, available); number_text(entry->ram_mb, required);
    draw_text(left + 16u, 278u, "SYSTEM RAM", UI_MUTED, UI_SURFACE);
    draw_text(left + 16u, 300u, available,
              memory_mb >= entry->ram_mb ? UI_GREEN : UI_RED, UI_SURFACE);
    draw_text(left + 16u + text_length(available) * 8u, 300u, " MB", UI_WHITE, UI_SURFACE);
    draw_text(left + 16u, 328u, "MODEL NEEDS", UI_MUTED, UI_SURFACE);
    draw_text(left + 16u, 350u, required, UI_WHITE, UI_SURFACE);
    draw_text(left + 16u + text_length(required) * 8u, 350u, " MB", UI_WHITE, UI_SURFACE);
    uint32_t button_top = screen_height - 58u;
    uint32_t button_color = memory_mb >= entry->ram_mb ? UI_YELLOW : UI_BORDER;
    fill_rect(left, button_top, width, 38u, button_color);
    draw_text(left + (width > 80u ? (width - 80u) / 2u : 4u), button_top + 15u,
              "LOAD MODEL", UI_INK, button_color);
}

static void draw_picker(uint32_t selected, uint32_t first, uint32_t memory_mb) {
    cursor_saved = 0u;
    fill_rect(0u, 0u, screen_width, screen_height, UI_DARK);
    fill_rect(0u, 0u, screen_width, 56u, UI_SURFACE);
    fill_rect(0u, 55u, screen_width, 1u, UI_BORDER);
    draw_text(20u, 14u, "BANANAMIND", UI_YELLOW, UI_SURFACE);
    draw_text(20u, 30u, "PORTABLE UEFI MODEL SYSTEM", UI_MUTED, UI_SURFACE);
    draw_text(screen_width - 152u, 20u, "MODEL LIBRARY", UI_WHITE, UI_SURFACE);
    draw_text(20u, 62u, "AVAILABLE ON THIS ISO", UI_MUTED, UI_DARK);
    for (uint32_t row = 0u; row < visible_cards() && first + row < model_count; ++row)
        draw_card(first + row, selected, first);
    draw_details(selected, memory_mb);
    draw_text(20u, screen_height - 22u, "Mouse or arrows + Enter", UI_MUTED, UI_DARK);
    cursor_draw();
}

static uint32_t pick_model(uint32_t memory_mb) {
    uint32_t selected = 0u, first = 0u;
    draw_picker(selected, first, memory_mb);
    uint8_t previous_down = 0u;
    for (;;) {
        poll_pointer();
        uint8_t clicked = pointer_down && !previous_down;
        previous_down = pointer_down;
        if (clicked) {
            uint32_t visible = visible_cards();
            if (inside(pointer_x, pointer_y, 20u, 76u, list_width(), visible * 54u)) {
                uint32_t row = ((uint32_t)pointer_y - 76u) / 54u;
                if (first + row < model_count && selected != first + row) {
                    uint32_t old = selected;
                    cursor_restore(); selected = first + row;
                    draw_card(old, selected, first); draw_card(selected, selected, first);
                    draw_details(selected, memory_mb); cursor_draw();
                }
            } else {
                uint32_t left = 42u + list_width();
                if (inside(pointer_x, pointer_y, left, screen_height - 58u,
                           screen_width - left - 20u, 38u)) {
                    cursor_restore(); return selected;
                }
            }
        }
        EFI_INPUT_KEY key;
        if (read_key(&key)) {
            uint32_t old = selected;
            if (key.ScanCode == SCAN_UP && selected) --selected;
            else if (key.ScanCode == SCAN_DOWN && selected + 1u < model_count) ++selected;
            else if (key.UnicodeChar == '\r') { cursor_restore(); return selected; }
            if (selected != old) {
                if (selected < first) first = selected;
                if (selected >= first + visible_cards()) first = selected - visible_cards() + 1u;
                draw_picker(selected, first, memory_mb);
            }
        }
        idle();
    }
}

static void draw_loading(const struct catalog_entry *entry, uint32_t percent) {
    cursor_saved = 0u;
    fill_rect(0u, 0u, screen_width, screen_height, UI_DARK);
    draw_text(24u, 22u, "BANANAMIND", UI_YELLOW, UI_DARK);
    uint32_t left = screen_width / 5u, width = screen_width - left * 2u;
    draw_text(left, screen_height / 2u - 30u, "LOADING SELECTED MODEL", UI_WHITE, UI_DARK);
    draw_text(left, screen_height / 2u - 6u, entry->name, UI_MUTED, UI_DARK);
    fill_rect(left, screen_height / 2u + 30u, width, 12u, UI_BORDER);
    fill_rect(left, screen_height / 2u + 30u, width * percent / 100u, 12u, UI_YELLOW);
    draw_text(left, screen_height / 2u + 58u,
              "Weights are loaded only now, after your selection.", UI_MUTED, UI_DARK);
}

static void release_model(void) {
    active_model = 0; active_chat = 0u;
    chat_history_length = 0u;
    if (arena_memory) { uefi_call_wrapper(boot_services->FreePool, 1, arena_memory); arena_memory = 0; }
    if (model_memory) { uefi_call_wrapper(boot_services->FreePool, 1, model_memory); model_memory = 0; }
}

static int load_model(struct catalog_entry *entry) {
    release_model();
    draw_loading(entry, 0u);
    CHAR16 path[96];
    if (!make_path(entry->filename, path)) return 0;
    EFI_FILE_HANDLE file;
    if (EFI_ERROR(open_file(volume_root, path, &file))) return 0;
    UINT64 size64 = 0u;
    if (EFI_ERROR(file_size(file, &size64)) || !size64 || size64 > 0xFFFFFFFFu) {
        uefi_call_wrapper(file->Close, 1, file); return 0;
    }
    if (EFI_ERROR(uefi_call_wrapper(boot_services->AllocatePool, 3, EfiLoaderData,
                                    (UINTN)size64, &model_memory))) {
        uefi_call_wrapper(file->Close, 1, file); return 0;
    }
    uint8_t *destination = model_memory;
    UINTN total = (UINTN)size64, completed = 0u;
    while (completed < total) {
        UINTN chunk = total - completed;
        if (chunk > 1024u * 1024u) chunk = 1024u * 1024u;
        UINTN requested = chunk;
        if (EFI_ERROR(uefi_call_wrapper(file->Read, 3, file, &requested,
                                        destination + completed)) || requested != chunk) {
            uefi_call_wrapper(file->Close, 1, file); release_model(); return 0;
        }
        completed += chunk;
        draw_loading(entry, (uint32_t)(completed * 100u / total));
    }
    uefi_call_wrapper(file->Close, 1, file);
    UINTN arena_size = (UINTN)entry->ram_mb * 1024u * 1024u;
    if (arena_size < 8u * 1024u * 1024u) arena_size = 8u * 1024u * 1024u;
    if (EFI_ERROR(uefi_call_wrapper(boot_services->AllocatePool, 3, EfiLoaderData,
                                    arena_size, &arena_memory))) {
        release_model(); return 0;
    }
    struct lm_arena arena = { (uint8_t *)arena_memory, (uint8_t *)arena_memory + arena_size };
    lm_matvec_configure(CPU_MATH_SSE2);
    static const uint32_t context_attempts[] = { 256u, 128u, 64u, 32u, 16u };
    uint8_t loaded = 0u;
    for (uint32_t attempt = 0u;
         attempt < sizeof(context_attempts) / sizeof(context_attempts[0]); ++attempt) {
        arena.next = (uint8_t *)arena_memory;
        if (litemodel_load_with_context(&runtime, model_memory, (uint32_t)size64,
                                        &arena, context_attempts[attempt])) {
            loaded = 1u;
            break;
        }
    }
    if (!loaded) {
        release_model(); return 0;
    }
    active_model = entry;
    active_chat = entry->chat || (runtime.header->flags & LITEMODEL_FLAG_CHAT);
    context_tokens_setting = 0u;
    chat_history_length = 0u;
    return 1;
}

static void output_scroll(void) {
    for (uint32_t y = output_top; y + 8u < output_bottom; ++y) {
        uint32_t *to = framebuffer + y * screen_stride + output_left;
        uint32_t *from = framebuffer + (y + 8u) * screen_stride + output_left;
        for (uint32_t x = 0u; x < output_right - output_left; ++x) to[x] = from[x];
    }
    fill_rect(output_left, output_bottom - 8u, output_right - output_left, 8u, UI_SURFACE);
    output_y = output_bottom - 8u;
}

static void output_char(char character) {
    if (character == '\n') {
        output_x = output_left; output_y += 8u;
        if (output_y + 8u > output_bottom) output_scroll();
        return;
    }
    if ((uint8_t)character < 32u && character != '\t') return;
    if (character == '\t') character = ' ';
    draw_glyph(output_x, output_y, character, UI_WHITE, UI_SURFACE);
    output_x += 8u;
    if (output_x + 8u > output_right) {
        output_x = output_left; output_y += 8u;
        if (output_y + 8u > output_bottom) output_scroll();
    }
}

static void output_text(const char *text) { while (*text) output_char(*text++); }

static uint32_t input_top(void) { return screen_height - 112u; }

static uint8_t minspark_active(void) {
    return active_model && runtime.header && runtime.header->architecture == LITEMODEL_ARCH_MINSPARK;
}

static uint32_t effective_context_tokens(void) {
    if (context_tokens_setting && context_tokens_setting < runtime.context_capacity)
        return context_tokens_setting;
    return runtime.context_capacity;
}

enum settings_action {
    SETTINGS_NONE, SETTINGS_MULTI_TURN, SETTINGS_KV_CACHE,
    SETTINGS_CONTEXT, SETTINGS_MAX_TOKENS, SETTINGS_TEMPERATURE
};

static void draw_checkbox(uint32_t left, uint32_t top, uint8_t checked,
                          uint8_t enabled) {
    fill_rect(left, top, 16u, 16u, enabled ? UI_BORDER : UI_SURFACE);
    fill_rect(left + 2u, top + 2u, 12u, 12u, UI_DARK);
    if (checked)
        draw_glyph(left + 4u, top + 4u, 'X', enabled ? UI_YELLOW : UI_MUTED, UI_DARK);
}

static void draw_settings(void) {
    fill_rect(0u, 56u, screen_width, 48u, UI_DARK);
    fill_rect(0u, 103u, screen_width, 1u, UI_BORDER);
    if (screen_width < 640u) {
        draw_text(20u, 73u, "GENERATION SETTINGS REQUIRE 640PX", UI_MUTED, UI_DARK);
        return;
    }
    uint8_t multi_available = active_chat;
    uint8_t kv_available = !minspark_active();
    draw_checkbox(28u, 72u, multi_turn_enabled && multi_available, multi_available);
    draw_text(50u, 76u, "MULTI", multi_available ? UI_WHITE : UI_MUTED, UI_DARK);
    draw_checkbox(160u, 72u, kv_cache_enabled && kv_available, kv_available);
    draw_text(182u, 76u, kv_available ? "KV CACHE" : "KV N/A",
              kv_available ? UI_WHITE : UI_MUTED, UI_DARK);

    char context_label[16] = "AUTO ";
    uint32_t offset = 5u;
    if (context_tokens_setting) {
        context_label[0] = 'K'; context_label[1] = 'V'; context_label[2] = ' ';
        offset = 3u;
    }
    number_text(effective_context_tokens(), context_label + offset);
    fill_rect(284u, 64u, 112u, 32u, UI_HOVER);
    draw_text(292u, 76u, context_label, UI_WHITE, UI_HOVER);

    char max_label[16] = "MAX ";
    number_text(max_generation_tokens, max_label + 4u);
    fill_rect(404u, 64u, 96u, 32u, UI_HOVER);
    draw_text(416u, 76u, max_label, UI_WHITE, UI_HOVER);

    char temperature_label[16] = "TEMP 0.0";
    temperature_label[5] = (char)('0' + temperature_tenths / 10u);
    temperature_label[7] = (char)('0' + temperature_tenths % 10u);
    fill_rect(508u, 64u, 112u, 32u, UI_HOVER);
    draw_text(520u, 76u, temperature_label, UI_WHITE, UI_HOVER);
}

static enum settings_action settings_hit(int32_t x, int32_t y) {
    if (screen_width < 640u || y < 64 || y >= 96) return SETTINGS_NONE;
    if (x >= 20 && x < 140) return SETTINGS_MULTI_TURN;
    if (x >= 152 && x < 276) return SETTINGS_KV_CACHE;
    if (x >= 284 && x < 396) return SETTINGS_CONTEXT;
    if (x >= 404 && x < 500) return SETTINGS_MAX_TOKENS;
    if (x >= 508 && x < 620) return SETTINGS_TEMPERATURE;
    return SETTINGS_NONE;
}

static void apply_settings_action(enum settings_action action) {
    static const uint32_t contexts[] = { 0u, 16u, 32u, 64u, 128u, 256u };
    static const uint32_t maximums[] = { 8u, 16u, 32u, 64u, 128u };
    static const uint32_t temperatures[] = { 0u, 2u, 5u, 8u, 10u };
    if (action == SETTINGS_MULTI_TURN && active_chat) {
        multi_turn_enabled = !multi_turn_enabled;
        chat_history_length = 0u;
    } else if (action == SETTINGS_KV_CACHE && !minspark_active()) {
        kv_cache_enabled = !kv_cache_enabled;
    } else if (action == SETTINGS_CONTEXT) {
        uint32_t current = 0u;
        while (current + 1u < sizeof(contexts) / sizeof(contexts[0]) &&
               contexts[current] != context_tokens_setting) ++current;
        for (uint32_t tries = 0u; tries < sizeof(contexts) / sizeof(contexts[0]); ++tries) {
            current = (current + 1u) % (sizeof(contexts) / sizeof(contexts[0]));
            if (!contexts[current] || contexts[current] <= runtime.context_capacity) {
                context_tokens_setting = contexts[current];
                break;
            }
        }
    } else if (action == SETTINGS_MAX_TOKENS) {
        uint32_t current = 0u;
        while (current + 1u < sizeof(maximums) / sizeof(maximums[0]) &&
               maximums[current] != max_generation_tokens) ++current;
        for (uint32_t tries = 0u; tries < sizeof(maximums) / sizeof(maximums[0]); ++tries) {
            current = (current + 1u) % (sizeof(maximums) / sizeof(maximums[0]));
            if (maximums[current] < effective_context_tokens()) {
                max_generation_tokens = maximums[current];
                break;
            }
        }
    } else if (action == SETTINGS_TEMPERATURE) {
        uint32_t current = 0u;
        while (current + 1u < sizeof(temperatures) / sizeof(temperatures[0]) &&
               temperatures[current] != temperature_tenths) ++current;
        current = (current + 1u) % (sizeof(temperatures) / sizeof(temperatures[0]));
        temperature_tenths = temperatures[current];
    }
}

static void draw_effort(void) {
    if (!minspark_active() || screen_width < 700u) return;
    uint32_t left = screen_width - 368u;
    uint32_t current = lm_arch_minspark_get_effort(&runtime);
    draw_text(left - 64u, 24u, "EFFORT", UI_MUTED, UI_SURFACE);
    static const char *labels[3] = { "LOW", "MED", "HIGH" };
    for (uint32_t index = 0u; index < 3u; ++index) {
        uint32_t button = left + index * 56u, loops = index + 2u;
        uint32_t background = loops == current ? UI_BORDER : UI_HOVER;
        fill_rect(button, 12u, 52u, 32u, background);
        draw_text(button + (index == 2u ? 10u : 14u), 24u, labels[index],
                  loops == current ? UI_YELLOW : UI_WHITE, background);
    }
}

static void begin_output(void) {
    uint32_t top = 120u, bottom = input_top() - 16u;
    fill_rect(20u, top, screen_width - 40u, bottom - top, UI_SURFACE);
    fill_rect(20u, top, 3u, bottom - top, UI_YELLOW);
    draw_text(36u, top + 16u, "RESPONSE", UI_MUTED, UI_SURFACE);
    output_left = 36u; output_top = top + 40u;
    output_right = screen_width - 36u; output_bottom = bottom - 16u;
    output_x = output_left; output_y = output_top;
}

static void draw_conversation(void) {
    cursor_saved = 0u;
    fill_rect(0u, 0u, screen_width, screen_height, UI_DARK);
    fill_rect(0u, 0u, screen_width, 56u, UI_SURFACE);
    fill_rect(0u, 55u, screen_width, 1u, UI_BORDER);
    draw_text(20u, 14u, "BANANAMIND", UI_YELLOW, UI_SURFACE);
    draw_text(20u, 31u, "LOCAL UEFI CONVERSATION", UI_MUTED, UI_SURFACE);
    draw_text(210u, 22u, active_model->name, UI_WHITE, UI_SURFACE);
    draw_effort();
    fill_rect(screen_width - 176u, 12u, 92u, 32u, UI_HOVER);
    draw_text(screen_width - 160u, 24u, "MODELS", UI_WHITE, UI_HOVER);
    draw_settings();
    begin_output();
    output_text("Ready. Enter a prompt below.\nNo installation, network, NVMe, or OS storage driver is in use.");
}

static void draw_input(const uint8_t *input, uint32_t length) {
    uint32_t top = input_top();
    fill_rect(20u, top, screen_width - 40u, screen_height - top - 20u, UI_SURFACE);
    fill_rect(20u, top, screen_width - 40u, 1u, UI_BORDER);
    draw_text(36u, top + 14u, "PROMPT", UI_MUTED, UI_SURFACE);
    uint32_t button_left = screen_width - 140u;
    fill_rect(button_left, top + 24u, 104u, 42u, UI_YELLOW);
    draw_text(button_left + 32u, top + 41u, "SEND", UI_INK, UI_YELLOW);
    uint32_t x = 36u, y = top + 40u;
    for (uint32_t index = 0u; index < length && y + 8u < screen_height - 24u; ++index) {
        draw_glyph(x, y, (char)input[index], UI_WHITE, UI_SURFACE);
        x += 8u;
        if (x + 8u >= button_left - 20u) { x = 36u; y += 8u; }
    }
}

static int read_prompt(uint8_t *input, uint32_t *length) {
    *length = 0u; draw_input(input, *length); cursor_draw();
    uint8_t previous_down = pointer_down;
    for (;;) {
        poll_pointer();
        uint8_t clicked = pointer_down && !previous_down;
        previous_down = pointer_down;
        if (clicked) {
            enum settings_action settings = settings_hit(pointer_x, pointer_y);
            if (settings != SETTINGS_NONE) {
                cursor_restore();
                apply_settings_action(settings);
                draw_settings();
                cursor_draw();
                continue;
            }
            if (inside(pointer_x, pointer_y, screen_width - 176u, 12u, 92u, 32u)) {
                cursor_restore(); return -1;
            }
            if (minspark_active() && pointer_y >= 12 && pointer_y < 44) {
                uint32_t left = screen_width - 368u;
                for (uint32_t index = 0u; index < 3u; ++index)
                    if (inside(pointer_x, pointer_y, left + index * 56u, 12u, 52u, 32u)) {
                        cursor_restore(); lm_arch_minspark_set_effort(&runtime, index + 2u);
                        draw_effort(); cursor_draw();
                    }
            }
            if (*length && inside(pointer_x, pointer_y, screen_width - 140u,
                                  input_top() + 24u, 104u, 42u)) {
                cursor_restore(); return 1;
            }
        }
        EFI_INPUT_KEY key;
        if (read_key(&key)) {
            if (key.ScanCode == SCAN_ESC) { cursor_restore(); return -1; }
            if (key.UnicodeChar == '\r' && *length) { cursor_restore(); return 1; }
            if (key.UnicodeChar == 8u && *length) --*length;
            else if (key.UnicodeChar >= 32u && key.UnicodeChar < 127u && *length + 1u < INPUT_BYTES)
                input[(*length)++] = (uint8_t)key.UnicodeChar;
            else { idle(); continue; }
            cursor_restore(); draw_input(input, *length); cursor_draw();
        }
        idle();
    }
}

static uint32_t chat_wrap(const uint8_t *input, uint32_t length, uint8_t output[INPUT_BYTES]) {
    static const char prefix[] = "<|user|>\n";
    static const char suffix[] = "\n<|assistant|>\n";
    uint32_t written = 0u;
    for (uint32_t index = 0u; prefix[index]; ++index) output[written++] = (uint8_t)prefix[index];
    uint32_t suffix_length = sizeof(suffix) - 1u;
    if (length > INPUT_BYTES - written - suffix_length) length = INPUT_BYTES - written - suffix_length;
    for (uint32_t index = 0u; index < length; ++index) output[written++] = input[index];
    for (uint32_t index = 0u; suffix[index]; ++index) output[written++] = (uint8_t)suffix[index];
    return written;
}

static void history_append(const uint8_t *bytes, uint32_t length) {
    if (!length) return;
    if (length >= sizeof(chat_history)) {
        bytes += length - sizeof(chat_history);
        length = sizeof(chat_history);
        chat_history_length = 0u;
    }
    if (length > sizeof(chat_history) - chat_history_length) {
        uint32_t remove = length - (sizeof(chat_history) - chat_history_length);
        for (uint32_t index = remove; index < chat_history_length; ++index)
            chat_history[index - remove] = chat_history[index];
        chat_history_length -= remove;
    }
    for (uint32_t index = 0u; index < length; ++index)
        chat_history[chat_history_length++] = bytes[index];
}

static void history_append_text(const char *text) {
    history_append((const uint8_t *)text, text_length(text));
}

static const uint8_t *prepare_model_input(const uint8_t *input, uint32_t length,
                                          uint32_t *model_length) {
    static uint8_t wrapped[INPUT_BYTES];
    if (!active_chat) {
        *model_length = length;
        return input;
    }
    if (!multi_turn_enabled) {
        *model_length = chat_wrap(input, length, wrapped);
        return wrapped;
    }
    history_append_text("<|user|>\n");
    history_append(input, length);
    history_append_text("\n<|assistant|>\n");
    *model_length = chat_history_length;
    return chat_history;
}

static uint32_t next_random(void) {
    random_state ^= random_state << 13;
    random_state ^= random_state >> 17;
    random_state ^= random_state << 5;
    return random_state;
}

static void store_token(uint16_t id) {
    if (!active_chat || !multi_turn_enabled || id == runtime.header->bos_id ||
        id == runtime.header->eos_id || id == runtime.header->pad_id) return;
    const struct litemodel_token *token = litemodel_token(&runtime, id);
    if (token) history_append(runtime.token_data + token->offset, token->length);
}

static void print_token(uint16_t id) {
    if (id == runtime.header->bos_id || id == runtime.header->eos_id ||
        id == runtime.header->pad_id) return;
    const struct litemodel_token *token = litemodel_token(&runtime, id);
    if (!token) return;
    for (uint32_t index = 0u; index < token->length; ++index)
        output_char((char)runtime.token_data[token->offset + index]);
}

static void generate(const uint8_t *input, uint32_t length) {
    uint32_t model_length = 0u;
    const uint8_t *model_input = prepare_model_input(input, length, &model_length);
    uint32_t context_tokens = effective_context_tokens();
    uint32_t reserved = max_generation_tokens;
    if (reserved >= context_tokens) reserved = context_tokens - 1u;
    uint32_t prompt_capacity = context_tokens - reserved;
    litemodel_reset(&runtime);
    uint16_t *tokens = runtime.prompt_tokens;
    uint32_t count = litemodel_tokenize(&runtime, model_input, model_length,
                                        tokens, prompt_capacity);
    begin_output();
    output_text("Generating...\n\n");
    uint8_t sequence = runtime.header->architecture == LITEMODEL_ARCH_MINSPARK;
    if (sequence) litemodel_forward_sequence(&runtime, tokens, count);
    else for (uint32_t position = 0u; position < count; ++position)
        litemodel_forward(&runtime, tokens[position], position);
    random_state ^= length + count + (uint32_t)pointer_x + ((uint32_t)pointer_y << 16);
    uint16_t next = litemodel_sample(&runtime, temperature_tenths, next_random());
    begin_output();
    uint32_t generation_limit = max_generation_tokens;
    if (generation_limit > context_tokens - count)
        generation_limit = context_tokens - count;
    for (uint32_t generated = 0u; generated < generation_limit; ++generated) {
        print_token(next);
        store_token(next);
        if (next == runtime.header->eos_id) break;
        uint32_t sequence_count = count + generated + 1u;
        tokens[sequence_count - 1u] = next;
        if (sequence || !kv_cache_enabled) {
            litemodel_reset(&runtime);
            if (sequence) {
                litemodel_forward_sequence(&runtime, tokens, sequence_count);
            } else {
                for (uint32_t position = 0u; position < sequence_count; ++position)
                    litemodel_forward(&runtime, tokens[position], position);
            }
        } else {
            litemodel_forward(&runtime, next, sequence_count - 1u);
        }
        next = litemodel_sample(&runtime, temperature_tenths, next_random());
    }
    if (active_chat && multi_turn_enabled) history_append_text("\n");
}

static void show_error(const char *title, const char *detail) {
    cursor_saved = 0u;
    fill_rect(0u, 0u, screen_width, screen_height, UI_DARK);
    draw_text(28u, 28u, title, UI_RED, UI_DARK);
    draw_text(28u, 58u, detail, UI_WHITE, UI_DARK);
    draw_text(28u, 94u, "Press Enter to continue.", UI_MUTED, UI_DARK);
    for (;;) { EFI_INPUT_KEY key; if (read_key(&key) && key.UnicodeChar == '\r') return; idle(); }
}

static EFI_STATUS fail_text(const CHAR16 *message) {
    uefi_call_wrapper(system_table->ConOut->ClearScreen, 1, system_table->ConOut);
    Print(L"BananaMind OS UEFI\r\n\r\n%.*s\r\n", 200, message);
    serial_text("BananaMind UEFI: fatal boot error\n");
    return EFI_LOAD_ERROR;
}

EFI_STATUS efi_main(EFI_HANDLE image, EFI_SYSTEM_TABLE *table) {
    InitializeLib(image, table);
    image_handle = image; system_table = table; boot_services = table->BootServices;
    serial_text("BananaMind UEFI: entered native frontend\n");
    if (!locate_model_volume()) return fail_text(L"The model catalog was not found on the EFI image.");
    serial_text("BananaMind UEFI: catalog read; zero model weights loaded\n");
    if (!parse_catalog()) return fail_text(L"The model catalog is invalid or empty.");
    if (!initialize_graphics()) return fail_text(L"A usable UEFI GOP graphics mode was not found.");
    serial_text("BananaMind UEFI: graphical model picker ready\n");
    uint32_t memory_mb = total_memory_mb();
    static uint8_t input[INPUT_BYTES];
    for (;;) {
        uint32_t selected = pick_model(memory_mb);
        if (!load_model(&models[selected])) {
            show_error("MODEL COULD NOT BE LOADED",
                       "The file is invalid, unsupported, or there is insufficient RAM.");
            continue;
        }
        serial_text("BananaMind UEFI: selected model weights loaded\n");
        draw_conversation();
        for (;;) {
            uint32_t length;
            int action = read_prompt(input, &length);
            if (action < 0) { release_model(); break; }
            generate(input, length);
        }
    }
}
