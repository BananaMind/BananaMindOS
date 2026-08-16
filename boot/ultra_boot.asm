; BananaMind OS no-GRUB El Torito loader. BIOS loads this first 2048-byte
; sector at 0000:7c00; the rest of the boot image is read with INT 13h EDD.

BITS 16
ORG 0x7c00

; The build generates ultra_models.inc with kernel/model positions and sizes.
%include "ultra_models.inc"

jmp stage2
nop
times 8-($-$$) db 0
boot_pvd_lba: dd 0             ; populated by xorriso -boot-info-table
boot_file_lba: dd 0
boot_file_size: dd 0
boot_file_checksum: dd 0
times 510-($-$$) db 0
dw 0xaa55

stage2:
    cli
    xor ax, ax
    mov ds, ax
    mov es, ax
    mov ss, ax
    mov sp, 0x7a00
    sti
    cld
    mov [boot_drive], dl

    mov ax, 0x0003
    int 0x10
    mov ax, 0x0600
    mov bh, 0x60
    xor cx, cx
    mov dx, 0x184f
    int 0x10

    mov ah, 0x88               ; contiguous KiB above the first MiB
    int 0x15
    jc memory_error
    mov [memory_kb], ax
    call find_payload

.menu:
    mov si, menu_message
    call print
    xor ah, ah
    int 0x16
    cmp al, '1'
    jb .letter
    cmp al, '9'
    ja .letter
    sub al, '1'
    jmp .selected
.letter:
    or al, 0x20
    cmp al, 'a'
    jb .menu
    cmp al, 'd'
    ja .menu
    sub al, 'a' - 9
.selected:
    xor ah, ah
    mov bx, ax
    shl bx, 4
    add bx, model_table
    mov [selected_entry], bx

    movzx ax, byte [bx + 12]
    mov dx, [memory_kb]
    add dx, 1024
    add dx, 1023
    shr dx, 10
    cmp dx, ax
    jae .load
    sub ax, dx
    cmp ax, 2
    ja .too_low
    mov si, warning_message
    call print
.warning_key:
    xor ah, ah
    int 0x16
    or al, 0x20
    cmp al, 'c'
    je .load
    cmp al, 'r'
    jne .warning_key
    jmp .menu
.too_low:
    mov si, too_low_message
    call print
    xor ah, ah
    int 0x16
    jmp .menu

.load:
    mov si, loading_message
    call print

    mov eax, [payload_lba]
    add eax, KERNEL_SECTOR
    mov edx, 0x00100000
    mov cx, KERNEL_SECTORS
    call read_range

    mov bx, [selected_entry]
    mov eax, [payload_lba]
    add eax, [bx + 4]
    mov edx, MODEL_ADDRESS
    mov ecx, [bx + 8]
    add ecx, 2047
    shr ecx, 11
    call read_range

    ; Minimal Multiboot 1 information block consumed by the same C kernel.
    mov dword [0x7000], 0x00000009       ; memory + modules
    mov dword [0x7004], 640
    movzx eax, word [memory_kb]
    mov dword [0x7008], eax
    mov dword [0x7014], 1
    mov dword [0x7018], 0x00007100
    mov dword [0x7100], MODEL_ADDRESS
    mov bx, [selected_entry]
    mov eax, [bx + 8]
    add eax, MODEL_ADDRESS
    mov dword [0x7104], eax
    cmp byte [bx + 13], 0
    jne .chat_string
    mov dword [0x7108], ultra_string
    jmp .string_done
.chat_string:
    mov dword [0x7108], chat_string
.string_done:
    mov dword [0x710c], 0
    call setup_video

    cli
    lgdt [gdt_descriptor]
    mov eax, cr0
    or eax, 1
    mov cr0, eax
    jmp dword 0x08:protected_entry

; Read CX 2048-byte CD sectors from EAX into extended-memory address EDX.
read_range:
    mov [next_lba], eax
    mov [load_address], edx
    mov [sectors_left], cx
.next:
    cmp word [sectors_left], 0
    je .done
    xor ax, ax
    mov ds, ax
    mov word [dap_count], 1
    mov dword [dap_lba], 0
    mov dword [dap_lba + 4], 0
    mov eax, [next_lba]
    mov dword [dap_lba], eax
    mov si, dap
    mov dl, [boot_drive]
    mov ah, 0x42
    int 0x13
    jc disk_error

    call unreal_es
    mov esi, 0x00080000
    mov edi, [load_address]
    mov ecx, 512
    a32 rep movsd
    xor ax, ax
    mov es, ax
    inc dword [next_lba]
    add dword [load_address], 2048
    dec word [sectors_left]
    jmp .next
.done:
    ret

; Find PAYLOAD.BIN in the ISO9660 root directory, avoiding any fixed ISO LBA.
find_payload:
    xor ax, ax
    mov ds, ax
    mov dword [dap_lba], 16
    mov dword [dap_lba + 4], 0
    mov si, dap
    mov dl, [boot_drive]
    mov ah, 0x42
    int 0x13
    jc disk_error
    mov ax, 0x8000
    mov ds, ax
    mov eax, [158]             ; PVD root record extent (156 + 2)
    mov [cs:dap_lba], eax
    xor ax, ax
    mov ds, ax
    mov si, dap
    mov dl, [boot_drive]
    mov ah, 0x42
    int 0x13
    jc disk_error
    mov ax, 0x8000
    mov ds, ax
    xor si, si
.record:
    movzx bx, byte [si]
    test bx, bx
    jz .missing
    cmp byte [si + 32], 13
    jne .advance
    cmp dword [si + 33], 0x4c594150    ; PAYL
    jne .advance
    cmp dword [si + 37], 0x2e44414f    ; OAD.
    jne .advance
    cmp dword [si + 41], 0x3b4e4942    ; BIN;
    jne .advance
    cmp byte [si + 45], '1'
    jne .advance
    mov eax, [si + 2]
    mov [cs:payload_lba], eax
    xor ax, ax
    mov ds, ax
    ret
.advance:
    add si, bx
    cmp si, 2048
    jb .record
.missing:
    xor ax, ax
    mov ds, ax
    mov si, payload_message
    jmp fatal

; Ask VBE for a linear 640x480x8 framebuffer. If unavailable, the kernel
; automatically retains its yellow VGA text-mode interface.
setup_video:
    xor ax, ax
    mov es, ax
    mov di, 0x7200
    mov ax, 0x4f01
    mov cx, 0x0101
    int 0x10
    cmp ax, 0x004f
    jne .done
    mov ax, 0x4f02
    mov bx, 0x4101
    int 0x10
    cmp ax, 0x004f
    jne .done
    or dword [0x7000], 0x00001000
    mov eax, [0x7200 + 40]
    mov [0x7000 + 88], eax
    mov dword [0x7000 + 92], 0
    movzx eax, word [0x7200 + 16]
    mov [0x7000 + 96], eax
    movzx eax, word [0x7200 + 18]
    mov [0x7000 + 100], eax
    movzx eax, word [0x7200 + 20]
    mov [0x7000 + 104], eax
    mov al, [0x7200 + 25]
    mov [0x7000 + 108], al
    mov byte [0x7000 + 109], 0
.done:
    ret

; Give ES a flat 4 GiB cached limit, then return to real mode for BIOS calls.
unreal_es:
    cli
    lgdt [gdt_descriptor]
    mov eax, cr0
    or eax, 1
    mov cr0, eax
    mov ax, 0x10
    mov es, ax
    and eax, 0xfffffffe
    mov cr0, eax
    sti
    ret

disk_error:
    mov si, disk_message
    jmp fatal
memory_error:
    mov si, memory_message
fatal:
    call print
    cli
.halt:
    hlt
    jmp .halt

print:
    lodsb
    test al, al
    jz .done
    mov ah, 0x0e
    mov bx, 0x0007
    int 0x10
    jmp print
.done:
    ret

align 4
dap:
    db 0x10, 0
dap_count: dw 1
    dw 0x0000, 0x8000
dap_lba: dq 0

align 8
gdt:
    dq 0
    dq 0x00cf9a000000ffff
    dq 0x00cf92000000ffff
gdt_end:
gdt_descriptor:
    dw gdt_end - gdt - 1
    dd gdt

boot_drive: db 0
memory_kb: dw 0
selected_entry: dw 0
payload_lba: dd 0
sectors_left: dw 0
next_lba: dd 0
load_address: dd 0
menu_message:
    db 13, 10, 'BANANAMIND ULTRA - CUSTOM BOOTLOADER (NO GRUB)', 13, 10
    db '1 Nano Base Q2       2 Nano Base Q4       3 Nano Base Q8', 13, 10
    db '4 Nano Chat Q2       5 Nano Chat Q4       6 Nano Chat Q8', 13, 10
    db '7 Micro Q4 VERY LOW  8 Micro Q8           9 Micro FP16', 13, 10
    db 'A MicroBanana FP16   B MicroBanana FP32', 13, 10
    db 'C Mini Q2            D Mini Q4 GOOD PCs', 13, 10
    db 'Select 1-9 or A-D: ', 0
loading_message: db 13, 10, 'Loading selected model...', 13, 10, 0
warning_message: db 13, 10, 'LOW RAM: C=continue, R=return: ', 0
too_low_message: db 13, 10, 'ERROR: not enough RAM. Press any key.', 13, 10, 0
disk_message: db 'Disk read error.', 13, 10, 0
payload_message: db 'PAYLOAD.BIN was not found.', 13, 10, 0
memory_message: db 'Unable to detect RAM.', 13, 10, 0
ultra_string: db 'micro2 ultra', 0
chat_string: db 'chat ultra', 0

align 4
model_table:
    dd name0, MODEL0_SECTOR, MODEL0_SIZE
    db 6, 0
    dw 0
    dd name1, MODEL1_SECTOR, MODEL1_SIZE
    db 8, 0
    dw 0
    dd name2, MODEL2_SECTOR, MODEL2_SIZE
    db 14, 0
    dw 0
    dd name3, MODEL3_SECTOR, MODEL3_SIZE
    db 6, 1
    dw 0
    dd name4, MODEL4_SECTOR, MODEL4_SIZE
    db 8, 1
    dw 0
    dd name5, MODEL5_SECTOR, MODEL5_SIZE
    db 14, 1
    dw 0
    dd name6, MODEL6_SECTOR, MODEL6_SIZE
    db 4, 0
    dw 0
    dd name7, MODEL7_SECTOR, MODEL7_SIZE
    db 5, 0
    dw 0
    dd name8, MODEL8_SECTOR, MODEL8_SIZE
    db 8, 0
    dw 0
    dd name9, MODEL9_SECTOR, MODEL9_SIZE
    db 4, 0
    dw 0
    dd name10, MODEL10_SECTOR, MODEL10_SIZE
    db 6, 0
    dw 0
    dd name11, MODEL11_SECTOR, MODEL11_SIZE
    db 12, 0
    dw 0
    dd name12, MODEL12_SECTOR, MODEL12_SIZE
    db 20, 0
    dw 0
name0: db 'Nano Base Q2', 0
name1: db 'Nano Base Q4', 0
name2: db 'Nano Base Q8', 0
name3: db 'Nano Chat Q2', 0
name4: db 'Nano Chat Q4', 0
name5: db 'Nano Chat Q8', 0
name6: db 'Micro Q4', 0
name7: db 'Micro Q8', 0
name8: db 'Micro FP16', 0
name9: db 'MicroBanana FP16', 0
name10: db 'MicroBanana FP32', 0
name11: db 'Mini Q2', 0
name12: db 'Mini Q4', 0

BITS 32
protected_entry:
    mov ax, 0x10
    mov ds, ax
    mov es, ax
    mov fs, ax
    mov gs, ax
    mov ss, ax
    mov esp, 0x0009f000
    mov eax, 0x2badb002
    mov ebx, 0x00007000
    jmp KERNEL_ENTRY

BITS 16
times 4096-($-$$) db 0
