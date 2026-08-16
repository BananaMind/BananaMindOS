.set MB_MAGIC,    0x1BADB002
.set MB_FLAGS,    0x00000007
.set MB_CHECKSUM, -(MB_MAGIC + MB_FLAGS)

.section .multiboot, "a"
.align 4
.long MB_MAGIC
.long MB_FLAGS
.long MB_CHECKSUM
/* Ask the Multiboot loader for a small linear 8-bit graphics framebuffer. */
.long 0
.long 640
.long 480
.long 8

.section .bss
.align 16
stack_bottom:
.skip 32768
stack_top:

.section .text
.global _start
.type _start, @function
.extern kernel_main
.extern _bss_start
.extern _bss_end
_start:
    cli
    cld
    mov %eax, %edx
    mov %ebx, %ebp
    mov $_bss_start, %edi
    mov $_bss_end, %ecx
    sub %edi, %ecx
    add $3, %ecx
    shr $2, %ecx
    xor %eax, %eax
    rep stosl
    mov $stack_top, %esp
    mov %edx, %eax
    mov %ebp, %ebx
    push %ebx
    push %eax
    call kernel_main
1:
    hlt
    jmp 1b
.size _start, . - _start

.section .note.GNU-stack,"",@progbits
