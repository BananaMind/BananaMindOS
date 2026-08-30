.set MB_MAGIC,    0x1BADB002
#ifdef COMPATIBILITY_KERNEL
.set MB_FLAGS,    0x00000003
#else
.set MB_FLAGS,    0x00000007
#endif
.set MB_CHECKSUM, -(MB_MAGIC + MB_FLAGS)

.section .multiboot, "a"
.align 4
.long MB_MAGIC
.long MB_FLAGS
.long MB_CHECKSUM
#ifndef COMPATIBILITY_KERNEL
/* Modern UI target. Compatibility mode deliberately ignores the framebuffer. */
/* Multiboot keeps the optional address fields in the header even when the
   a.out-kludge flag is clear.  Leave those five fields zero, then request a
   deterministic VBE mode for bootloaders which do not have a GRUB gfxmode
   command to override it (notably the DOSBox-X hard-disk image). */
.long 0
.long 0
.long 0
.long 0
.long 0
.long 0
.long 1024
.long 768
.long 32
#endif

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
    /* GCC's i386 ABI expects ESP to be 16-byte aligned immediately before a
       call.  Two 32-bit arguments consume eight bytes, so reserve another
       eight bytes before pushing them.  Without this padding, SSE2 functions
       can fault on compiler-generated MOVAPS stack spills. */
    mov $stack_top, %esp
    and $-16, %esp
    sub $8, %esp
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
