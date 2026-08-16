ENTRY(_start)

SECTIONS
{
    . = 1M;
    .multiboot : { KEEP(*(.multiboot)) }
    .text ALIGN(4096) : { *(.text*) }
    .rodata ALIGN(4096) : { *(.rodata*) }
    .data ALIGN(4096) : { *(.data*) }
    .bss ALIGN(4096) : {
        _bss_start = .;
        *(COMMON)
        *(.bss*)
        _bss_end = .;
    }
    _kernel_end = .;
}
