#include <stdint.h>

#include "cpu.h"

#define EFLAGS_ID (1u << 21)
#define CPUID_FXSR (1u << 24)
#define CPUID_SSE (1u << 25)
#define CPUID_SSE2 (1u << 26)

static int cpuid_available(void) {
    uint32_t original;
    uint32_t changed;
    uint32_t requested;
    __asm__ volatile ("pushfl\n\tpopl %0" : "=r"(original));
    requested = original ^ EFLAGS_ID;
    __asm__ volatile ("pushl %0\n\tpopfl" : : "r"(requested) : "cc");
    __asm__ volatile ("pushfl\n\tpopl %0" : "=r"(changed));
    __asm__ volatile ("pushl %0\n\tpopfl" : : "r"(original) : "cc");
    return ((changed ^ original) & EFLAGS_ID) != 0u;
}

static uint32_t cpuid_features(void) {
    uint32_t maximum;
    uint32_t unused_b;
    uint32_t unused_c;
    uint32_t unused_d;
    __asm__ volatile ("cpuid"
                      : "=a"(maximum), "=b"(unused_b),
                        "=c"(unused_c), "=d"(unused_d)
                      : "a"(0u), "c"(0u));
    if (maximum < 1u) return 0u;
    uint32_t features;
    __asm__ volatile ("cpuid"
                      : "=a"(unused_b), "=b"(unused_c),
                        "=c"(maximum), "=d"(features)
                      : "a"(1u), "c"(0u));
    return features;
}

enum cpu_math_backend cpu_initialize_math(void) {
    if (!cpuid_available()) return CPU_MATH_X87;
    uint32_t features = cpuid_features();
    if ((features & (CPUID_FXSR | CPUID_SSE)) != (CPUID_FXSR | CPUID_SSE))
        return CPU_MATH_X87;

    uint32_t cr0;
    uint32_t cr4;
    __asm__ volatile ("mov %%cr0, %0" : "=r"(cr0));
    cr0 &= ~((1u << 2) | (1u << 3));
    cr0 |= 1u << 1;
    __asm__ volatile ("mov %0, %%cr0" : : "r"(cr0));
    __asm__ volatile ("mov %%cr4, %0" : "=r"(cr4));
    cr4 |= (1u << 9) | (1u << 10);
    __asm__ volatile ("mov %0, %%cr4" : : "r"(cr4));
    static const uint32_t mxcsr = 0x1F80u;
    __asm__ volatile ("ldmxcsr %0" : : "m"(mxcsr));
    return features & CPUID_SSE2 ? CPU_MATH_SSE2 : CPU_MATH_SSE;
}

const char *cpu_math_name(enum cpu_math_backend backend) {
    if (backend == CPU_MATH_SSE2) return "SSE2";
    if (backend == CPU_MATH_SSE) return "SSE";
    return "x87";
}
