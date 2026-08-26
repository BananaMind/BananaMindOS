#ifndef CPU_H
#define CPU_H

#include <stdint.h>

enum cpu_math_backend {
    CPU_MATH_X87 = 0,
    CPU_MATH_SSE = 1,
    CPU_MATH_SSE2 = 2,
};

/* Safe on 486: tests EFLAGS.ID before executing CPUID. */
enum cpu_math_backend cpu_initialize_math(void);
const char *cpu_math_name(enum cpu_math_backend backend);

#endif
