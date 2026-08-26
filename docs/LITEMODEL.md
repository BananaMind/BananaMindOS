# `.litemodel` format version 1

`.litemodel` is BananaMind OS's common portable model container. The file
format describes tensor dimensions, quantization, tokenizer data, and an opaque
architecture configuration blob. It does not combine the architecture
implementations: those remain in separate converter modules under
`tools/architectures/` and separate kernel modules under `src/arch_*.c`.

## File layout

All integer and floating-point fields are little-endian.

1. 128-byte common header
2. architecture-specific configuration blob, padded to four bytes
3. ordered model weights
4. vocabulary index (`vocab_size` entries)
5. concatenated token bytes
6. BPE merge entries

The magic is the eight bytes `LITEMDL\x1a`; the current version is 1. Header
offsets and sizes are validated before architecture parsing. Each token entry
contains a 32-bit byte offset, 16-bit byte length, and 16 reserved bits. Each
merge entry contains four 16-bit values: left token, right token, result token,
and merge rank.

Architecture IDs are:

| ID | Architecture | Kernel source |
|---:|---|---|
| 1 | BananaMind family | `src/arch_banana.c` |
| 2 | Llama / SmolLM | `src/arch_llama.c` |
| 3 | GPT-X2 | `src/arch_gptx.c` |
| 4 | Rose X1 | `src/arch_rose.c` |
| 5 | min-spark Meiosis | `src/arch_minspark.c` |

## Matrix encoding

FP32 stores rows as little-endian 32-bit floats. FP16 stores IEEE binary16.
Q8, Q4, and Q2 are row-wise symmetric quantizations. Every quantized row begins
with an FP32 scale followed by packed signed values:

- Q8: one signed byte per value, range -127 through 127.
- Q4: two two's-complement nibbles per byte, range -7 through 7.
- Q2: four codes per byte representing -1, 0, or 1.

The ordered tensors are architecture-defined. `convert_litemodel.py` selects a
module from `tools/architectures/`, and the matching `src/arch_*.c` module parses
that exact order. Adding an architecture therefore does not change the common
container or require putting unrelated inference code into one source file.

## Boot and loading contract

GRUB configuration must not use `module` for `.litemodel` files. The kernel
boots first, reads `/boot/models/CATALOG.CFG` through its ISO9660/ATAPI path,
shows names and quantizations, and reads only the selected file into the model
arena. Changing models reuses that arena. Consequently, model weights occupy
no RAM during GRUB or the initial OS/model-library startup.

The runtime allocates the KV/context arena only as part of that post-selection
load. It attempts capacities from 256 down to 16 tokens and exposes the largest
successful value as `Auto` in the GUI. A smaller manual setting limits the
active sequence without changing the on-disk container. Chat history is stored
by the frontend as text and retokenized on each turn; it is not model weight
data and is cleared when multi-turn is disabled or a different model is loaded.
