# `.litemodel` format version 2

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

The magic is the eight bytes `LITEMDL\x1a`; the current version is 2. Header
offsets and sizes are validated before architecture parsing. Each token entry
contains a 32-bit byte offset, 16-bit byte length, and 16 flag bits. Token flag
bit 0 marks an atomic special token such as `<|im_start|>`. Version 2 merge
entries contain four 32-bit values: left token, right token, result token, and
merge rank. This is required by Gemma 3 and Qwen3.5 tokenizers. The loader also
accepts version 1 files and expands their original four 16-bit merge fields at
runtime. Both versions keep the `.litemodel` filename extension.

Architecture IDs are:

| ID | Architecture | Kernel source |
|---:|---|---|
| 1 | BananaMind family | `src/arch_banana.c` |
| 2 | Llama / SmolLM | `src/arch_llama.c` |
| 3 | GPT-X2 | `src/arch_gptx.c` |
| 4 | Rose X1 | `src/arch_rose.c` |
| 5 | min-spark Meiosis | `src/arch_minspark.c` |
| 6 | LiquidAI LFM2 | `src/arch_lfm2.c` |
| 7 | Google Gemma 3 | `src/arch_gemma3.c` |
| 8 | Qwen3.5 text hybrid | `src/arch_qwen35.c` |

Version 2 tokenization recognizes marked control tokens atomically and supports
32-bit vocabulary IDs and merge ranks. Gemma's SentencePiece-style leading and
inter-word `▁` marker is selected by a container flag. Reserved embedding rows
without a tokenizer piece are retained as inert, non-encodable IDs.

## Matrix encoding

FP32 stores rows as little-endian 32-bit floats. FP16 stores IEEE binary16.
Q1 through Q8 are row-wise symmetric quantizations. Every quantized row begins
with an FP32 scale followed by a dense, little-endian bit stream. Value zero
starts at the least-significant bit of the first packed byte; values may cross
byte boundaries:

- Q1: one binary code representing `-scale` or `+scale`.
- Q2: four codes per byte representing -1, 0, or 1.
- Q3-Q8: two's-complement signed codes with the most-negative code unused;
  their ranges are `[-3,3]`, `[-7,7]`, `[-15,15]`, `[-31,31]`,
  `[-63,63]`, and `[-127,127]` respectively.

The physical matrix payload is exactly Qn's nominal `n` bits per weight plus
one 32-bit scale per row. Vectors such as normalization weights remain FP32.

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
