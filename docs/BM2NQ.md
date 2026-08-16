# BM2NQ version 1

BM2NQ is a little-endian, mmap/in-place-friendly transformer format for the
supported BananaMind decoder families. It omits tensor names and stores tensors
in an architecture-specific fixed order so the freestanding runtime needs very
little parsing state.

## File layout

```text
+----------------------+ 0
| bm2n_header (112 B)  |
+----------------------+
| ordered weights      | weights_offset
+----------------------+
| token index entries  | token_index_offset (8 B x vocab_size)
+----------------------+
| concatenated bytes   | token_data_offset
+----------------------+
| BPE merge records    | merges_offset (8 B x merges_count)
+----------------------+ file_size
```

The packed header is defined in `include/bm2n.h`. It begins with
`BM2NQ\r\n\x1a`, records the model dimensions, precision, RoPE/RMS parameters,
special token IDs, section offsets, and a CRC-32 covering the payload.

## Matrix encodings

Vectors such as RMSNorm weights and scalar refresh alphas remain float32.
Matrices use `[out_features, in_features]` row order and no row padding.

- Q8: each row is a float32 scale plus signed bytes in `[-127,127]`.
- Q4: each row is a float32 scale plus signed nibbles in `[-7,7]`, low first.
- Q2: each row is a float32 scale plus two-bit ternary codes; 0/1/2 decode to
  `-1/0/+1`, while code 3 is reserved.
- FP16: raw little-endian IEEE-754 binary16 matrix elements.
- FP32: raw little-endian IEEE-754 binary32 matrix elements.

For quantized rows, `scale = max(abs(row))/divisor`, where the divisor is 127
for Q8, 7 for Q4, and 1 for Q2.

## Architecture tensor order

BananaMind-2-Nano and Mini store embeddings, then for every layer: input norm,
Q/K/V/O projections, Q/K norms, post-attention norm, SwiGLU gate/up/down, then
the final norm.

BananaMind-2-Micro adds three refresh norms, refresh gate/value/output
projections, the `[hidden,9]` depthwise kernel, and scalar alpha between
attention and post-attention norm.

MicroBananaMind-v1 uses embeddings, then per-layer input norm, Q/K/V/O,
post-attention norm and SwiGLU matrices, followed by the final norm. It has no
Q/K normalization. All supported checkpoints tie the LM head, so BM2NQ stores
the embedding once and reuses it for logits.

Architecture identity is inferred from the header dimension tuple. Version 1
currently recognizes Nano `(8192,256,10)`, Mini `(8192,384,14)`, BananaMind-2-
Micro `(2048,128,9)`, and MicroBananaMind-v1 `(1536,128,4)`.
