# BananaMind OS

BananaMind OS is a freestanding 32-bit x86 inference system for local
BananaMind models. It boots on QEMU's emulated Intel 486, needs no Linux or
libc at runtime, displays a yellow prompt/response interface, streams up to
16 generated tokens, and shows live tokens per second (TPS).

Supported checkpoints and formats:

| Checkpoint | Variants in the ISO | Recommended RAM |
|---|---|---:|
| BananaMind-2-Nano Base | Q2, Q4, Q8 | 6, 8, 14 MiB |
| BananaMind-2-Nano-Chat | Q2, Q4, Q8 | 6, 8, 14 MiB |
| BananaMind-2-Micro | Q4, Q8, FP16 | 4, 5, 8 MiB |
| MicroBananaMind-v1 | FP16, FP32 | 4, 6 MiB |
| BananaMind-2-Mini | Q2, Q4 | 12, 20 MiB |

Micro Q4 is marked **VERY LOW QUALITY** and Mini Q4 is intended for good PCs.
The kernel infers each supported architecture from its dimensions and
implements GQA attention, RoPE, RMSNorm, SwiGLU, Q/K normalization where used,
and BananaMind-2-Micro's XSA refresh gate.

## Build

On Debian/Ubuntu, install GNU make, GCC with 32-bit support, binutils, NASM,
Python 3, curl, xorriso, GRUB rescue tools, and optionally QEMU:

```sh
sudo apt install make gcc-multilib binutils nasm python3 python3-numpy \
  curl xorriso grub-pc-bin qemu-system-x86
make check
make iso ultra
```

Model downloads are pinned to exact Hugging Face revisions. They are stored
under `build/`, converted locally, and are not committed to Git.

The build creates two bootable images:

- `build/bananamind-os.iso`: GRUB menu, all 13 models, graphical framebuffer.
- `build/bananamind-ultra.iso`: custom BananaMind BIOS/El Torito loader, all
  13 models, no GRUB. It loads only the selected model and requests the same
  640x480 yellow framebuffer, with yellow VGA text as fallback.

The custom loader does not make a 0.7-1 MiB Micro system possible. Micro Q4's
weights alone are 1.54 MiB; kernel state, tokenizer, BSS, refresh history, and
KV cache raise the verified minimum to 4 MiB. Unlike GRUB, the custom loader
does successfully boot Micro Q4 at exactly 4 MiB in QEMU.

Useful build stages:

```sh
make download        # all pinned source checkpoints
make models          # all BM2NQ files
make kernel          # freestanding i486 kernel only
make iso             # standard full ISO
make ultra           # custom-loader full ISO
```

## Run

Standard graphical ISO (defaults to Nano Chat Q4):

```sh
make run
# equivalent:
qemu-system-i386 -cpu 486 -m 8 -cdrom build/bananamind-os.iso -boot d
```

Custom-loader ISO:

```sh
make run-ultra
# Micro Q4: choose 7 at the yellow custom menu (4 MiB)
qemu-system-i386 -cpu 486 -m 4 -cdrom build/bananamind-ultra.iso -boot d
```

Examples for larger selections:

```sh
# MicroBananaMind-v1 FP32: choose B
qemu-system-i386 -cpu 486 -m 6 -cdrom build/bananamind-ultra.iso -boot d

# Mini Q4: choose D
qemu-system-i386 -cpu 486 -m 20 -cdrom build/bananamind-ultra.iso -boot d
```

At up to 2 MiB below the recommended amount, the loader/kernel offers
Continue or Return. Farther below it displays Not Enough RAM and returns to
model selection. The model still must physically fit; selecting Continue
cannot bypass that limit.

## Host runner and demo prompt

The NumPy/BLAS runner reads the exact custom format used by the OS:

```sh
make host-chat
make host-micro
make host-microv1
make host-mini
```

For the strongest small demo tested here, Nano Chat Q4 answers correctly:

```sh
python3 tools/run_bm2n.py build/chat-q4.bm --chat --kernel-math \
  --prompt "What is the first letter of the alphabet? Answer only A."
```

Observed output: `The first letter of the alphabet is A.` Mini Q4 produced a
repetitive but correct `A. A. ...`; Mini Q2 did not produce a coherent answer.
Quantization can change greedy output, so Q2 and Micro Q4 should be treated as
size demonstrations, not reliable assistants.

Convert a supported safetensors checkpoint manually:

```sh
python3 tools/convert.py --model model.safetensors --tokenizer tokenizer.json \
  --config config.json --bits 4 --output model-q4.bm
python3 tools/bm2n_info.py model-q4.bm
python3 tools/run_bm2n.py model-q4.bm --prompt "Hello"
```

`--bits` accepts 2, 4, 8, 16, or 32. FP16/FP32 are intended for the requested
Micro checkpoints; Q2/Q4/Q8 use row-wise custom quantization.

## Hardware status and licensing

QEMU is tested with `-cpu 486`. Real 486 hardware is plausible but untested;
a 486DX/x87, legacy BIOS, VGA/VBE, PS/2 keyboard, and ATAPI-compatible boot
path are expected. A 486SX without a floating-point coprocessor is unsupported.

Project source is MIT licensed. The downloaded BananaMind model repositories
declare Apache-2.0; generated model files and ISOs are ignored by Git and are
created locally. See [the BM2NQ format](docs/BM2NQ.md) for the binary layout.
