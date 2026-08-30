# BananaMind OS

BananaMind OS is a portable, installation-free x86 inference system. It boots
straight from an ISO, uses no Linux kernel or userspace at runtime, and presents
one focused graphical interface for selecting a local model and talking to it.

The graphical kernel runs on original Pentium and newer machines. It starts from
an x87-safe baseline, detects CPU features at runtime, and automatically selects
x87, SSE, or SSE2 matrix kernels. A 486 or newer can also use the QOI-backed
High Quality catalog with the supplied photographic background, transparent
icons, rounded frosted panels, and localized redraws. A separate i486 kernel
provides the original keyboard-driven text interface for small compatible
models. GRUB never receives a model as a boot module: it loads one kernel, and
the chosen `.litemodel` is read from the ISO only after the operating system and
model picker are running.

## Supported models

All architectures use the same architecture-neutral `.litemodel` container.
Their converters and inference implementations remain separate source modules.

| Model | Runtime module | Modes |
|---|---|---|
| BananaMind 2 Nano / Nano Chat | BananaMind | base, chat |
| BananaMind 2 Micro | BananaMind | base |
| MicroBananaMind v1 | BananaMind | base |
| BananaMind 2 Mini / Mini Chat | BananaMind | base, chat |
| BananaMind 2 Medium / Medium Chat | BananaMind | base, chat |
| BananaMind 2 Pro | BananaMind | base |
| BananaMind 2 Pro Preview Chat | BananaMind | chat |
| SmolLM-135M | Llama | base |
| SmolLM2-135M | Llama | base |
| SmolLM2-360M-Instruct | Llama | chat |
| LFM2.5-230M / 350M | LFM2 hybrid | chat |
| Gemma 3 270M | Gemma 3 | base |
| Qwen3.5-0.8B | Qwen3.5 hybrid text runtime | chat |
| GPT-X2.5-135M | GPT-X2 | base |
| min-spark 1.1 | Meiosis | base |
| Rose-Mini | Rose X1 | base |

Supra2 Medium Base and Supra2 Medium Instruct are intentionally not included.
Exact repositories, pinned revisions, supported quantizations, RAM estimates,
and presets live in [`models/registry.json`](models/registry.json).
Gemma 3 is gated: accept Google's Gemma terms on Hugging Face and set
`HF_TOKEN` before running the builder. Qwen3.5 conversion intentionally keeps
the causal-language stack and omits vision and speculative-decoding tensors,
because BananaMind OS currently accepts text prompts only.

Every model architecture supports densely packed Q1 through Q8 matrices as
well as FP16 and FP32 where exposed by the registry. Q1 is binary, Q2 retains
the existing ternary representation, and Q3-Q8 are signed symmetric formats.

## Build a portable ISO



On Debian or Ubuntu, install the host-side build tools:

```sh
sudo apt install make gcc-multilib binutils python3 curl xorriso grub-pc-bin \
  grub-efi-amd64-bin gnu-efi dosfstools mtools syslinux parted ovmf \
  qemu-system-x86
```

Run the interactive builder:

```sh
./build-model-iso.sh
```

On Windows, `build-model-iso.bat` provides the same model menu. Python can run
the downloader and converter natively; GRUB ISO creation requires the build to
run under WSL with the packages above.

The builder downloads only the models selected, pins them to the revisions in
the registry, quantizes them locally, writes `.litemodel` files, creates the
post-boot catalog, and builds the ISO. Downloads and generated weights remain
under `build/` and are not installed on the host.

Non-interactive examples:

```sh
./build-model-iso.sh --preset 25 --yes
./build-model-iso.sh --models mini-chat:4,rose-mini:2 --yes --output build/custom.iso
./build-model-iso.sh --models nano-base:1,2,3,4,5,6,7,8,mini-chat:4,8 --yes
./build-model-iso.sh --preset 25 --uefi --yes
```

After a model ID and colon, additional comma-separated numbers select more
quantizations of that same model. Repeating the full model ID also remains
supported.

The supplied RAM profiles can also be built through Make:

```sh
make preset-10
make preset-25
make preset-100
make preset-250
# or all four:
make preset-isos
```

Each profile contains several useful choices that individually fit its target
RAM class; the OS loads only the selected model. Generated images are named
`build/bananamind-{10,25,100,250}mb.iso`.

## DOSBox-X images

DOSBox-X cannot directly start the no-emulation El Torito boot record used by
the GRUB ISO. The DOSBox-X target therefore creates a small bootable FAT hard
disk and a separate ATAPI model CD:

```sh
make dosboxx
dosbox-x -conf dosbox-x.conf
```

The generated files are `build/bananamind-dosboxx.img` and
`build/bananamind-dosboxx-micro-q4.iso`. The CD contains only BananaMind 2
Micro Q4 plus the UI assets and catalog. The hard disk contains the boot menu
and kernels but zero model weights; the Q4 weights remain on the mounted CD
until selected in the model library.

For a single-model Nano Chat Q4 CD instead, run:

```sh
make dosboxx-nano-chat
dosbox-x -conf dosbox-x-nano-chat.conf
```

This keeps the same weight-free boot disk and creates
`build/bananamind-dosboxx-nano-chat-q4.iso`.

To enter the commands manually at the DOSBox-X prompt, run:

```text
IMGMOUNT C build/bananamind-dosboxx.img -ide 1m
IMGMOUNT D build/bananamind-dosboxx-micro-q4.iso -t iso -ide 2m
BOOT C:
```

## Default development ISO

`make iso` builds `build/bananamind-os.iso` from the small BananaMind models
already described by `models/default.cfg`. Useful commands are:

```sh
make kernel       # modern kernel.elf and kernel-486.elf
make hq-assets    # convert the catalog artwork to standard QOI files
make litemodels   # default .litemodel files
make iso          # portable GRUB ISO
make uefi-iso     # portable x86-64 UEFI ISO
make check        # converter unit tests
```

`BUILD_NUMBER` is compiled into the BIOS and UEFI model-library screens. The
maintainer checkout contains an ignored `dev_folder/`, so every Make invocation
there increments the number before compiling. Normal clones do not contain that
folder; rebuilding unchanged source therefore keeps the published build number.

The default graphical kernel is a universal Pentium-and-later build. GRUB offers
three frontends: High Quality mode for 486 and newer, the current
graphical mode for Pentium and newer, and the text-mode 486 compatibility path.
The kernel safely
detects `CPUID` at runtime and selects x87 on Pentium/Pentium II, SSE on Pentium
III, or SSE2 on Pentium 4 and newer:

```sh
make iso
```

`GUI_CPU` can still be set to `pentium`, `pentium2`, `pentium3`, or `pentium4`
for a CPU-specific build. Each variant uses its own object directory, so
switching targets does not require `make clean`.

Run the modern GUI in QEMU:

```sh
qemu-system-i386 -cpu qemu32 -m 128 -cdrom build/bananamind-os.iso -boot d
```

For a native x86-64 UEFI machine, build and run:

```sh
make uefi-iso
make run-uefi
```

For a manual QEMU command, attach a firmware-visible pointing device with
`-device qemu-xhci -device usb-tablet`. The UEFI frontend accepts both the
absolute-pointer protocol used by QEMU tablets and the relative-pointer
protocol used by many physical firmware implementations. Arrow keys and Enter
remain available when firmware exposes no pointer protocol.

This produces `build/bananamind-os-uefi.iso`. Its x86-64 EFI GRUB image
chainloads the native BananaMindOS frontend. The frontend uses UEFI GOP and
filesystem services, reads only `CATALOG.CFG` before showing the model picker,
and loads the selected `.litemodel` afterward. Secure Boot must be disabled
because the locally built EFI binaries are unsigned.

No NVMe driver is included or required for this boot path: the firmware reads
the portable FAT boot image. BananaMindOS does not mount or modify installed
disks. Raw-writing the ISO to removable media keeps the same installation-free
behavior.

Choose “486 compatibility mode” in GRUB for a 486DX/x87 machine. That entry
loads `kernel-486.elf`, ignores the graphical interface, and lists only catalog
entries marked `legacy`. A 486SX without an FPU is unsupported.

## Runtime design

The modern UI is not a desktop environment. It is a single model-library and
conversation workflow with mouse and keyboard input. The Models button returns
to the post-boot library without rebooting, while Send submits the prompt.
High Quality mode streams its standard QOI artwork from `/boot/assets`, keeping
only one 2 KiB compressed-input sector in memory. It draws the background once;
mouse motion restores only the cursor rectangle and model selection redraws only
the affected rows and details panel. The High Quality inference screen continues
the same background and frosted-panel design, with a QOI Models icon, the supplied
transparent Send icon, response and prompt panels, and text-only Auto, Max, and
Temperature controls. Loading a selection opens a glass progress modal over the
catalog and updates only its progress fill and percentage.
Its catalog starts in Basic mode, grouping every locally available quantization
under one model name. The precision slider beneath Minimum RAM chooses the exact
Q/FP file that will be loaded and immediately updates the displayed memory
requirement. The Advanced mode checkbox beside the catalog heading restores one
row per quantization.
The BIOS model library normally blocks models whose catalog RAM requirement is
larger than installed system RAM. Physically holding `?` and `G` together
enables the hidden unsafe override in the DOSBox-X, default, custom, and preset
BIOS builds. A warning must be acknowledged first because forcing an oversized
model can corrupt memory, crash, or reset the machine. Hard physical bounds are
still checked where the loader can determine them.
When min-spark is active, the header exposes Low, Medium, and High effort
controls, corresponding to two, three, and four recurrent Meiosis loops.

The generation bar is available in both the BIOS and native UEFI graphical
frontends:

- **Multi** is enabled only for catalog entries marked as chat checkpoints. It
  preserves prior user and assistant turns in a bounded history buffer. Turning
  it off clears that history and makes each prompt an independent request.
- **KV Cache** reuses transformer keys and values when enabled. Turning it off
  recomputes the complete active sequence for every generated token, which uses
  the same context but is substantially slower. min-spark shows `KV N/A`
  because its Meiosis runtime is a full-sequence recurrent architecture.
- **Auto N** shows the automatically selected context/KV capacity. Model load
  tries 256, 128, 64, 32, then 16 tokens and keeps the largest capacity allowed
  by both the checkpoint and available RAM. Clicking it cycles through manual
  capacities that fit the allocated cache.
- **Max** cycles the maximum generated response through 8, 16, 32, 64, and 128
  tokens, subject to the remaining context capacity.
- **Temp** cycles 0.0, 0.2, 0.5, 0.8, and 1.0. A temperature of 0.0 is greedy;
  nonzero values sample from the model's softmax distribution.

The kernel includes PS/2 mouse support, ATA Packet Interface CD reads, a small
ISO9660 reader, BPE tokenization, quantized matrix kernels, and separate runtime
adapters for BananaMind, Llama, GPT-X2, Rose X1, Meiosis, LFM2, Gemma 3, and
Qwen3.5. min-spark uses its own full-sequence recurrent loop because that
architecture has no KV cache. LFM2 and Qwen3.5 retain their recurrent state in
their own architecture modules rather than mixing it into the transformer
runtime.

See [the `.litemodel` format](docs/LITEMODEL.md) for the portable container.
The earlier BM2NQ implementation remains documented in
[`docs/BM2NQ.md`](docs/BM2NQ.md) for the optional ultra-loader compatibility
path.

Project source is MIT licensed. Downloaded checkpoints and derivative model
files remain subject to the license terms of their respective model repositories.
