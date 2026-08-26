# BananaMind OS

BananaMind OS is a portable, installation-free x86 inference system. It boots
straight from an ISO, uses no Linux kernel or userspace at runtime, and presents
one focused graphical interface for selecting a local model and talking to it.

The graphical kernel runs on original Pentium and newer machines. It starts from
an x87-safe baseline, detects CPU features at runtime, and automatically selects
x87, SSE, or SSE2 matrix kernels. A separate i486 kernel provides the original
keyboard-driven text interface for small compatible models. GRUB never receives
a model as a boot module: it loads one kernel, and the chosen `.litemodel` is
read from the ISO only after the operating system and model picker are running.

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
| GPT-X2.5-135M | GPT-X2 | base |
| min-spark 1.1 | Meiosis | base |
| Rose-Mini | Rose X1 | base |

Supra2 Medium Base and Supra2 Medium Instruct are intentionally not included.
Exact repositories, pinned revisions, supported quantizations, RAM estimates,
and presets live in [`models/registry.json`](models/registry.json).

## Build a portable ISO



On Debian or Ubuntu, install the host-side build tools:

```sh
sudo apt install make gcc-multilib binutils python3 curl xorriso grub-pc-bin \
  grub-efi-amd64-bin gnu-efi dosfstools mtools ovmf qemu-system-x86
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
./build-model-iso.sh --preset 25 --uefi --yes
```

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

## Default development ISO

`make iso` builds `build/bananamind-os.iso` from the small BananaMind models
already described by `models/default.cfg`. Useful commands are:

```sh
make kernel       # modern kernel.elf and kernel-486.elf
make litemodels   # default .litemodel files
make iso          # portable GRUB ISO
make uefi-iso     # portable x86-64 UEFI ISO
make check        # converter unit tests
```

The default graphical kernel is a universal Pentium-and-later build. It safely
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
ISO9660 reader, byte-level BPE, quantized matrix kernels, and separate runtime
adapters for BananaMind, Llama, GPT-X2, Rose X1, and Meiosis. min-spark uses its
own full-sequence recurrent loop because that architecture has no KV cache.

See [the `.litemodel` format](docs/LITEMODEL.md) for the portable container.
The earlier BM2NQ implementation remains documented in
[`docs/BM2NQ.md`](docs/BM2NQ.md) for the optional ultra-loader compatibility
path.

Project source is MIT licensed. Downloaded checkpoints and derivative model
files remain subject to the license terms of their respective model repositories.
