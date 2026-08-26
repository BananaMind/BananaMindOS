#!/usr/bin/env python3
"""Package BananaMindOS's x86-64 EFI application and model store as an ISO."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


MIB = 1024 * 1024


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def require(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(f"Missing required tool: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", type=Path, required=True)
    parser.add_argument("--grub", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()

    for tool in ("mkfs.vfat", "mcopy", "mmd", "xorriso"):
        require(tool)
    if not (args.models / "CATALOG.CFG").is_file():
        raise SystemExit(f"Missing model catalog: {args.models / 'CATALOG.CFG'}")

    args.work.mkdir(parents=True, exist_ok=True)
    iso_root = args.work / "root"
    iso_root.mkdir(parents=True, exist_ok=True)
    image = iso_root / "efi.img"
    content_bytes = args.app.stat().st_size + args.grub.stat().st_size
    content_bytes += sum(path.stat().st_size for path in args.models.iterdir() if path.is_file())
    image_mib = max(64, ((content_bytes + 16 * MIB + 16 * MIB - 1) // (16 * MIB)) * 16)
    with image.open("wb") as stream:
        stream.truncate(image_mib * MIB)
    run(["mkfs.vfat", "-F", "32", "-n", "BANANAMIND", str(image)])
    for directory in ("::/EFI", "::/EFI/BOOT", "::/MODELS"):
        run(["mmd", "-i", str(image), directory])
    run(["mcopy", "-i", str(image), str(args.grub), "::/EFI/BOOT/BOOTX64.EFI"])
    run(["mcopy", "-i", str(image), str(args.app), "::/EFI/BOOT/BANANA.EFI"])
    for path in sorted(args.models.iterdir()):
        if path.is_file():
            run(["mcopy", "-i", str(image), str(path), f"::/MODELS/{path.name}"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    run([
        "xorriso", "-as", "mkisofs", "-R", "-J", "-V", "BANANAMIND_UEFI",
        "-efi-boot-part", "--efi-boot-image",
        "-eltorito-alt-boot", "-e", "efi.img", "-no-emul-boot",
        "-o", str(args.output), str(iso_root),
    ])
    print(f"Ready: {args.output}")
    print("UEFI GRUB loads only BANANA.EFI; model weights remain on the FAT image until selected.")


if __name__ == "__main__":
    main()
