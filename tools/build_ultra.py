#!/usr/bin/env python3
"""Build the no-GRUB El Torito boot image used by BananaMind Ultra."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

SECTOR = 2048


def aligned(value: int) -> int:
    return (value + SECTOR - 1) // SECTOR * SECTOR


def symbols(elf: Path) -> dict[str, int]:
    output = subprocess.check_output(["nm", "-n", str(elf)], text=True)
    result = {}
    for line in output.splitlines():
        fields = line.split()
        if len(fields) == 3:
            result[fields[2]] = int(fields[0], 16)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel-elf", required=True, type=Path)
    parser.add_argument("--kernel-bin", required=True, type=Path)
    parser.add_argument("--loader", required=True, type=Path)
    parser.add_argument("--include", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--payload", required=True, type=Path)
    parser.add_argument("models", nargs=13, type=Path)
    args = parser.parse_args()

    kernel = args.kernel_bin.read_bytes()
    kernel_sectors = aligned(len(kernel)) // SECTOR
    syms = symbols(args.kernel_elf)
    model_address = aligned(syms["_kernel_end"])
    next_sector = kernel_sectors
    definitions = [
        "%define KERNEL_SECTOR 0",
        f"%define KERNEL_SECTORS {kernel_sectors}",
        f"%define KERNEL_ENTRY 0x{syms['_start']:x}",
        f"%define MODEL_ADDRESS 0x{model_address:x}",
    ]
    model_data = []
    for index, path in enumerate(args.models):
        data = path.read_bytes()
        definitions += [
            f"%define MODEL{index}_SECTOR {next_sector}",
            f"%define MODEL{index}_SIZE {len(data)}",
        ]
        model_data.append(data)
        next_sector += aligned(len(data)) // SECTOR
    args.include.write_text("\n".join(definitions) + "\n", encoding="ascii")

    subprocess.run(
        ["nasm", "-f", "bin", "-I", str(args.include.parent) + "/",
         str(args.loader), "-o", str(args.output) + ".loader"],
        check=True,
    )
    loader = Path(str(args.output) + ".loader").read_bytes()
    if len(loader) != 2 * SECTOR:
        raise ValueError(f"loader is {len(loader)} bytes, expected {2 * SECTOR}")

    args.output.write_bytes(loader)
    with args.payload.open("wb") as out:
        out.write(kernel)
        out.write(bytes(aligned(len(kernel)) - len(kernel)))
        for data in model_data:
            out.write(data)
            out.write(bytes(aligned(len(data)) - len(data)))
    print(f"wrote {args.output} and {args.payload} "
          f"({args.payload.stat().st_size / 1048576:.2f} MiB, 13 models)")


if __name__ == "__main__":
    main()
