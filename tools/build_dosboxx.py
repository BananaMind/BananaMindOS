#!/usr/bin/env python3
"""Build the DOSBox-X boot hard disk and a single-model CD."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
IMAGE_BYTES = 32 * 1024 * 1024
PARTITION_OFFSET = 1024 * 1024
PARTITION_SECTOR = PARTITION_OFFSET // 512
SYSLINUX_FILES = ("mboot.c32", "menu.c32", "libcom32.c32", "libutil.c32")


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def require(command: str, guidance: str) -> None:
    if shutil.which(command) is None:
        raise SystemExit(f"Missing required tool: {command}\n{guidance}")


def syslinux_paths() -> tuple[Path, dict[str, Path]]:
    directories = (
        Path("/usr/lib/syslinux/bios"),
        Path("/usr/lib/syslinux/modules/bios"),
        Path("/usr/share/syslinux"),
    )
    modules: dict[str, Path] = {}
    for filename in SYSLINUX_FILES:
        for directory in directories:
            candidate = directory / filename
            if candidate.is_file():
                modules[filename] = candidate
                break
    mbr_candidates = (
        Path("/usr/lib/syslinux/bios/mbr.bin"),
        Path("/usr/lib/syslinux/mbr/mbr.bin"),
        Path("/usr/share/syslinux/mbr.bin"),
    )
    mbr = next((path for path in mbr_candidates if path.is_file()), None)
    missing = [name for name in SYSLINUX_FILES if name not in modules]
    if mbr is None or missing:
        detail = ", ".join((["mbr.bin"] if mbr is None else []) + missing)
        raise SystemExit(
            f"Missing BIOS Syslinux files: {detail}\n"
            "Install the syslinux and syslinux-common packages."
        )
    return mbr, modules


def catalog_line(catalog_path: Path, model_id: str, variant: str) -> str:
    matches = []
    for line in catalog_path.read_text(encoding="ascii").splitlines():
        fields = line.split("|")
        if len(fields) == 8 and fields[0] == model_id and fields[3] == variant:
            matches.append(line)
    if len(matches) != 1:
        raise SystemExit(
            f"models/default.cfg must contain exactly one {model_id} {variant} entry"
        )
    return matches[0]


def micro_q4_catalog_line(catalog_path: Path) -> str:
    """Backward-compatible helper retained for callers of the original builder."""
    return catalog_line(catalog_path, "micro2", "Q4")


def build_boot_image(output: Path, kernel: Path, compatibility_kernel: Path,
                     config: Path) -> None:
    for tool in ("parted", "mkfs.vfat", "syslinux", "mmd", "mcopy"):
        require(tool, "Install parted, dosfstools, syslinux, and mtools.")
    mbr, modules = syslinux_paths()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    with temporary.open("wb") as image:
        image.truncate(IMAGE_BYTES)
    run(["parted", "-s", str(temporary), "mklabel", "msdos", "mkpart",
         "primary", "fat16", "1MiB", "100%", "set", "1", "boot", "on"])
    run(["mkfs.vfat", "-F", "16", "--offset", str(PARTITION_SECTOR),
         "-n", "BMOSBOOT", str(temporary)])
    run(["syslinux", "--install", "--offset", str(PARTITION_OFFSET),
         str(temporary)])

    mbr_data = mbr.read_bytes()
    if len(mbr_data) != 440:
        raise SystemExit(f"Unexpected Syslinux MBR size: {len(mbr_data)} bytes")
    with temporary.open("r+b") as image:
        image.write(mbr_data)

    target = f"{temporary}@@{PARTITION_OFFSET}"
    run(["mmd", "-i", target, "::/boot"])
    run(["mcopy", "-o", "-i", target, str(kernel), "::/boot/kernel.elf"])
    run(["mcopy", "-o", "-i", target, str(compatibility_kernel),
         "::/boot/kernel-486.elf"])
    run(["mcopy", "-o", "-i", target, str(config), "::/syslinux.cfg"])
    for filename in SYSLINUX_FILES:
        run(["mcopy", "-o", "-i", target, str(modules[filename]), f"::/{filename}"])

    sector = temporary.read_bytes()[:512]
    if sector[510:512] != b"\x55\xaa" or not (sector[446 + 0] & 0x80):
        raise SystemExit("Generated hard-disk image has no active bootable partition")
    temporary.replace(output)


def build_model_iso(output: Path, model: Path, catalog_path: Path,
                    assets: Path, model_id: str, variant: str,
                    model_filename: str, volume_label: str) -> None:
    require("xorriso", "Install xorriso to create the model CD image.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dosboxx-iso-", dir=output.parent) as work:
        stage = Path(work)
        model_dir = stage / "boot" / "models"
        asset_dir = stage / "boot" / "assets"
        model_dir.mkdir(parents=True)
        asset_dir.mkdir(parents=True)
        shutil.copy2(model, model_dir / model_filename)
        catalog = "# id|filename|name|variant|ram_mb|mode|cpu|description\n"
        selected = catalog_line(catalog_path, model_id, variant).split("|")
        selected[1] = model_filename
        catalog += "|".join(selected) + "\n"
        (model_dir / "CATALOG.CFG").write_text(catalog, encoding="ascii")
        for asset in sorted(assets.glob("*.QOI")):
            shutil.copy2(asset, asset_dir / asset.name)
        if not any(asset_dir.iterdir()):
            raise SystemExit(f"No QOI assets found in {assets}")
        temporary = output.with_suffix(output.suffix + ".part")
        run(["xorriso", "-as", "mkisofs", "-R", "-J", "-iso-level", "3",
             "-V", volume_label, "-o", str(temporary), str(stage)])
        temporary.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boot-output", type=Path, required=True)
    parser.add_argument("--models-output", type=Path, required=True)
    parser.add_argument("--kernel", type=Path, required=True)
    parser.add_argument("--compat-kernel", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--catalog-id", default="micro2")
    parser.add_argument("--variant", default="Q4")
    parser.add_argument("--model-filename", default="MIC4.LITEMODEL")
    parser.add_argument("--volume-label", default="BMOS_MICRO_Q4")
    args = parser.parse_args()
    build_boot_image(args.boot_output, args.kernel, args.compat_kernel, args.config)
    build_model_iso(
        args.models_output, args.model, args.catalog, args.assets,
        args.catalog_id, args.variant, args.model_filename, args.volume_label,
    )
    print(f"\nReady: {args.boot_output}")
    print(f"Ready: {args.models_output}")
    print(
        f"The boot disk contains no model weights; {args.catalog_id} "
        f"{args.variant} stays on the CD until selected."
    )


if __name__ == "__main__":
    main()
