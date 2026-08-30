#!/usr/bin/env python3
"""Interactive downloader, quantizer, catalog writer, and ISO builder."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = ROOT / "models" / "registry.json"
BUILD = ROOT / "build"


def run(command: list[str], hidden: str | None = None):
    shown = ["<HF_TOKEN>" if hidden and part == hidden else part for part in command]
    print("+", " ".join(shown), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def require(command: str, guidance: str):
    if shutil.which(command) is None:
        raise SystemExit(f"Missing required tool: {command}\n{guidance}")


def load_registry():
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    models = {model["id"]: model for model in registry["models"]}
    return registry, models


def parse_selections(text: str, models: dict) -> list[tuple[dict, int]]:
    selections = []
    seen = set()
    current_model_id = None
    for item in (part.strip() for part in text.split(",")):
        if not item:
            continue
        try:
            if ":" in item:
                model_id, bits_text = item.rsplit(":", 1)
                model_id = model_id.strip()
                current_model_id = model_id
            elif current_model_id is not None:
                model_id, bits_text = current_model_id, item
            else:
                raise ValueError
            bits_text = bits_text.strip()
            bits = int(bits_text)
            model = models[model_id]
        except (ValueError, KeyError) as error:
            raise SystemExit(
                f"Invalid selection {item!r}; use model-id:bits or model-id:bits,bits"
            ) from error
        if bits_text not in model["quants"]:
            available = ", ".join(model["quants"])
            raise SystemExit(f"{model_id} has no {bits}-bit option; choose {available}")
        key = (model_id, bits)
        if key not in seen:
            selections.append((model, bits))
            seen.add(key)
    if not selections:
        raise SystemExit("No models selected")
    return selections


def interactive_selection(registry: dict, models: dict) -> tuple[list[tuple[dict, int]], str]:
    print("\nBananaMind OS portable ISO builder\n")
    print("RAM presets (models stay on the ISO until selected):")
    for ram in ("10", "25", "100", "250"):
        print(f"  {ram:>3} MB  " + ", ".join(registry["presets"][ram]))
    answer = input("\nPreset [10/25/100/250] or C for custom: ").strip().lower()
    if answer in registry["presets"]:
        return parse_selections(",".join(registry["presets"][answer]), models), answer
    if answer != "c":
        raise SystemExit("Cancelled")
    ordered = list(models.values())
    print("\nAvailable models:")
    for index, model in enumerate(ordered, 1):
        quants = "/".join(f"Q{bits}" if int(bits) <= 8 else f"FP{bits}"
                          for bits in model["quants"])
        print(f"  {index:2}. {model['name']:<38} {quants}")
    numbers = input("\nModel numbers (comma-separated): ").strip()
    chosen = []
    for value in numbers.split(","):
        try:
            model = ordered[int(value.strip()) - 1]
        except (ValueError, IndexError) as error:
            raise SystemExit(f"Invalid model number {value!r}") from error
        choices = list(model["quants"])
        default = "4" if "4" in choices else choices[0]
        bits = input(
            f"{model['name']} precision(s) [{'/'.join(choices)}], "
            f"comma-separated, default {default}: "
        ).strip() or default
        chosen.append(f"{model['id']}:{bits}")
    return parse_selections(",".join(chosen), models), "custom"


def download(url: str, destination: Path, gated: bool = False):
    if destination.exists() and destination.stat().st_size:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    command = ["curl", "-L", "--fail", "--retry", "3"]
    token = os.environ.get("HF_TOKEN", "")
    if gated:
        if not token:
            raise SystemExit(
                "This model is gated. Accept its Hugging Face license and set HF_TOKEN "
                "to a read token before building."
            )
        command.extend(["--header", f"Authorization: Bearer {token}"])
    command.extend(["-o", str(temporary), url])
    run(command, hidden=f"Authorization: Bearer {token}" if token else None)
    temporary.replace(destination)


def disk_filename(model_id: str, bits: int) -> str:
    stem = "".join(character for character in model_id.upper() if character.isalnum())[:14]
    precision = f"Q{bits}" if bits <= 8 else f"F{bits}"
    return f"{stem}{precision}.LITEMODEL"


def build_model(model: dict, bits: int) -> Path:
    source = BUILD / "downloads" / model["id"] / model["revision"]
    base = f"https://huggingface.co/{model['repo']}/resolve/{model['revision']}"
    model_filename = model.get("model_file", "model.safetensors")
    for filename in (model_filename, "tokenizer.json", "config.json"):
        download(f"{base}/{filename}", source / filename, bool(model.get("gated")))
    output = BUILD / "litemodels" / f"{model['id']}-{bits}.litemodel"
    converter_inputs = [
        source / model_filename, source / "tokenizer.json", source / "config.json",
        ROOT / "tools" / "convert_litemodel.py",
        ROOT / "tools" / "litemodel_common.py",
        ROOT / "tools" / "quantization.py",
        ROOT / "tools" / "architectures" / f"{model['architecture']}.py",
    ]
    if output.exists() and all(
        output.stat().st_mtime >= dependency.stat().st_mtime
        for dependency in converter_inputs
    ):
        return output
    command = [
        sys.executable, "tools/convert_litemodel.py",
        "--model", str(source / model_filename),
        "--tokenizer", str(source / "tokenizer.json"),
        "--config", str(source / "config.json"),
        "--architecture", model["architecture"], "--bits", str(bits),
        "--output", str(output),
    ]
    if model["chat"]:
        command.extend(["--chat", "--chat-style", model.get("chat_style", "banana")])
    run(command)
    return output


def write_catalog(stage_models: Path, built: list[tuple[dict, int, Path]]):
    lines = ["# id|filename|name|variant|ram_mb|mode|cpu|description"]
    for model, bits, source in built:
        filename = disk_filename(model["id"], bits)
        shutil.copy2(source, stage_models / filename)
        variant = f"Q{bits}" if bits <= 8 else f"FP{bits}"
        mode = "chat" if model["chat"] else "base"
        cpu = "legacy" if model["legacy"] else "modern"
        ram = model["quants"][str(bits)]
        lines.append(
            f"{model['id']}|{filename}|{model['name']}|{variant}|{ram}|{mode}|{cpu}|{model['description']}"
        )
    (stage_models / "CATALOG.CFG").write_text("\n".join(lines) + "\n", encoding="ascii")


def create_iso(selections: list[tuple[dict, int]], label: str,
               output: Path | None, uefi: bool):
    require("curl", "Install curl or run this builder in WSL.")
    require("make", "Install GNU make and the i686-capable GCC/binutils toolchain.")
    built = []
    for model, bits in selections:
        built.append((model, bits, build_model(model, bits)))
    if uefi:
        for tool in ("grub-mkstandalone", "mkfs.vfat", "mcopy", "mmd", "xorriso"):
            require(tool, "Install grub-efi-amd64-bin, dosfstools, mtools, and xorriso in WSL/Linux.")
        run(["make", "uefi-app"])
        stage_models = BUILD / "selected-uefi-models"
        if stage_models.exists():
            shutil.rmtree(stage_models)
        stage_models.mkdir(parents=True)
        write_catalog(stage_models, built)
        destination = output or BUILD / f"bananamind-{label}-uefi.iso"
        run([
            sys.executable, "tools/package_uefi_iso.py",
            "--app", str(BUILD / "uefi" / "BANANA.EFI"),
            "--grub", str(BUILD / "uefi" / "BOOTX64.EFI"),
            "--models", str(stage_models),
            "--work", str(BUILD / "selected-uefi-iso"),
            "--output", str(destination),
        ])
        print(f"\nReady: {destination}")
        print("UEFI GRUB loads only the frontend and catalog; weights load after selection.")
        return

    require("grub-mkrescue", "Install grub-pc-bin and xorriso (on Windows, use WSL).")
    run(["make", "kernel"])
    run(["make", "hq-assets"])
    stage = BUILD / "selected-iso"
    if stage.exists():
        shutil.rmtree(stage)
    stage_models = stage / "boot" / "models"
    stage_assets = stage / "boot" / "assets"
    (stage / "boot" / "grub").mkdir(parents=True)
    stage_models.mkdir(parents=True)
    stage_assets.mkdir(parents=True)
    shutil.copy2(BUILD / "kernel.elf", stage / "boot" / "kernel.elf")
    shutil.copy2(BUILD / "kernel-486.elf", stage / "boot" / "kernel-486.elf")
    shutil.copy2(ROOT / "grub" / "grub.cfg", stage / "boot" / "grub" / "grub.cfg")
    for asset in (BUILD / "hq-assets").glob("*.QOI"):
        shutil.copy2(asset, stage_assets / asset.name)
    write_catalog(stage_models, built)
    destination = output or BUILD / f"bananamind-{label}.iso"
    destination.parent.mkdir(parents=True, exist_ok=True)
    run(["grub-mkrescue", "-iso-level", "3", "-o", str(destination), str(stage)])
    print(f"\nReady: {destination}")
    print("GRUB loads only the selected kernel; model weights are read from the ISO after boot.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("10", "25", "100", "250"))
    parser.add_argument(
        "--models",
        help="model selections, e.g. nano-base:2,4,8,mini-chat:4",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--uefi", action="store_true", help="build the native x86-64 UEFI ISO")
    parser.add_argument("--yes", action="store_true", help="skip the size confirmation")
    args = parser.parse_args()
    registry, models = load_registry()
    if args.models:
        selections, label = parse_selections(args.models, models), "custom"
    elif args.preset:
        selections = parse_selections(",".join(registry["presets"][args.preset]), models)
        label = f"{args.preset}mb"
    else:
        selections, selected = interactive_selection(registry, models)
        label = f"{selected}mb" if selected != "custom" else "custom"
    print("\nSelected:")
    for model, bits in selections:
        print(f"  - {model['name']} ({'Q' if bits <= 8 else 'FP'}{bits})")
    if not args.yes and input("\nDownload, quantize, and build now? [y/N] ").strip().lower() != "y":
        raise SystemExit("Cancelled")
    create_iso(selections, label, args.output, args.uefi)


if __name__ == "__main__":
    main()
