#!/usr/bin/env python3
"""Update the compiled build number for a developer checkout."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def write_if_changed(path: Path, contents: str) -> None:
    if path.is_file() and path.read_text(encoding="ascii") == contents:
        return
    path.write_text(contents, encoding="ascii")


def update_build_number(root: Path) -> int:
    number_path = root / "BUILD_NUMBER"
    try:
        number = int(number_path.read_text(encoding="ascii").strip())
    except (OSError, ValueError) as error:
        raise SystemExit(f"Invalid or missing {number_path}") from error

    if (root / "dev_folder").is_dir():
        number += 1
        write_if_changed(number_path, f"{number}\n")

    header = (
        "#ifndef BANANAMIND_BUILD_NUMBER_H\n"
        "#define BANANAMIND_BUILD_NUMBER_H\n\n"
        f"#define BANANAMIND_BUILD_NUMBER {number}u\n\n"
        "#endif\n"
    )
    write_if_changed(root / "include" / "build_number.h", header)
    return number


if __name__ == "__main__":
    print(update_build_number(ROOT))
