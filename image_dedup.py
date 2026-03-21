#!/usr/bin/env python3
"""
Image deduplication and renaming tool.

- Scans a folder recursively for images
- Detects exact duplicates using MD5 hash
- Deletes duplicates (keeps one copy)
- Renames all remaining images to sequential numbers (001.jpg, 002.jpg, ...)
  into a flat output folder

Usage:
    python image_dedup.py <input_folder> <output_folder>

Example:
    python image_dedup.py ./my_images ./cleaned_images
"""

import os
import sys
import shutil
import hashlib
from pathlib import Path
from collections import defaultdict


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif"}


def md5_hash(filepath: Path, chunk_size: int = 65536) -> str:
    """Return MD5 hash of a file."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def find_images(root: Path) -> list[Path]:
    """Recursively find all image files under root."""
    images = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(path)
    return images


def deduplicate(images: list[Path]) -> tuple[list[Path], int]:
    """
    Remove exact duplicates by MD5 hash.
    Keeps the first occurrence (alphabetically by full path).
    Returns (unique_images, num_deleted).
    """
    seen: dict[str, Path] = {}
    duplicates: list[Path] = []

    for img in images:
        h = md5_hash(img)
        if h in seen:
            duplicates.append(img)
            print(f"  [DUPLICATE] {img}  →  duplicate of {seen[h]}")
        else:
            seen[h] = img

    # Delete duplicates
    for dup in duplicates:
        dup.unlink()
        print(f"  [DELETED]   {dup}")

    unique = list(seen.values())
    return unique, len(duplicates)


def rename_sequential(images: list[Path], output_dir: Path) -> None:
    """
    Copy unique images to output_dir with sequential names:
    001.jpg, 002.jpg, ...
    Files are sorted by their original full path before numbering.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    images_sorted = sorted(images)

    padding = len(str(len(images_sorted)))  # e.g. 3 digits for up to 999 files

    for i, src in enumerate(images_sorted, start=1):
        ext = src.suffix.lower()
        new_name = f"{str(i).zfill(padding)}{ext}"
        dst = output_dir / new_name

        # Avoid overwriting if somehow a name collision occurs
        if dst.exists():
            base = dst.stem
            counter = 1
            while dst.exists():
                dst = output_dir / f"{base}_{counter}{ext}"
                counter += 1

        shutil.copy2(src, dst)
        print(f"  [RENAMED]   {src.name}  →  {new_name}")


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    input_dir = Path(sys.argv[1]).resolve()
    output_dir = Path(sys.argv[2]).resolve()

    if not input_dir.is_dir():
        print(f"Error: input folder not found: {input_dir}")
        sys.exit(1)

    if input_dir == output_dir:
        print("Error: input and output folders must be different.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Input  : {input_dir}")
    print(f"  Output : {output_dir}")
    print(f"{'='*60}\n")

    # Step 1: Find all images
    print(">> Scanning for images...")
    images = find_images(input_dir)
    print(f"   Found {len(images)} image(s) in total.\n")

    if not images:
        print("No images found. Exiting.")
        sys.exit(0)

    # Step 2: Deduplicate
    print(">> Detecting and removing duplicates...")
    unique_images, num_deleted = deduplicate(images)
    print(f"\n   Removed {num_deleted} duplicate(s). {len(unique_images)} unique image(s) remain.\n")

    # Step 3: Rename and copy to output folder
    print(">> Renaming and copying to output folder...")
    rename_sequential(unique_images, output_dir)

    print(f"\n{'='*60}")
    print(f"  Done!")
    print(f"  Original images : {len(images)}")
    print(f"  Duplicates removed : {num_deleted}")
    print(f"  Unique images saved to output : {len(unique_images)}")
    print(f"  Output folder : {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
