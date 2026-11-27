"""Prepare the dataset layout expected by ``src/data.py``.

Usage:
    python data/get_dataset.py --archive path/to/car-license-plate-DatasetNinja.tar

The archive is extracted to ``data/raw`` so that ``data/raw/ds/img`` and
``data/raw/ds/ann`` exist.  The script uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import sys
import tarfile
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST = PROJECT_ROOT / "data" / "raw"


def safe_members(members, destination: Path):
    """Reject archive members that would escape the destination directory."""
    destination = destination.resolve()
    for member in members:
        target = (destination / member.name).resolve()
        if destination not in target.parents and target != destination:
            raise RuntimeError(f"unsafe archive member: {member.name}")
        yield member


def extract(archive: Path, destination: Path) -> None:
    """Extract a ``.tar``, ``.tar.gz``, ``.tgz`` or ``.zip`` archive."""
    suffix = archive.name.lower()
    destination.mkdir(parents=True, exist_ok=True)
    if suffix.endswith((".tar", ".tar.gz", ".tgz")):
        with tarfile.open(archive, "r:*") as handle:
            handle.extractall(destination, members=safe_members(handle, destination))
    elif suffix.endswith(".zip"):
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(destination)
    else:
        raise ValueError(f"unsupported archive type: {archive.name}")


def check_layout(destination: Path) -> bool:
    """Verify the DatasetNinja directory layout and print counts."""
    image_dir = destination / "ds" / "img"
    annotation_dir = destination / "ds" / "ann"
    fine = True
    if not image_dir.is_dir():
        print(f"missing directory: {image_dir}")
        fine = False
    if not annotation_dir.is_dir():
        print(f"missing directory: {annotation_dir}")
        fine = False
    if not fine:
        return False
    n_images = len(list(image_dir.glob("*.png")))
    n_annotations = len(list(annotation_dir.glob("*.json")))
    print(f"images: {n_images}  annotations: {n_annotations}")
    if n_images == 0 or n_images != n_annotations:
        print("warning: image and annotation counts do not match")
        return False
    print(f"dataset ready under {destination}")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="downloaded tar or zip archive")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    args = parser.parse_args(argv)

    if args.archive:
        if not args.archive.is_file():
            print(f"archive not found: {args.archive}", file=sys.stderr)
            return 1
        extract(args.archive, args.dest)

    if check_layout(args.dest):
        return 0

    print(
        "\nDownload the Car License Plate dataset (CC0 1.0):\n"
        "  https://www.kaggle.com/datasets/andrewmvd/car-plate-detection\n"
        "or the DatasetNinja conversion named car-license-plate-detection,\n"
        "then rerun:\n"
        "  python data/get_dataset.py --archive <archive>\n"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
