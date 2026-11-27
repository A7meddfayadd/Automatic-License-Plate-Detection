"""Consistent image loading, ground-truth parsing and leakage-free splits.

Every module must load images through :func:`load_image`.  The dataset mixes
8-bit RGB, 8-bit RGBA and 16-bit RGBA PNGs, so the loader normalises all of
them to either RGB uint8 0-255 or grayscale uint8 0-255.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "ds"
SPLIT_PATH = PROJECT_ROOT / "data" / "splits" / "dev_test.json"
SEED = 42
DEV_FRACTION = 0.7


@dataclass(frozen=True)
class Sample:
    """One annotated image.

    Boxes are ``(N, 4)`` float32 arrays in ``x1, y1, x2, y2`` pixel coordinates
    at the image's native resolution.
    """

    name: str
    image_path: Path
    annotation_path: Path
    boxes: np.ndarray
    group: str


def _to_uint8_bgr(raw: np.ndarray) -> np.ndarray:
    """Normalise 8/16-bit, 3/4-channel PNG data to BGR uint8."""
    if raw.dtype == np.uint16:
        raw = (raw >> 8).astype(np.uint8)
    elif raw.dtype != np.uint8:
        raise ValueError(f"unsupported image dtype: {raw.dtype}")
    if raw.ndim == 2:
        return cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
    if raw.ndim != 3 or raw.shape[2] not in (3, 4):
        raise ValueError(f"unsupported image shape: {raw.shape}")
    return np.ascontiguousarray(raw[..., :3])


def load_image(path: str | Path, color: bool = True) -> np.ndarray:
    """Load one image as RGB or grayscale ``uint8`` in ``0-255``.

    Args:
        path: PNG file to read.
        color: return RGB (True) or grayscale (False).

    Returns:
        ``(H, W, 3)`` RGB array or ``(H, W)`` grayscale array, uint8.
    """
    path = Path(path)
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"could not read image: {path}")
    bgr = _to_uint8_bgr(raw)
    if color:
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def load_boxes(
    annotation_path: str | Path,
    image_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Read rectangle annotations from a Supervisely-style JSON file.

    Args:
        annotation_path: JSON produced by DatasetNinja.
        image_shape: optional ``(height, width)`` used to clip boxes.

    Returns:
        ``(N, 4)`` float32 array of ``x1, y1, x2, y2`` boxes; may be empty.
    """
    with open(annotation_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    width = float(payload["size"]["width"])
    height = float(payload["size"]["height"])
    if image_shape is not None:
        height, width = float(image_shape[0]), float(image_shape[1])

    boxes: list[list[float]] = []
    for obj in payload.get("objects", []):
        if obj.get("geometryType") != "rectangle":
            continue
        points = obj["points"]["exterior"]
        (xa, ya), (xb, yb) = points[0], points[1]
        x1, x2 = sorted((min(max(xa, 0.0), width), min(max(xb, 0.0), width)))
        y1, y2 = sorted((min(max(ya, 0.0), height), min(max(yb, 0.0), height)))
        if x2 - x1 >= 2.0 and y2 - y1 >= 2.0:
            boxes.append([x1, y1, x2, y2])
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def image_digest(path: str | Path) -> str:
    """Hash the normalised grayscale pixels to detect duplicate scenes."""
    gray = load_image(path, color=False)
    hasher = hashlib.md5()
    hasher.update(str(gray.shape).encode("ascii"))
    hasher.update(gray.tobytes())
    return hasher.hexdigest()


def discover_samples(raw_dir: str | Path = RAW_DIR) -> list[Sample]:
    """Load every image/annotation pair under ``raw_dir``.

    The dataset uses ``img/`` and ``ann/`` subdirectories with matching
    ``<name>.png`` / ``<name>.png.json`` files.
    """
    raw_dir = Path(raw_dir)
    image_dir = raw_dir / "img"
    annotation_dir = raw_dir / "ann"
    if not image_dir.is_dir() or not annotation_dir.is_dir():
        raise FileNotFoundError(
            f"expected {image_dir} and {annotation_dir}; run data/get_dataset.py first"
        )

    samples: list[Sample] = []
    for image_path in sorted(image_dir.glob("*.png")):
        annotation_path = annotation_dir / f"{image_path.name}.json"
        if not annotation_path.is_file():
            raise FileNotFoundError(f"missing annotation for {image_path.name}")
        boxes = load_boxes(annotation_path)
        samples.append(
            Sample(
                name=image_path.name,
                image_path=image_path,
                annotation_path=annotation_path,
                boxes=boxes,
                group=image_digest(image_path),
            )
        )
    if not samples:
        raise RuntimeError(f"no images found in {image_dir}")
    return samples


def group_samples(samples: list[Sample]) -> dict[str, list[Sample]]:
    """Group images whose normalised pixels are identical."""
    groups: dict[str, list[Sample]] = {}
    for sample in samples:
        groups.setdefault(sample.group, []).append(sample)
    for members in groups.values():
        members.sort(key=lambda item: item.name)
    return dict(sorted(groups.items()))


def canonical_samples(samples: list[Sample]) -> list[Sample]:
    """Return one deterministic representative per duplicate group."""
    representatives = [members[0] for members in group_samples(samples).values()]
    return sorted(representatives, key=lambda item: item.name)


def build_split(
    samples: list[Sample],
    seed: int = SEED,
    dev_fraction: float = DEV_FRACTION,
) -> dict:
    """Split duplicate groups, not files, into development and test sets.

    Duplicate copies carry slightly different boxes and must never straddle the
    split.  The canonical copy (lexicographically first name) represents its
    group in evaluation; the other copies are listed under ``duplicates``.
    """
    if not 0.0 < dev_fraction < 1.0:
        raise ValueError("dev_fraction must be in (0, 1)")

    groups = group_samples(samples)
    group_ids = sorted(groups)
    random.Random(seed).shuffle(group_ids)
    n_dev = int(round(dev_fraction * len(group_ids)))
    dev_ids = set(group_ids[:n_dev])

    split = {
        "seed": seed,
        "dev_fraction": dev_fraction,
        "split_level": "duplicate_image_group",
        "n_files": len(samples),
        "n_unique_scenes": len(group_ids),
        "dev": sorted(
            members[0].name for gid, members in groups.items() if gid in dev_ids
        ),
        "test": sorted(
            members[0].name for gid, members in groups.items() if gid not in dev_ids
        ),
        "duplicates": {
            members[0].name: [item.name for item in members[1:]]
            for members in groups.values()
            if len(members) > 1
        },
    }
    if set(split["dev"]) & set(split["test"]):
        raise AssertionError("a duplicate group leaked across the split")
    return split


def ensure_split(
    samples: list[Sample] | None = None,
    seed: int = SEED,
    dev_fraction: float = DEV_FRACTION,
    split_path: str | Path = SPLIT_PATH,
) -> dict:
    """Create the split file once and reuse it on later runs."""
    split_path = Path(split_path)
    if split_path.is_file():
        with open(split_path, encoding="utf-8") as handle:
            split = json.load(handle)
        if split.get("seed") != seed or split.get("dev_fraction") != dev_fraction:
            raise ValueError(f"{split_path} was built with different settings")
        return split

    if samples is None:
        samples = discover_samples()
    split = build_split(samples, seed=seed, dev_fraction=dev_fraction)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with open(split_path, "w", encoding="utf-8") as handle:
        json.dump(split, handle, indent=2)
    return split


def get_split_samples(
    split_name: str,
    samples: list[Sample] | None = None,
    split: dict | None = None,
) -> list[Sample]:
    """Return canonical samples belonging to ``dev`` or ``test``."""
    if split_name not in ("dev", "test"):
        raise ValueError("split_name must be 'dev' or 'test'")
    if samples is None:
        samples = discover_samples()
    if split is None:
        split = ensure_split(samples)
    names = set(split[split_name])
    selected = [sample for sample in samples if sample.name in names]
    if len(selected) != len(names):
        raise AssertionError("split names do not match the dataset")
    return selected
