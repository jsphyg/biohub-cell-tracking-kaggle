"""Reference implementation for the Kaggle notebook baseline.

The hosted notebook is self-contained because Kaggle notebooks do not import
files from this repository automatically. Keep changes synchronized manually
until we decide whether to package this as a Kaggle Dataset.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.spatial import cKDTree


def locate_competition_root(candidates: list[str] | None = None) -> Path:
    candidates = candidates or [
        "/kaggle/input/biohub-cell-tracking-during-development",
        "/kaggle/input/competitions/biohub-cell-tracking-during-development",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if (path / "test").exists():
            return path
    raise FileNotFoundError("Could not find the competition input. Attach the competition data first.")


def detect_cells_3d(image: np.ndarray, sigma=(1.5, 2.0, 2.0), threshold=0.08, min_distance=3):
    """Return approximate (z, y, x) centroids for bright 3D blobs."""
    image = image.astype(np.float32, copy=False)
    lo, hi = np.percentile(image, [1, 99.8])
    if hi <= lo:
        return np.empty((0, 3), dtype=np.float32)
    x = np.clip((image - lo) / (hi - lo), 0, 1)
    small = gaussian_filter(x, sigma=sigma)
    large = gaussian_filter(x, sigma=tuple(s * 2.0 for s in sigma))
    dog = small - large
    local = maximum_filter(dog, size=2 * min_distance + 1)
    cutoff = max(float(np.quantile(dog, 0.995)), threshold * float(dog.max()))
    mask = (dog == local) & (dog >= cutoff)
    coords = np.argwhere(mask).astype(np.float32)
    return coords


def link_frames(prev_xyz, curr_xyz, scale=(1.625, 0.40625, 0.40625), max_distance=15.0):
    """Greedy one-to-one links in physical micrometer coordinates."""
    if len(prev_xyz) == 0 or len(curr_xyz) == 0:
        return []
    prev_phys = np.asarray(prev_xyz) * np.asarray(scale)
    curr_phys = np.asarray(curr_xyz) * np.asarray(scale)
    tree = cKDTree(curr_phys)
    distances, indices = tree.query(prev_phys, distance_upper_bound=max_distance)
    candidates = sorted(
        ((float(d), i, int(j)) for i, (d, j) in enumerate(zip(distances, indices)) if np.isfinite(d)),
        key=lambda x: x[0],
    )
    used = set()
    links = []
    for _, i, j in candidates:
        if j not in used:
            used.add(j)
            links.append((i, j))
    return links


def build_submission(dataset_name, frames, detector=detect_cells_3d):
    nodes, edges = [], []
    next_id = 1
    previous = []
    for t, image in enumerate(frames):
        coords = detector(image)
        current = []
        for local_index, (z, y, x) in enumerate(coords):
            node_id = next_id
            next_id += 1
            nodes.append({"dataset": dataset_name, "row_type": "node", "node_id": node_id,
                          "t": t, "z": round(z), "y": round(y), "x": round(x),
                          "source_id": -1, "target_id": -1})
            current.append((local_index, node_id, (z, y, x)))
        if previous and current:
            links = link_frames([p[2] for p in previous], [c[2] for c in current])
            for i, j in links:
                edges.append({"dataset": dataset_name, "row_type": "edge", "node_id": -1,
                              "t": -1, "z": -1, "y": -1, "x": -1,
                              "source_id": previous[i][1], "target_id": current[j][1]})
        previous = current
    columns = ["dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]
    result = pd.DataFrame(nodes + edges, columns=columns)
    result.insert(0, "id", np.arange(len(result), dtype=np.int64))
    return result

