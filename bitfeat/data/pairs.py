"""MegaDepth ImagePairDataset (train_local npz + h5) and SyntheticPairDataset.

Expected layout under ``root`` (default ``/workspace/datasets/megadepth``)::

    megadepth1500/images/{0015,0022}/*.jpg
    megadepth1500/depths/{0015,0022}/*.h5
    Undistorted_SfM/{0015,0022}/{images,depths}   # symlinks
    index/train_local/*.npz                 # preferred train index

``train_local`` npz keys: image_paths, depth_paths, intrinsics, poses, pair_infos
where each pair is ``(array([i, j]), overlap)``. Paths are relative to root.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _as_float32(x: Any) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x, dtype=np.float32), dtype=torch.float32)


def _relative_pose(pose0: np.ndarray, pose1: np.ndarray) -> np.ndarray:
    """T_0to1 from world-to-camera poses (4x4)."""
    T0 = np.asarray(pose0, dtype=np.float64)
    T1 = np.asarray(pose1, dtype=np.float64)
    return (T1 @ np.linalg.inv(T0)).astype(np.float32)


def _load_rgb(path: Path, image_size: Tuple[int, int]) -> Tuple[torch.Tensor, Tuple[int, int]]:
    with Image.open(path) as im:
        im = im.convert("RGB")
        ow, oh = im.size
        h, w = image_size
        im = im.resize((w, h), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous(), (oh, ow)


def _load_depth_h5(path: Optional[Path], image_size: Tuple[int, int]) -> Optional[torch.Tensor]:
    if path is None or not Path(path).is_file():
        return None
    try:
        import h5py
    except ImportError as e:
        raise ImportError("h5py is required to load MegaDepth depths: pip install h5py") from e
    try:
        with h5py.File(path, "r") as f:
            depth = np.asarray(f["depth"] if "depth" in f else f[list(f.keys())[0]], dtype=np.float32)
    except OSError:
        return None
    if depth.ndim == 3:
        depth = depth.squeeze()
    h, w = image_size
    pil = Image.fromarray(depth, mode="F").resize((w, h), Image.NEAREST)
    return torch.from_numpy(np.asarray(pil, dtype=np.float32).copy()).unsqueeze(0)


def _scale_K(K: np.ndarray, orig_hw: Tuple[int, int], new_hw: Tuple[int, int]) -> np.ndarray:
    oh, ow = orig_hw
    nh, nw = new_hw
    sx, sy = nw / float(ow), nh / float(oh)
    K2 = np.asarray(K, dtype=np.float32).copy()
    K2[0, 0] *= sx
    K2[0, 2] *= sx
    K2[1, 1] *= sy
    K2[1, 2] *= sy
    return K2


def _resolve_under_root(root: Path, rel: Optional[str]) -> Optional[Path]:
    """Resolve scene_info path relative to root, with Undistorted_SfM / megadepth1500 fallbacks."""
    if rel is None:
        return None
    if isinstance(rel, float) and np.isnan(rel):
        return None
    rel_s = str(rel).strip()
    if not rel_s or rel_s == "None":
        return None
    direct = root / rel_s
    if direct.is_file():
        return direct
    name = Path(rel_s).name
    parts = Path(rel_s).parts
    scene = None
    for i, p in enumerate(parts):
        if re.fullmatch(r"\d{4}", p):
            scene = p
            break
        if p == "Undistorted_SfM" and i + 1 < len(parts):
            scene = parts[i + 1]
            break
    if scene is None:
        return None
    candidates = [
        root / "megadepth1500" / "images" / scene / name,
        root / "megadepth1500" / "depths" / scene / name,
        root / "Undistorted_SfM" / scene / "images" / name,
        root / "Undistorted_SfM" / scene / "depths" / name,
        root / "images" / scene / name,
        root / "depths" / scene / name,
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def _parse_pair_entry(entry: Any) -> Tuple[int, int, float]:
    """Normalize pair_infos entry to (i0, i1, overlap)."""
    idx = entry[0]
    overlap = float(entry[1])
    i0, i1 = int(idx[0]), int(idx[1])
    return i0, i1, overlap


class ImagePairDataset(Dataset):
    """MegaDepth-style overlapping image pairs with depth + pose.

    Returns:
      image0/1 (3,H,W), depth0/1 (1,H,W), K0/K1 (3,3), T_0to1 (4,4),
      has_depth (bool), pose (=T_0to1 alias).
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        image_size: Tuple[int, int] = (480, 480),
        scene_info_dir: Optional[str] = None,
        index_dir: Optional[str] = None,
        min_overlap: float = 0.1,
        max_overlap: float = 0.7,
        max_pairs_per_scene: Optional[int] = None,
        require_depth: bool = False,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.min_overlap = min_overlap
        self.max_overlap = max_overlap
        self.require_depth = require_depth
        self.samples: List[Dict[str, Any]] = []

        if not self.root.is_dir():
            return

        npz_dirs: List[Path] = []
        for d in (index_dir, scene_info_dir):
            if d:
                p = Path(d)
                if not p.is_absolute():
                    p = self.root / p
                npz_dirs.append(p)
        npz_dirs.extend(
            [
                self.root / "index" / "train_local",
                self.root / "train_local",
                self.root / "index" / "megadepth_test_1500_scene_info",
                self.root / "megadepth_test_1500_scene_info",
                self.root / "scene_info",
            ]
        )

        for d in npz_dirs:
            if d.is_dir() and any(d.glob("*.npz")):
                before = len(self.samples)
                self._load_scene_info_npz(d)
                if len(self.samples) > before:
                    break

        if max_pairs_per_scene is not None and max_pairs_per_scene > 0:
            self.samples = self.samples[: int(max_pairs_per_scene) * 64]

    def _load_scene_info_npz(self, directory: Path) -> None:
        for npz_path in sorted(directory.glob("*.npz")):
            data = np.load(str(npz_path), allow_pickle=True)
            image_paths = data["image_paths"]
            depth_paths = data["depth_paths"] if "depth_paths" in data.files else None
            intrinsics = data["intrinsics"]
            poses = data["poses"]
            pair_infos = list(data["pair_infos"])
            for entry in pair_infos:
                i0, i1, overlap = _parse_pair_entry(entry)
                if overlap < self.min_overlap or overlap > self.max_overlap:
                    continue
                if image_paths[i0] is None or image_paths[i1] is None:
                    continue
                if poses[i0] is None or poses[i1] is None:
                    continue
                if intrinsics[i0] is None or intrinsics[i1] is None:
                    continue
                img0 = _resolve_under_root(self.root, image_paths[i0])
                img1 = _resolve_under_root(self.root, image_paths[i1])
                if img0 is None or img1 is None:
                    continue
                d0 = d1 = None
                if depth_paths is not None:
                    d0 = _resolve_under_root(self.root, depth_paths[i0])
                    d1 = _resolve_under_root(self.root, depth_paths[i1])
                if self.require_depth and (d0 is None or d1 is None):
                    continue
                self.samples.append(
                    {
                        "image0": img0,
                        "image1": img1,
                        "depth0": d0,
                        "depth1": d1,
                        "K0": np.asarray(intrinsics[i0], dtype=np.float32),
                        "K1": np.asarray(intrinsics[i1], dtype=np.float32),
                        "pose0": np.asarray(poses[i0], dtype=np.float32),
                        "pose1": np.asarray(poses[i1], dtype=np.float32),
                        "T_0to1": None,
                        "overlap": overlap,
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        s = self.samples[index]
        h, w = self.image_size
        img0, orig0 = _load_rgb(s["image0"], self.image_size)
        img1, orig1 = _load_rgb(s["image1"], self.image_size)
        K0 = _scale_K(s["K0"], orig0, self.image_size)
        K1 = _scale_K(s["K1"], orig1, self.image_size)
        if s["T_0to1"] is not None:
            T_0to1 = np.asarray(s["T_0to1"], dtype=np.float32)
        else:
            T_0to1 = _relative_pose(s["pose0"], s["pose1"])

        depth0 = _load_depth_h5(s["depth0"], self.image_size)
        depth1 = _load_depth_h5(s["depth1"], self.image_size)
        has_depth = depth0 is not None and depth1 is not None
        if depth0 is None:
            depth0 = torch.zeros(1, h, w)
        if depth1 is None:
            depth1 = torch.zeros(1, h, w)

        T = _as_float32(T_0to1)
        return {
            "image0": img0,
            "image1": img1,
            "depth0": depth0,
            "depth1": depth1,
            "K0": _as_float32(K0),
            "K1": _as_float32(K1),
            "T_0to1": T,
            "pose": T.clone(),
            "has_depth": torch.tensor(has_depth, dtype=torch.bool),
        }


class SyntheticPairDataset(Dataset):
    """Random RGB pairs with identity pose for smoke training / CI."""

    def __init__(
        self,
        length: int = 32,
        image_size: Tuple[int, int] = (480, 480),
        seed: int = 0,
    ) -> None:
        self.length = length
        self.image_size = image_size
        self.seed = seed

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Dict[str, Any]:
        g = torch.Generator().manual_seed(self.seed + index)
        h, w = self.image_size
        img0 = torch.rand(3, h, w, generator=g)
        noise = 0.05 * torch.randn(3, h, w, generator=g)
        img1 = (img0 + noise).clamp(0.0, 1.0)
        T = torch.eye(4)
        K = torch.tensor(
            [[w * 0.9, 0.0, w / 2.0], [0.0, h * 0.9, h / 2.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        depth = torch.ones(1, h, w) * 5.0
        return {
            "image0": img0,
            "image1": img1,
            "pose": T,
            "T_0to1": T.clone(),
            "K0": K,
            "K1": K.clone(),
            "depth0": depth,
            "depth1": depth.clone(),
            "has_depth": torch.tensor(True),
        }
