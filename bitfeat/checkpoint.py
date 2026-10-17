"""Training checkpoints: atomic save, retention, and exact resume.

A checkpoint stores everything needed to continue a run as if it had never
stopped: model + optimizer + GradScaler state, the number of completed
optimizer steps, the resolved config, and the Python / NumPy / torch (CPU and
CUDA) RNG states. Combined with the step-indexed sampler in ``bitfeat.train``
this makes ``--resume`` pick up the exact data order and loss sampling.

Layout inside a run directory::

    <output_dir>/ckpt_step00001000.pt   # periodic, only the last K are kept
    <output_dir>/final.pt               # written once when max_steps is reached
"""

from __future__ import annotations

import os
import random
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

CHECKPOINT_FORMAT = 1
FINAL_NAME = "final.pt"
_CKPT_RE = re.compile(r"^ckpt_step(\d+)\.pt$")


def checkpoint_path(output_dir: str | Path, step: int) -> Path:
    return Path(output_dir) / f"ckpt_step{int(step):08d}.pt"


def list_checkpoints(output_dir: str | Path) -> List[Tuple[int, Path]]:
    """Periodic checkpoints in ``output_dir`` as ``(step, path)``, oldest first."""
    d = Path(output_dir)
    if not d.is_dir():
        return []
    found = []
    for p in d.iterdir():
        m = _CKPT_RE.match(p.name)
        if m and p.is_file():
            found.append((int(m.group(1)), p))
    return sorted(found)


def _torch_load(path: str | Path, map_location: Any = "cpu") -> Dict[str, Any]:
    # Checkpoints contain only tensors and plain Python types, so the safe
    # weights_only unpickler is enough.
    return torch.load(str(path), map_location=map_location, weights_only=True)


def latest_checkpoint(output_dir: str | Path) -> Optional[Path]:
    """Return the checkpoint with the highest step (periodic or final), if any."""
    cands = list_checkpoints(output_dir)
    best: Optional[Tuple[int, Path]] = cands[-1] if cands else None
    final = Path(output_dir) / FINAL_NAME
    if final.is_file():
        step = int(_torch_load(final).get("step", -1))
        if best is None or step >= best[0]:
            best = (step, final)
    return best[1] if best else None


def capture_rng_state() -> Dict[str, Any]:
    # Stored as tensors / plain Python types so the whole checkpoint loads with
    # torch.load(weights_only=True) (as bitfeat.extract does).
    name, keys, pos, has_gauss, cached = np.random.get_state()
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": {
            "name": str(name),
            "keys": torch.from_numpy(np.asarray(keys, dtype=np.int64)),
            "pos": int(pos),
            "has_gauss": int(has_gauss),
            "cached_gaussian": float(cached),
        },
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Optional[Dict[str, Any]]) -> None:
    if not state:
        return
    if "python" in state:
        py = state["python"]
        random.setstate((py[0], tuple(py[1]), py[2]))
    if "numpy" in state:
        ns = state["numpy"]
        np.random.set_state(
            (
                ns["name"],
                ns["keys"].cpu().numpy().astype(np.uint32),
                ns["pos"],
                ns["has_gauss"],
                ns["cached_gaussian"],
            )
        )
    if "torch" in state:
        torch.set_rng_state(state["torch"].cpu())
    cuda_states = state.get("cuda")
    if cuda_states and torch.cuda.is_available():
        if len(cuda_states) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all([s.cpu() for s in cuda_states])


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    step: int,
    config: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Atomically write a checkpoint (tmp file + rename, so a kill never leaves a torn file)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "step": int(step),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "config": config,
        "rng": capture_rng_state(),
        "torch_version": str(torch.__version__),
        "extra": extra or {},
    }
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, str(tmp))
    os.replace(tmp, path)
    return path


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Any = None,
    map_location: Any = "cpu",
    restore_rng: bool = True,
    validate: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Restore state in place and return the raw checkpoint dict (``step``, ``config``, ...).

    ``validate`` is called with the raw checkpoint before anything is restored
    (raise from it to reject e.g. an incompatible config).
    """
    ckpt = _torch_load(path, map_location=map_location)
    if validate is not None:
        validate(ckpt)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])
    if restore_rng:
        restore_rng_state(ckpt.get("rng"))
    return ckpt


def prune_checkpoints(output_dir: str | Path, keep_last: int) -> List[Path]:
    """Delete all but the newest ``keep_last`` periodic checkpoints (final.pt is never touched)."""
    if keep_last <= 0:
        return []
    removed = []
    for _, p in list_checkpoints(output_dir)[:-keep_last]:
        try:
            p.unlink()
            removed.append(p)
        except FileNotFoundError:
            pass
    return removed
