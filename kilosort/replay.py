"""Immutable center capture and replay for clustering experiments.

Replay artifacts are development fixtures, never production sorter inputs. The
loader verifies every array and the canonical metadata hash before returning a
capture. ``replay_center`` mirrors the center-local branch of
``clustering_qr.run`` after ``get_data_cpu``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from kilosort import clustering_qr, hierarchical, swarmsplitter


SCHEMA = "kilosort.center-replay.v1"


@dataclass(frozen=True)
class CenterCapture:
    Xd: torch.Tensor
    igood: np.ndarray
    ichan: np.ndarray
    st0: np.ndarray | None
    metadata: dict[str, Any]
    manifest_sha256: str


@dataclass(frozen=True)
class CenterReplayResult:
    labels: np.ndarray
    Wall: torch.Tensor
    initial_labels: np.ndarray | None
    graph: Any | None
    tree: np.ndarray | None
    tree_stats: np.ndarray | None


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _array_record(path: Path) -> dict[str, Any]:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "shape": list(array.shape),
        "dtype": array.dtype.str,
    }


def _write_array(path: Path, value: np.ndarray) -> dict[str, Any]:
    np.save(path, np.ascontiguousarray(value), allow_pickle=False)
    return _array_record(path)


def save_center_capture(
    directory: str | Path,
    *,
    Xd: torch.Tensor,
    igood: np.ndarray,
    ichan: np.ndarray,
    st0: np.ndarray | None,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Save one center capture and return its sealed manifest."""

    target = Path(directory)
    if not isinstance(Xd, torch.Tensor) or Xd.ndim != 2 or Xd.device.type != "cpu":
        raise ValueError("Xd must be a two-dimensional CPU tensor")
    igood = np.asarray(igood)
    ichan = np.asarray(ichan)
    if igood.ndim != 1 or igood.size != Xd.shape[0]:
        raise ValueError("igood must be one-dimensional and align with Xd rows")
    if ichan.ndim != 1:
        raise ValueError("ichan must be one-dimensional")
    if st0 is not None:
        st0 = np.asarray(st0)
        if st0.ndim != 1 or st0.size != Xd.shape[0]:
            raise ValueError("st0 must be one-dimensional and align with Xd rows")

    target.mkdir(parents=True, exist_ok=False)
    arrays = {
        "Xd": _write_array(target / "Xd.npy", Xd.detach().numpy()),
        "igood": _write_array(target / "igood.npy", igood),
        "ichan": _write_array(target / "ichan.npy", ichan),
    }
    if st0 is not None:
        arrays["st0"] = _write_array(target / "st0.npy", st0)
    manifest = {
        "schema": SCHEMA,
        "metadata": dict(metadata),
        "arrays": arrays,
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        _canonical_bytes(manifest)
    ).hexdigest()
    (target / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    return manifest


def load_center_capture(
    directory: str | Path,
    *,
    expected_metadata: Mapping[str, Any] | None = None,
) -> CenterCapture:
    """Load a capture only after all content and metadata identities match."""

    source = Path(directory)
    try:
        manifest = json.loads((source / "manifest.json").read_text())
    except Exception as exc:
        raise ValueError(f"cannot read center replay manifest: {source}") from exc
    recorded_seal = manifest.pop("manifest_sha256", None)
    actual_seal = hashlib.sha256(_canonical_bytes(manifest)).hexdigest()
    if recorded_seal != actual_seal:
        raise ValueError("center replay manifest seal mismatch")
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"unsupported center replay schema: {manifest.get('schema')!r}")
    if expected_metadata is not None and manifest.get("metadata") != dict(expected_metadata):
        raise ValueError("center replay metadata mismatch")

    loaded = {}
    for name, record in manifest.get("arrays", {}).items():
        path = source / record.get("file", "")
        current = _array_record(path)
        if current != record:
            raise ValueError(f"center replay array identity mismatch: {name}")
        # Copy-on-write mappings stay disk-immutable while remaining writable
        # enough for safe zero-copy conversion to Torch.
        loaded[name] = np.load(path, mmap_mode="c", allow_pickle=False)
    for required in ("Xd", "igood", "ichan"):
        if required not in loaded:
            raise ValueError(f"center replay manifest is missing {required}")
    if loaded["Xd"].ndim != 2 or loaded["igood"].shape != (loaded["Xd"].shape[0],):
        raise ValueError("center replay arrays do not reconcile")
    if "st0" in loaded and loaded["st0"].shape != (loaded["Xd"].shape[0],):
        raise ValueError("center replay st0 does not reconcile")

    return CenterCapture(
        Xd=torch.from_numpy(loaded["Xd"]),
        igood=np.asarray(loaded["igood"]),
        ichan=np.asarray(loaded["ichan"]),
        st0=np.asarray(loaded["st0"]) if "st0" in loaded else None,
        metadata=dict(manifest["metadata"]),
        manifest_sha256=str(recorded_seal),
    )


def replay_center(
    capture: CenterCapture,
    *,
    settings: Mapping[str, Any],
    n_channels: int,
    n_pcs: int,
    device: torch.device = torch.device("cpu"),
) -> CenterReplayResult:
    """Replay clustering, tree splitting, and template reconstruction."""

    Xd = capture.Xd
    if Xd.shape[0] < 1000:
        labels = np.zeros(Xd.shape[0], dtype=np.int32)
        initial = graph = tree = tree_stats = None
    else:
        labels, leaf_labels, graph, initial = clustering_qr.cluster(
            Xd,
            nskip=int(settings["cluster_downsampling"]),
            n_neigh=int(settings.get("cluster_neighbors", 10)),
            max_sub=settings.get("max_cluster_subset"),
            lam=1,
            seed=int(settings.get("cluster_init_seed", 1)),
            device=device,
            niter=int(settings.get("cluster_iters", clustering_qr.CLUSTER_ITERS)),
        )
        tree, tree_stats, members = hierarchical.maketree(
            graph, labels, leaf_labels
        )
        tree, tree_stats = swarmsplitter.split(
            Xd.numpy(),
            tree,
            tree_stats,
            labels,
            members,
            meta=capture.st0,
            split_ccg_threshold=float(
                settings.get(
                    "split_ccg_threshold", swarmsplitter.SPLIT_CCG_THRESHOLD
                )
            ),
            refrac_veto=bool(settings.get("refractory_merge_veto", True)),
            refrac_veto_ratio=float(
                settings.get(
                    "refractory_veto_ratio", swarmsplitter.REFRAC_VETO_RATIO
                )
            ),
            refrac_veto_alpha=float(
                settings.get(
                    "refractory_veto_alpha", swarmsplitter.REFRAC_VETO_ALPHA
                )
            ),
        )
        labels = swarmsplitter.new_clusters(
            labels, members, tree, tree_stats
        )
        initial = (
            initial.cpu().numpy() if hasattr(initial, "cpu") else np.asarray(initial)
        )

    Wall = clustering_qr.mean_cluster_templates(
        Xd, labels, capture.ichan, n_channels, n_pcs
    )
    return CenterReplayResult(
        labels=labels,
        Wall=Wall,
        initial_labels=initial,
        graph=graph,
        tree=tree,
        tree_stats=tree_stats,
    )


__all__ = [
    "CenterCapture",
    "CenterReplayResult",
    "load_center_capture",
    "replay_center",
    "save_center_capture",
]
