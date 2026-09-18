"""Export a completed Kilosort run as a Vision-native bundle.

This module deliberately lives in Kilosort: the EI is computed from the
Kilosort spike arrays and the native Litke reader, so an export cannot
accidentally attach a different sort's spikes to a Vision analysis.  Vision's
optional writer dependency is loaded only when :func:`export_vision` is used.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Set, Tuple

import numpy as np
from numba import njit, prange

from .litke import LitkeRecording


class VisionExportError(ValueError):
    """An export precondition or output validation failed."""


EI_MODES = ("exact", "accurate", "truncated64")
_ALPHA_RATE = 20000.0
_EI_BLOCK = 8


def _quality_ids(path: Path) -> Set[int]:
    if not path.is_file():
        raise VisionExportError(f"missing Kilosort quality table: {path}")
    good: Set[int] = set()
    with path.open("r", encoding="utf-8") as fh:
        header = fh.readline().strip().split("\t")
        if len(header) < 2 or header[0] != "cluster_id":
            raise VisionExportError(f"invalid Kilosort quality table: {path}")
        for line in fh:
            fields = line.rstrip("\n").split("\t")
            if len(fields) >= 2 and fields[1].strip().lower() == "good":
                good.add(int(fields[0]))
    return good


def load_kilosort_spikes(
    sort_dir: os.PathLike, good_only: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return one-based Vision IDs, sorted sample times, and dense cell IDs."""
    sort_dir = Path(sort_dir)
    try:
        times = np.asarray(np.load(sort_dir / "spike_times.npy")).reshape(-1)
        clusters = np.asarray(np.load(sort_dir / "spike_clusters.npy")).reshape(-1)
    except FileNotFoundError as exc:
        raise VisionExportError(f"missing Kilosort spike array: {exc.filename}") from exc
    if times.size == 0:
        raise VisionExportError("Kilosort spike arrays are empty")
    if times.size != clusters.size:
        raise VisionExportError("spike_times and spike_clusters lengths differ")
    if (not np.all(np.isfinite(times)) or not np.all(np.isfinite(clusters)) or
            np.any(times != np.floor(times)) or np.any(clusters != np.floor(clusters))):
        raise VisionExportError("Kilosort spike arrays must contain finite integers")
    times = times.astype(np.int64, copy=False)
    clusters = clusters.astype(np.int64, copy=False)
    if np.any(times < 0) or np.any(clusters < 0):
        raise VisionExportError("Kilosort spike times and IDs must be non-negative")
    if good_only:
        good = _quality_ids(sort_dir / "cluster_KSLabel.tsv")
        keep = np.isin(clusters, np.fromiter(good, dtype=np.int64, count=len(good)))
        times, clusters = times[keep], clusters[keep]
        if times.size == 0:
            raise VisionExportError("good-only export contains no spikes")
    ids = np.unique(clusters)
    dense = np.searchsorted(ids, clusters).astype(np.int32, copy=False)
    order = np.lexsort((times, dense))
    return ids + 1, times[order], dense[order]


@njit(inline="always")
def _java_float_to_short(value):
    if value != value:
        integer = 0
    elif value >= 2147483647.0:
        integer = 2147483647
    elif value <= -2147483648.0:
        integer = -2147483648
    else:
        integer = int(value)
    value = integer & 0xFFFF
    if value >= 32768:
        value -= 65536
    return np.int16(value)


@njit(parallel=True, cache=True)
def _filter_chunk(raw, mean, alpha, out, offset, truncate):
    n_time, n_electrodes = raw.shape
    n_blocks = (n_electrodes + _EI_BLOCK - 1) // _EI_BLOCK
    for block in prange(n_blocks):
        first = block * _EI_BLOCK
        last = min(first + _EI_BLOCK, n_electrodes)
        for electrode in range(first, last):
            baseline = mean[electrode]
            for sample in range(n_time):
                value = np.float32(raw[sample, electrode])
                delta = np.float32(value - baseline)
                baseline = np.float32(
                    np.float64(baseline) + alpha * np.float64(delta))
                residual = np.float32(value - baseline)
                out[electrode, offset + sample] = (
                    _java_float_to_short(residual) if truncate else residual)
            mean[electrode] = baseline


@njit(parallel=True, cache=True)
def _accumulate(buf, bases, cells, avg, err, bounds, exact):
    n_electrodes = buf.shape[0]
    n_points = avg.shape[2]
    for electrode in prange(n_electrodes):
        for block in range(bounds.shape[0] - 1):
            for k in range(bounds[block], bounds[block + 1]):
                cell = cells[k]
                base = bases[k]
                for point in range(n_points):
                    value = buf[electrode, base + point]
                    if exact:
                        avg[electrode, cell, point] = np.float32(
                            avg[electrode, cell, point] + np.float32(value))
                        ivalue = np.int32(value)
                        err[electrode, cell, point] = np.float32(
                            err[electrode, cell, point] + np.float32(ivalue * ivalue))
                    else:
                        fvalue = np.float64(value)
                        avg[electrode, cell, point] += fvalue
                        err[electrode, cell, point] += fvalue * fvalue


@njit(parallel=True, cache=True)
def _finish_exact(avg, err, counts):
    n_electrodes, n_cells, n_points = avg.shape
    for electrode in prange(n_electrodes):
        for cell in range(n_cells):
            count = np.float32(counts[cell])
            for point in range(n_points):
                mean = np.float32(avg[electrode, cell, point] / count)
                avg[electrode, cell, point] = mean
                variance = np.float32(
                    err[electrode, cell, point] -
                    np.float32(np.float32(count * mean) * mean))
                variance = np.float32(variance / count)
                err[electrode, cell, point] = np.float32(np.sqrt(np.float64(variance)))


def _finish_accurate(avg, err, counts):
    divisor = counts.astype(np.float64)[None, :, None]
    avg /= divisor
    variance = err / divisor - avg * avg
    err[...] = np.sqrt(np.maximum(variance, 0.0))


def _cell_blocks(cells: np.ndarray, block_size: int, n_cells: int):
    block = cells // block_size
    n_blocks = (n_cells + block_size - 1) // block_size
    order = np.argsort(block, kind="stable")
    bounds = np.zeros(n_blocks + 1, dtype=np.int64)
    bounds[1:] = np.cumsum(np.bincount(block, minlength=n_blocks))
    return order, bounds


def compute_ei_from_spikes(
    cell_ids, times, cells, raw_path, left=67, right=133,
    time_constant=0.01, mode="exact", chunk=1 << 19, cell_block=128,
):
    """Compute Vision-compatible EIs directly from Kilosort spike arrays."""
    if mode not in EI_MODES:
        raise VisionExportError(f"mode must be one of {EI_MODES}, got {mode!r}")
    if left < 0 or right < 0 or chunk <= 0 or cell_block <= 0:
        raise VisionExportError("EI window and block sizes must be positive")
    cell_ids = np.asarray(cell_ids, dtype=np.int64).reshape(-1)
    times = np.asarray(times, dtype=np.int64).reshape(-1)
    cells = np.asarray(cells, dtype=np.int32).reshape(-1)
    if times.size != cells.size or len(np.unique(cell_ids)) != len(cell_ids):
        raise VisionExportError("EI cell IDs and spike arrays are inconsistent")
    if np.any(cells < 0) or np.any(cells >= len(cell_ids)):
        raise VisionExportError("EI dense cell IDs are out of range")
    # The Vision writer wants spikes grouped by cell, while the streaming
    # accumulator uses searchsorted and therefore needs global time order.
    # Keep this copy local so export_vision can still write grouped neurons.
    order = np.argsort(times, kind="stable")
    times, cells = times[order], cells[order]

    with LitkeRecording(raw_path, drop_ttl=False) as recording:
        n_samples = int(recording.n_samples)
        array_id = int(recording.array_id)
        n_electrodes = int(recording.num_electrodes)
        ttl_times = recording.detect_ttl_pipeline_edges()
        keep = times + right <= n_samples - 1
        times, cells = times[keep], cells[keep]
        counts = np.bincount(cells, minlength=len(cell_ids)).astype(np.int64)
        n_points = left + right + 1
        truncate = mode in ("exact", "truncated64")
        exact = mode == "exact"
        dtype = np.float64 if not truncate else np.int16
        acc_dtype = np.float32 if exact else np.float64
        mean_dtype = np.float32 if truncate else np.float64
        avg = np.zeros((n_electrodes, len(cell_ids), n_points), dtype=acc_dtype)
        err = np.zeros_like(avg)
        pad = n_points - 1
        buf = np.zeros((n_electrodes, pad + chunk), dtype=dtype)
        mean = np.zeros(n_electrodes, dtype=mean_dtype)
        alpha = np.float64(np.float32(1.0 / (time_constant * _ALPHA_RATE)))
        pointer = 0
        for start in range(0, n_samples, chunk):
            n_time = min(chunk, n_samples - start)
            raw = np.asarray(recording[start:start + n_time], dtype=np.int16)
            _filter_chunk(raw, mean, alpha, buf, pad, truncate)
            stop = int(np.searchsorted(times, start + n_time - right, side="left"))
            if stop > pointer:
                sl = slice(pointer, stop)
                bases = (times[sl] - left - start + pad).astype(np.int64)
                order, bounds = _cell_blocks(cells[sl], cell_block, len(cell_ids))
                _accumulate(buf, bases[order], cells[sl][order], avg, err, bounds, exact)
                pointer = stop
            if n_time == chunk:
                buf[:, :pad] = buf[:, chunk:]
        if pointer != len(times):
            raise VisionExportError(f"EI accumulator missed {len(times) - pointer} spikes")
        if exact:
            _finish_exact(avg, err, counts)
        else:
            _finish_accurate(avg, err, counts)
    return cell_ids, counts, np.ascontiguousarray(avg.transpose(1, 0, 2)), \
        np.ascontiguousarray(err.transpose(1, 0, 2)), ttl_times, n_samples, array_id


def _destination(output_dir, dataset_name, overwrite):
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    name = dataset_name or output.name
    if not name or Path(name).name != name or name != output.name:
        raise VisionExportError("Vision dataset name must match its containing directory")
    expected = tuple(output / f"{name}{ext}" for ext in (".neurons", ".globals", ".ei"))
    existing = tuple(path for path in expected + (output / "vision_export.json",) if path.exists())
    if existing and not overwrite:
        raise VisionExportError("refusing to overwrite an existing Vision bundle; pass --overwrite")
    if overwrite:
        for path in existing:
            path.unlink()
    return output, name, expected


def export_vision(
    sort_dir, raw_path, output_dir=None, dataset_name=None, good_only=False,
    overwrite=False, left=67, right=133, time_constant=0.01,
    mode="exact", chunk=1 << 19, cell_block=128,
) -> Dict[str, object]:
    """Write ``.neurons``, ``.globals``, ``.ei`` and provenance JSON."""
    sort = Path(sort_dir).expanduser().resolve()
    raw = Path(raw_path).expanduser().resolve()
    if not sort.is_dir() or not raw.exists():
        raise VisionExportError("sort directory and raw Litke recording must exist")
    destination, name, expected = _destination(
        output_dir or sort, dataset_name, overwrite)
    try:
        import visionwriter as vw
    except ImportError as exc:
        raise VisionExportError(
            "Vision export needs the kilosort1 environment with visionwriter") from exc

    vision_ids, times, dense = load_kilosort_spikes(sort, good_only=good_only)
    ids, counts, avg, err, ttl_times, n_samples, array_id = compute_ei_from_spikes(
        vision_ids, times, dense, raw, left=left, right=right,
        time_constant=time_constant, mode=mode, chunk=chunk, cell_block=cell_block)
    if not np.array_equal(ids, vision_ids):
        raise VisionExportError("EI calculator changed the exported cell IDs")
    by_cell = {
        int(cid): np.asarray(times[dense == i], dtype=np.int32)
        for i, cid in enumerate(vision_ids)
    }
    by_ei = {
        int(cid): vw.WriteableEIData(
            np.asarray(avg[i, 1:], dtype=np.float32),
            np.asarray(err[i, 1:], dtype=np.float32), int(counts[i]))
        for i, cid in enumerate(vision_ids)
    }
    with vw.NeuronsFileWriter(str(destination) + os.sep, name) as writer:
        # Vision's writer stores these starts in its reserved channel-0 TTL table.
        writer.write_neuron_file(by_cell, ttl_times, int(n_samples))
    with vw.GlobalsFileWriter(str(destination) + os.sep, name) as writer:
        writer.write_simplified_litke_array_globals_file(
            int(array_id) & 0xFFF, 0, 0, "Kilosort Vision export", "", 0, int(n_samples))
    writer = vw.EIWriter(str(destination), name, left, right, int(array_id),
                         overwrite_existing=True)
    try:
        writer.write_eis_by_cell_id(by_ei)
    finally:
        writer.close()
    manifest = {
        "format": "Vision native Kilosort export",
        "dataset_name": name, "sort_dir": str(sort), "raw_path": str(raw),
        "array_id": int(array_id), "n_samples": int(n_samples), "sample_rate": 20000,
        "ttl_count": int(len(ttl_times)), "vision_ttl_channel": 0,
        "kilosort_cluster_id_base": 0, "vision_cell_id_base": 1,
        "cell_count": int(len(vision_ids)), "spike_count": int(len(times)),
        "good_only": bool(good_only),
        "ei": {"left_samples": int(left), "right_samples": int(right),
               "time_constant": float(time_constant), "mode": mode},
        "files": [path.name for path in expected],
    }
    (destination / "vision_export.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
