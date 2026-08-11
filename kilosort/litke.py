"""Native Litke packed-bin reader for Kilosort (no convert step).

Litke MEA recordings store samples as packed 12-bit values (plus a 16-bit TTL
channel on odd electrode counts). Lab tooling converts to int16 via bin2py
before sorting; that rewrite is multi-GB and blocked when the Cython extension
is not built.

This module provides an array-like ``file_object`` that BinaryRWFile /
``run_kilosort(..., file_object=...)`` can stream from directly:

* Header parse is pure Python (big-endian Vision tags).
* Sample unpack is Numba JIT (already a kilosort dependency) — no Cython,
  no CUDA. Matches bin2py's ``unpack_bin_{even,odd}_num_electrodes``.
* Multi-file folders (``data000000.bin``, ``data000001.bin``, …) are joined
  by sample index the same way ``PyBinFileReader`` does.
* By default the TTL channel is dropped so ``shape[1]`` equals the recording
  channel count used by the lab converter / probe maps (512 or 519).

Usage
-----
>>> from kilosort.litke import LitkeRecording
>>> rec = LitkeRecording('/path/to/data000')   # folder or single .bin
>>> # rec.shape == (n_samples, n_channels), dtype int16
>>> from kilosort.io import BinaryRWFile
>>> bfile = BinaryRWFile(
...     filename=str(rec.paths[0]), n_chan_bin=rec.shape[1],
...     fs=rec.fs, file_object=rec, device='cpu')

Accuracy: unpack is bit-exact vs the reference pack/unpack in tests and vs
bin2py's published bit layout. Do not rewrite the nibble packing.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import BinaryIO, List, Optional, Sequence, Tuple, Union

import numpy as np
from numba import njit


# --- Vision / Litke header tags (big-endian uint32) -------------------------

_NBYTES_32 = 4
_HEADER_LENGTH_TAG = 0
_TIME_TAG = 1
_COMMENT_TAG = 2
_FORMAT_TAG = 3
_ARRAY_ID_TAG = 4
_FREQUENCY_TAG = 5
_TRIGGER_TAG = 6
_DATASET_ID_TAG = 7
_TRIGGER_TAG_V2 = 8
_DATA_TAG = 499

_HEADER_LENGTH_BYTES = 4
_TIME_LENGTH_BYTES = 12
_FORMAT_LENGTH_BYTES = 4
_ARRAY_LENGTH_BYTES = 8
_FREQUENCY_LENGTH_BYTES = 4
_TRIGGER_LENGTH_BYTES = 8
_TRIGGER_V2_LENGTH_BYTES = 16
_DATA_TAG_LENGTH_BYTES = 4


def bytes_per_sample(n_electrodes: int) -> int:
    """Packed bytes per sample including TTL (channel 0 when odd count)."""
    if n_electrodes % 2 == 0:
        return 3 * n_electrodes // 2
    return 2 + (n_electrodes - 1) * 3 // 2


def parse_litke_header(source: Union[BinaryIO, bytes, bytearray, memoryview]) -> dict:
    """Parse a Litke/Vision bin header from a file stream or raw bytes.

    Returns a dict with keys: header_length, time_base, seconds_time, comment,
    dataset_identifier, format, array_id, num_electrodes, frequency, n_samples.
    """
    if hasattr(source, 'read'):
        return _parse_header_stream(source)
    return _parse_header_bytes(source)


def _parse_header_stream(f: BinaryIO) -> dict:
    initial = f.tell()
    f.seek(0)
    try:
        tag, size = struct.unpack('>II', f.read(8))
        if tag != _HEADER_LENGTH_TAG or size != _HEADER_LENGTH_BYTES:
            raise ValueError('Not a Litke bin: missing header-length tag')
        header_length, = struct.unpack('>I', f.read(_HEADER_LENGTH_BYTES))

        time_base = seconds_time = dformat = num_electrodes = array_id = -1
        frequency = n_samples = -1
        comment = dataset_identifier = ''

        tag = -1
        while tag != _DATA_TAG:
            tag, size = struct.unpack('>II', f.read(8))
            if tag == _TIME_TAG:
                if size != _TIME_LENGTH_BYTES:
                    raise ValueError('Time tag size mismatch')
                time_base, seconds_time = struct.unpack('>IQ', f.read(12))
            elif tag == _COMMENT_TAG:
                comment = f.read(size).decode('utf-8', errors='replace')
            elif tag == _FORMAT_TAG:
                dformat, = struct.unpack('>I', f.read(4))
            elif tag == _ARRAY_ID_TAG:
                num_electrodes, array_id = struct.unpack('>II', f.read(8))
            elif tag == _FREQUENCY_TAG:
                frequency, = struct.unpack('>I', f.read(4))
            elif tag == _TRIGGER_TAG:
                f.read(8)  # Vision ignores contents
            elif tag == _TRIGGER_TAG_V2:
                f.read(16)
            elif tag == _DATASET_ID_TAG:
                dataset_identifier = f.read(size).decode('utf-8', errors='replace')
            elif tag == _DATA_TAG:
                if size != _DATA_TAG_LENGTH_BYTES:
                    raise ValueError('Data tag size mismatch')
                n_samples, = struct.unpack('>I', f.read(4))
            else:
                raise ValueError(f'Unknown Litke header tag {tag}')
    finally:
        f.seek(initial)

    return {
        'header_length': int(header_length),
        'time_base': int(time_base),
        'seconds_time': int(seconds_time),
        'comment': comment,
        'dataset_identifier': dataset_identifier,
        'format': int(dformat),
        'array_id': int(array_id),
        'num_electrodes': int(num_electrodes),
        'frequency': int(frequency),
        'n_samples': int(n_samples),
    }


def _parse_header_bytes(buf: Union[bytes, bytearray, memoryview]) -> dict:
    class _Mem:
        def __init__(self, b):
            self._b = memoryview(b)
            self._i = 0

        def tell(self):
            return self._i

        def seek(self, pos):
            self._i = pos

        def read(self, n):
            out = self._b[self._i:self._i + n]
            self._i += n
            return bytes(out)

    return _parse_header_stream(_Mem(buf))


# --- Pack / unpack (reference + Numba) --------------------------------------

def pack_samples_even(data: np.ndarray) -> np.ndarray:
    """Pack even electrode count (e.g. 520 incl. TTL) → uint8 buffer.

    Reference implementation of bin2py pack; used by tests and small writers.
    """
    data = np.asarray(data, dtype=np.int16)
    n_samples, n_elec = data.shape
    if n_elec % 2 != 0:
        raise ValueError('even pack requires even electrode count')
    out = np.empty(n_samples * (3 * n_elec // 2), dtype=np.uint8)
    k = 0
    for i in range(n_samples):
        for j in range(0, n_elec, 2):
            first = int(data[i, j]) + 2048
            second = int(data[i, j + 1]) + 2048
            out[k] = (first >> 4) & 0xFF
            out[k + 1] = ((first & 0xF) << 4) | ((second >> 8) & 0xF)
            out[k + 2] = second & 0xFF
            k += 3
    return out


def pack_samples_odd(data: np.ndarray) -> np.ndarray:
    """Pack odd electrode count (e.g. 513 incl. TTL) → uint8 buffer."""
    data = np.asarray(data, dtype=np.int16)
    n_samples, n_elec = data.shape
    if n_elec % 2 == 0:
        raise ValueError('odd pack requires odd electrode count')
    out = np.empty(n_samples * (2 + (n_elec - 1) * 3 // 2), dtype=np.uint8)
    k = 0
    for i in range(n_samples):
        ttl = int(data[i, 0])
        out[k] = (ttl >> 8) & 0xFF
        out[k + 1] = ttl & 0xFF
        k += 2
        for j in range(1, n_elec, 2):
            first = int(data[i, j]) + 2048
            second = int(data[i, j + 1]) + 2048
            out[k] = (first >> 4) & 0xFF
            out[k + 1] = ((first & 0xF) << 4) | ((second >> 8) & 0xF)
            out[k + 2] = second & 0xFF
            k += 3
    return out


def pack_samples(data: np.ndarray) -> np.ndarray:
    """Pack (n_samples, n_electrodes) int16 → packed uint8 bytes."""
    data = np.asarray(data, dtype=np.int16)
    if data.shape[1] % 2 == 0:
        return pack_samples_even(data)
    return pack_samples_odd(data)


@njit(cache=True)
def _unpack_even_numba(buf: np.ndarray, n_samples: int, n_elec: int,
                       out: np.ndarray) -> None:
    k = 0
    for i in range(n_samples):
        for j in range(0, n_elec, 2):
            b1 = np.int32(buf[k])
            b2 = np.int32(buf[k + 1])
            b3 = np.int32(buf[k + 2])
            k += 3
            out[i, j] = np.int16(((b1 << 4) | (b2 >> 4)) - 2048)
            out[i, j + 1] = np.int16((((b2 & 0xF) << 8) | b3) - 2048)


@njit(cache=True)
def _unpack_odd_numba(buf: np.ndarray, n_samples: int, n_elec: int,
                      out: np.ndarray) -> None:
    k = 0
    for i in range(n_samples):
        b1 = np.int32(buf[k])
        b2 = np.int32(buf[k + 1])
        k += 2
        out[i, 0] = np.int16((b1 << 8) | (b2 & 0xFF))
        for j in range(1, n_elec, 2):
            b1 = np.int32(buf[k])
            b2 = np.int32(buf[k + 1])
            b3 = np.int32(buf[k + 2])
            k += 3
            out[i, j] = np.int16(((b1 << 4) | (b2 >> 4)) - 2048)
            out[i, j + 1] = np.int16((((b2 & 0xF) << 8) | b3) - 2048)


def unpack_samples_python(buf: np.ndarray, n_samples: int,
                          n_elec: int) -> np.ndarray:
    """Pure-Python unpack (oracle for tests; slow)."""
    buf = np.asarray(buf, dtype=np.uint8).ravel()
    out = np.empty((n_samples, n_elec), dtype=np.int16)
    k = 0
    if n_elec % 2 == 0:
        for i in range(n_samples):
            for j in range(0, n_elec, 2):
                b1, b2, b3 = int(buf[k]), int(buf[k + 1]), int(buf[k + 2])
                k += 3
                out[i, j] = ((b1 << 4) | (b2 >> 4)) - 2048
                out[i, j + 1] = (((b2 & 0xF) << 8) | b3) - 2048
    else:
        for i in range(n_samples):
            b1, b2 = int(buf[k]), int(buf[k + 1])
            k += 2
            out[i, 0] = (b1 << 8) | (b2 & 0xFF)
            for j in range(1, n_elec, 2):
                b1, b2, b3 = int(buf[k]), int(buf[k + 1]), int(buf[k + 2])
                k += 3
                out[i, j] = ((b1 << 4) | (b2 >> 4)) - 2048
                out[i, j + 1] = (((b2 & 0xF) << 8) | b3) - 2048
    return out


def unpack_samples(buf: np.ndarray, n_samples: int, n_elec: int,
                   out: Optional[np.ndarray] = None) -> np.ndarray:
    """Unpack packed Litke bytes → (n_samples, n_elec) int16 (Numba)."""
    buf = np.ascontiguousarray(buf, dtype=np.uint8).ravel()
    need = n_samples * bytes_per_sample(n_elec)
    if buf.size < need:
        raise ValueError(
            f'buffer has {buf.size} bytes, need {need} for '
            f'{n_samples} samples × {n_elec} electrodes'
        )
    if out is None:
        out = np.empty((n_samples, n_elec), dtype=np.int16)
    elif out.shape != (n_samples, n_elec) or out.dtype != np.int16:
        raise ValueError('out must be int16 with shape (n_samples, n_elec)')
    if n_elec % 2 == 0:
        _unpack_even_numba(buf, n_samples, n_elec, out)
    else:
        _unpack_odd_numba(buf, n_samples, n_elec, out)
    return out


# --- Array-like multi-file recording ----------------------------------------

def _list_bin_paths(path: Union[str, Path], ext: str = '.bin') -> List[Path]:
    path = Path(path)
    if path.is_dir():
        files = sorted(
            p for p in path.iterdir()
            if p.is_file() and p.suffix == ext
        )
        if not files:
            raise FileNotFoundError(f'No {ext} files in {path}')
        return files
    if not path.is_file():
        raise FileNotFoundError(path)
    return [path]


class LitkeRecording:
    """Array-like view of a Litke recording for Kilosort ``file_object``.

    Parameters
    ----------
    path : str or Path
        Folder of multi-part ``.bin`` files, or a single ``.bin`` path.
    drop_ttl : bool
        If True (default), channel 0 (TTL) is omitted so ``shape[1]`` matches
        the lab converter output and Litke probe maps (512 or 519 channels).
    ext : str
        File extension to collect in folder mode (default ``.bin``).
    """

    def __init__(self, path: Union[str, Path], drop_ttl: bool = True,
                 ext: str = '.bin'):
        self.paths = _list_bin_paths(path, ext=ext)
        self.drop_ttl = bool(drop_ttl)

        # Open all parts read-only; header only on first file.
        self._files: List[BinaryIO] = [open(p, 'rb') for p in self.paths]
        try:
            self.header = parse_litke_header(self._files[0])
        except Exception:
            self.close()
            raise

        self.num_electrodes = int(self.header['num_electrodes'])
        self.array_id = int(self.header['array_id'])
        self.fs = float(self.header['frequency'])
        self.header_n_samples = int(self.header['n_samples'])
        self.header_length = int(self.header['header_length'])
        self.bytes_per_sample = bytes_per_sample(self.num_electrodes)
        self._even = (self.num_electrodes % 2 == 0)

        # Sample index boundaries across files (like PyBinFileReader).
        self.sample_edges: List[int] = [0]
        scount = 0
        for i, p in enumerate(self.paths):
            fsize = os.stat(p).st_size
            body = fsize - (self.header_length if i == 0 else 0)
            n = body // self.bytes_per_sample
            scount += n
            self.sample_edges.append(scount)
        self.n_samples = scount

        n_out = self.num_electrodes - (1 if self.drop_ttl else 0)
        if n_out <= 0:
            self.close()
            raise ValueError('no recording channels after drop_ttl')
        self._n_chan = n_out
        self.dtype = np.dtype(np.int16)
        self.shape = (self.n_samples, self._n_chan)

    # -- context / lifecycle -------------------------------------------------

    def close(self) -> None:
        for f in getattr(self, '_files', []) or []:
            try:
                f.close()
            except Exception:
                pass
        self._files = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # -- properties ----------------------------------------------------------

    @property
    def n_chan(self) -> int:
        return self._n_chan

    def __repr__(self) -> str:
        return (
            f'LitkeRecording(path={self.paths[0].parent if len(self.paths) > 1 else self.paths[0]!s}, '
            f'shape={self.shape}, fs={self.fs}, array_id={self.array_id}, '
            f'drop_ttl={self.drop_ttl})'
        )

    # -- indexing ------------------------------------------------------------

    def _file_index_for_sample(self, sample: int) -> int:
        if sample < 0 or sample >= self.n_samples:
            raise IndexError(
                f'sample {sample} out of bounds [0, {self.n_samples})'
            )
        # sample_edges is sorted; find rightmost edge <= sample
        edges = self.sample_edges
        lo, hi = 0, len(edges) - 2
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if edges[mid] <= sample:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def _read_raw_samples(self, start: int, n: int) -> np.ndarray:
        """Read and unpack ``n`` samples starting at global sample ``start``.

        Returns int16 array shape (n, num_electrodes) including TTL.
        """
        if n <= 0:
            return np.zeros((0, self.num_electrodes), dtype=np.int16)
        end = start + n
        if start < 0 or end > self.n_samples:
            raise IndexError(
                f'requested samples [{start}, {end}) outside '
                f'[0, {self.n_samples})'
            )

        out = np.empty((n, self.num_electrodes), dtype=np.int16)
        written = 0
        sample = start
        while written < n:
            fi = self._file_index_for_sample(sample)
            file_start = self.sample_edges[fi]
            file_end = self.sample_edges[fi + 1]
            take = min(n - written, file_end - sample)
            local = sample - file_start
            # byte offset inside this file
            data_off = (self.header_length if fi == 0 else 0)
            byte_off = data_off + local * self.bytes_per_sample
            nbytes = take * self.bytes_per_sample
            f = self._files[fi]
            f.seek(byte_off)
            raw = np.frombuffer(f.read(nbytes), dtype=np.uint8)
            if raw.size != nbytes:
                raise IOError(
                    f'short read on {self.paths[fi]}: got {raw.size}, '
                    f'want {nbytes}'
                )
            unpack_samples(
                raw, take, self.num_electrodes,
                out=out[written:written + take],
            )
            written += take
            sample += take
        return out

    def __getitem__(self, idx):
        """Numpy-like indexing: ``rec[t0:t1]`` or ``rec[t0:t1, chans]``.

        Time index may be a slice or integer. Channel index optional.
        """
        if isinstance(idx, tuple):
            if len(idx) != 2:
                raise IndexError('LitkeRecording supports 1D or 2D indexing')
            t_idx, c_idx = idx
        else:
            t_idx, c_idx = idx, slice(None)

        # Resolve time → (start, n) contiguous read when possible
        if isinstance(t_idx, slice):
            start, stop, step = t_idx.indices(self.n_samples)
            if step != 1:
                # Fall back to gathering individual samples (rare).
                times = np.arange(start, stop, step)
                if times.size == 0:
                    data = np.zeros((0, self.num_electrodes), dtype=np.int16)
                else:
                    # Read contiguous span then subsample (usually cheaper).
                    span = self._read_raw_samples(int(times[0]),
                                                  int(times[-1] - times[0] + 1))
                    data = span[times - times[0]]
            else:
                data = self._read_raw_samples(start, stop - start)
        elif isinstance(t_idx, (int, np.integer)):
            t = int(t_idx)
            if t < 0:
                t += self.n_samples
            data = self._read_raw_samples(t, 1)
            data = data[0]  # (n_elec,)
            if self.drop_ttl:
                data = data[1:]
            return data[c_idx]
        else:
            times = np.asarray(t_idx, dtype=np.int64).ravel()
            if times.size == 0:
                data = np.zeros((0, self.num_electrodes), dtype=np.int16)
            else:
                t0, t1 = int(times.min()), int(times.max())
                span = self._read_raw_samples(t0, t1 - t0 + 1)
                data = span[times - t0]

        if self.drop_ttl:
            data = data[:, 1:]
        return data[:, c_idx]


def open_litke(path: Union[str, Path], drop_ttl: bool = True) -> LitkeRecording:
    """Convenience constructor matching lab default (TTL dropped)."""
    return LitkeRecording(path, drop_ttl=drop_ttl)
