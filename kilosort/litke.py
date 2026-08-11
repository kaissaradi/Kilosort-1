"""Native Litke packed-bin reader for Kilosort (no convert step).

Litke MEA bins pack 12-bit electrode samples. **Electrode 0 is the TTL /
visual-stimulus trigger channel — not spikes.** Lab converters strip it before
sorting; this module does the same by default and can export TTL separately.

Array-like ``file_object`` for BinaryRWFile / ``run_kilosort(..., file_object=)``:

* Pure-Python Vision header; Numba unpack matching bin2py bit layout
  (bit-exact vs lab ``bin2py_cythonext`` on real 519 data).
* Multi-file folders joined like ``PyBinFileReader``.
* ``drop_ttl=True`` (default): electrode 0 omitted → 512/519 neural channels.
* ``get_ttl`` / ``save_ttl`` / ``detect_ttl_onsets``: keep stim sync separately.

Usage
-----
>>> from kilosort.litke import LitkeRecording
>>> rec = LitkeRecording('/path/to/data000')   # TTL dropped for sorting
>>> rec.save_ttl('ttl_chan0.npy')              # stim triggers only
>>> onsets = rec.detect_ttl_onsets()           # rising edges (lab thr=1000)
>>> from kilosort.io import BinaryRWFile
>>> bfile = BinaryRWFile(
...     filename=str(rec.paths[0]), n_chan_bin=rec.n_chan,
...     fs=rec.fs, file_object=rec, device='cpu')
"""

from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import BinaryIO, List, Optional, Sequence, Tuple, Union

import numpy as np
from numba import njit, prange


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


@njit(cache=True, parallel=True)
def _unpack_even_numba(buf: np.ndarray, n_samples: int, n_elec: int,
                       out: np.ndarray) -> None:
    # Independent per-sample byte base → prange-safe, bit-identical to serial.
    bps = 3 * (n_elec // 2)
    for i in prange(n_samples):
        k = i * bps
        for j in range(0, n_elec, 2):
            b1 = np.int32(buf[k])
            b2 = np.int32(buf[k + 1])
            b3 = np.int32(buf[k + 2])
            k += 3
            out[i, j] = np.int16(((b1 << 4) | (b2 >> 4)) - 2048)
            out[i, j + 1] = np.int16((((b2 & 0xF) << 8) | b3) - 2048)


@njit(cache=True, parallel=True)
def _unpack_odd_numba(buf: np.ndarray, n_samples: int, n_elec: int,
                      out: np.ndarray) -> None:
    # Odd: 2-byte TTL prefix + 3 bytes per remaining pair.
    bps = 2 + 3 * ((n_elec - 1) // 2)
    for i in prange(n_samples):
        k = i * bps
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


@njit(cache=True, parallel=True)
def _unpack_even_drop_ttl_numba(buf: np.ndarray, n_samples: int, n_elec: int,
                                out: np.ndarray) -> None:
    """Even board: skip electrode 0, write electrodes 1..n_elec-1 → out (n, n_elec-1)."""
    bps = 3 * (n_elec // 2)
    for i in prange(n_samples):
        k = i * bps
        # First pair: electrode 0 (TTL) + electrode 1
        b1 = np.int32(buf[k])
        b2 = np.int32(buf[k + 1])
        b3 = np.int32(buf[k + 2])
        k += 3
        # skip electrode 0; keep electrode 1 at out[:, 0]
        out[i, 0] = np.int16((((b2 & 0xF) << 8) | b3) - 2048)
        # Remaining pairs map electrodes 2,3,... → out columns 1,2,...
        col = 1
        for j in range(2, n_elec, 2):
            b1 = np.int32(buf[k])
            b2 = np.int32(buf[k + 1])
            b3 = np.int32(buf[k + 2])
            k += 3
            out[i, col] = np.int16(((b1 << 4) | (b2 >> 4)) - 2048)
            out[i, col + 1] = np.int16((((b2 & 0xF) << 8) | b3) - 2048)
            col += 2


@njit(cache=True, parallel=True)
def _unpack_odd_drop_ttl_numba(buf: np.ndarray, n_samples: int, n_elec: int,
                               out: np.ndarray) -> None:
    """Odd board: skip 16-bit TTL prefix, write neural electrodes to out."""
    bps = 2 + 3 * ((n_elec - 1) // 2)
    for i in prange(n_samples):
        k = i * bps + 2  # skip electrode 0 (raw 16-bit TTL)
        col = 0
        for j in range(1, n_elec, 2):
            b1 = np.int32(buf[k])
            b2 = np.int32(buf[k + 1])
            b3 = np.int32(buf[k + 2])
            k += 3
            out[i, col] = np.int16(((b1 << 4) | (b2 >> 4)) - 2048)
            out[i, col + 1] = np.int16((((b2 & 0xF) << 8) | b3) - 2048)
            col += 2


def unpack_samples_drop_ttl(buf: np.ndarray, n_samples: int, n_elec: int,
                            out: Optional[np.ndarray] = None) -> np.ndarray:
    """Unpack neural channels only: shape ``(n_samples, n_elec - 1)``.

    Bit-identical to ``unpack_samples(...)[:, 1:]`` without allocating the TTL
    column. Used by :class:`LitkeRecording` when ``drop_ttl=True`` (sort path).
    """
    if n_elec < 2:
        raise ValueError('drop_ttl unpack requires n_elec >= 2')
    buf = np.ascontiguousarray(buf, dtype=np.uint8).ravel()
    need = n_samples * bytes_per_sample(n_elec)
    if buf.size < need:
        raise ValueError(
            f'buffer has {buf.size} bytes, need {need} for '
            f'{n_samples} samples × {n_elec} electrodes'
        )
    n_out = n_elec - 1
    if out is None:
        out = np.empty((n_samples, n_out), dtype=np.int16)
    elif out.shape != (n_samples, n_out) or out.dtype != np.int16:
        raise ValueError(
            f'out must be int16 with shape (n_samples, {n_out})'
        )
    if n_elec % 2 == 0:
        _unpack_even_drop_ttl_numba(buf, n_samples, n_elec, out)
    else:
        _unpack_odd_drop_ttl_numba(buf, n_samples, n_elec, out)
    return out


@njit(cache=True, parallel=True)
def _unpack_ttl_even_numba(buf: np.ndarray, n_samples: int, n_elec: int,
                           out: np.ndarray) -> None:
    """Electrode 0 only for even boards (first 12-bit of each sample)."""
    bps = 3 * n_elec // 2
    for i in prange(n_samples):
        base = i * bps
        b1 = np.int32(buf[base])
        b2 = np.int32(buf[base + 1])
        out[i] = np.int16(((b1 << 4) | (b2 >> 4)) - 2048)


@njit(cache=True, parallel=True)
def _unpack_ttl_odd_numba(buf: np.ndarray, n_samples: int, n_elec: int,
                          out: np.ndarray) -> None:
    """Electrode 0 only for odd boards (raw 16-bit TTL prefix)."""
    bps = 2 + (n_elec - 1) * 3 // 2
    for i in prange(n_samples):
        base = i * bps
        b1 = np.int32(buf[base])
        b2 = np.int32(buf[base + 1])
        out[i] = np.int16((b1 << 8) | (b2 & 0xFF))


def unpack_ttl(buf: np.ndarray, n_samples: int, n_elec: int,
               out: Optional[np.ndarray] = None) -> np.ndarray:
    """Unpack only electrode 0 (TTL / stim) from packed Litke bytes.

    Bit-identical to ``unpack_samples(...)[:, 0]`` but does not materialize the
    full (n_samples, n_elec) matrix — important for full-recording TTL export
    on memory-constrained hosts (~18 GiB avoided on 519-ch data000).
    """
    buf = np.ascontiguousarray(buf, dtype=np.uint8).ravel()
    need = n_samples * bytes_per_sample(n_elec)
    if buf.size < need:
        raise ValueError(
            f'buffer has {buf.size} bytes, need {need} for '
            f'{n_samples} samples × {n_elec} electrodes'
        )
    if out is None:
        out = np.empty(n_samples, dtype=np.int16)
    elif out.shape != (n_samples,) or out.dtype != np.int16:
        raise ValueError('out must be int16 with shape (n_samples,)')
    if n_elec % 2 == 0:
        _unpack_ttl_even_numba(buf, n_samples, n_elec, out)
    else:
        _unpack_ttl_odd_numba(buf, n_samples, n_elec, out)
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


# Default edge threshold used by MEA-fieldlab / convert_litke_to_kilosort
# when detecting stimulus triggers on electrode 0.
DEFAULT_TTL_THRESHOLD = 1000


class LitkeRecording:
    """Array-like view of a Litke recording for Kilosort ``file_object``.

    Electrode **0 is the TTL / visual-stim trigger channel**, not a spike
    channel. With ``drop_ttl=True`` (default) it is excluded from ``shape`` and
    ``__getitem__`` so Kilosort only sees neural electrodes (512 or 519).
    Use :meth:`get_ttl`, :meth:`save_ttl`, and :meth:`detect_ttl_onsets` to keep
    the stim sync stream.

    Parameters
    ----------
    path : str or Path
        Folder of multi-part ``.bin`` files, or a single ``.bin`` path.
    drop_ttl : bool
        If True (default), electrode 0 (TTL / stim triggers) is omitted so
        ``shape[1]`` matches the lab converter output and Litke probe maps
        (512 or 519 channels). **Leave True for spike sorting.**
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

    def _iter_packed_chunks(self, start: int, n: int):
        """Yield ``(packed_uint8, take)`` for contiguous global samples.

        Shared by full unpack and TTL-only paths so multi-file edge math stays
        in one place.
        """
        if n <= 0:
            return
        end = start + n
        if start < 0 or end > self.n_samples:
            raise IndexError(
                f'requested samples [{start}, {end}) outside '
                f'[0, {self.n_samples})'
            )
        written = 0
        sample = start
        while written < n:
            fi = self._file_index_for_sample(sample)
            file_start = self.sample_edges[fi]
            file_end = self.sample_edges[fi + 1]
            take = min(n - written, file_end - sample)
            local = sample - file_start
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
            yield raw, take
            written += take
            sample += take

    def _read_raw_samples(self, start: int, n: int) -> np.ndarray:
        """Read and unpack ``n`` samples starting at global sample ``start``.

        Returns int16 array shape (n, num_electrodes) including TTL when
        ``drop_ttl=False``. With ``drop_ttl=True`` (default sort path), returns
        shape (n, num_electrodes-1) via a TTL-skipping unpack — bit-identical
        to full unpack then ``[:, 1:]`` without the extra channel column.
        """
        if self.drop_ttl:
            if n <= 0:
                return np.zeros((0, self._n_chan), dtype=np.int16)
            out = np.empty((n, self._n_chan), dtype=np.int16)
            written = 0
            for raw, take in self._iter_packed_chunks(start, n):
                unpack_samples_drop_ttl(
                    raw, take, self.num_electrodes,
                    out=out[written:written + take],
                )
                written += take
            return out

        if n <= 0:
            return np.zeros((0, self.num_electrodes), dtype=np.int16)
        out = np.empty((n, self.num_electrodes), dtype=np.int16)
        written = 0
        for raw, take in self._iter_packed_chunks(start, n):
            unpack_samples(
                raw, take, self.num_electrodes,
                out=out[written:written + take],
            )
            written += take
        return out

    def _read_ttl_samples(self, start: int, n: int) -> np.ndarray:
        """Read electrode 0 only — bit-identical to ``_read_raw_samples()[:, 0]``."""
        if n <= 0:
            return np.zeros(0, dtype=np.int16)
        out = np.empty(n, dtype=np.int16)
        written = 0
        for raw, take in self._iter_packed_chunks(start, n):
            unpack_ttl(
                raw, take, self.num_electrodes,
                out=out[written:written + take],
            )
            written += take
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

        # Resolve time → (start, n) contiguous read when possible.
        # _read_raw_samples already drops TTL when drop_ttl=True.
        n_full = self._n_chan if self.drop_ttl else self.num_electrodes
        if isinstance(t_idx, slice):
            start, stop, step = t_idx.indices(self.n_samples)
            if step != 1:
                # Fall back to gathering individual samples (rare).
                times = np.arange(start, stop, step)
                if times.size == 0:
                    data = np.zeros((0, n_full), dtype=np.int16)
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
            data = data[0]  # (n_chan,)
            return data[c_idx]
        else:
            times = np.asarray(t_idx, dtype=np.int64).ravel()
            if times.size == 0:
                data = np.zeros((0, n_full), dtype=np.int16)
            else:
                t0, t1 = int(times.min()), int(times.max())
                span = self._read_raw_samples(t0, t1 - t0 + 1)
                data = span[times - t0]

        return data[:, c_idx]

    # -- TTL / stim trigger channel (electrode 0) ----------------------------

    def get_ttl(self, start: int = 0, n_samples: Optional[int] = None) -> np.ndarray:
        """Return electrode 0 (TTL / visual-stim triggers) as int16.

        This channel is **not** neural data. It is always electrode index 0 in
        the packed Litke layout, independent of ``drop_ttl``.

        Parameters
        ----------
        start : int
            First sample index (inclusive).
        n_samples : int or None
            Number of samples. ``None`` reads through the end of the recording.

        Returns
        -------
        np.ndarray
            Shape ``(n_samples,)``, dtype int16.
        """
        start = int(start)
        if n_samples is None:
            n_samples = self.n_samples - start
        n_samples = int(n_samples)
        # TTL-only path: do not allocate (n, n_elec) just to drop all but ch0.
        return self._read_ttl_samples(start, n_samples)

    def save_ttl(self, path: Union[str, Path], start: int = 0,
                 n_samples: Optional[int] = None,
                 chunk_samples: int = 100_000) -> Path:
        """Write electrode 0 (TTL) to ``.npy`` (int16 vector) for later use.

        Does not include TTL in the Kilosort ``file_object`` stream. Prefer this
        over sorting with ``drop_ttl=False``.

        Parameters
        ----------
        path : path-like
            Output path. Should end in ``.npy`` (``np.save``).
        start, n_samples
            Sample window; same meaning as :meth:`get_ttl`.
        chunk_samples : int
            Disk-friendly read size when exporting long recordings.

        Returns
        -------
        Path
            Resolved output path.
        """
        path = Path(path)
        start = int(start)
        total = self.n_samples - start if n_samples is None else int(n_samples)
        if total < 0 or start + total > self.n_samples:
            raise IndexError('TTL export window out of bounds')

        out = np.empty(total, dtype=np.int16)
        done = 0
        while done < total:
            take = min(chunk_samples, total - done)
            out[done:done + take] = self.get_ttl(start + done, take)
            done += take
        np.save(path, out)
        return path.resolve()

    def detect_ttl_onsets(self, threshold: int = DEFAULT_TTL_THRESHOLD,
                          start: int = 0,
                          n_samples: Optional[int] = None,
                          chunk_samples: int = 100_000) -> np.ndarray:
        """Sample indices of rising TTL edges (lab converter convention).

        Matches MEA-fieldlab / ``convert_litke_to_kilosort``: a rising edge is
        where the signal goes from ``< -threshold`` to ``>= -threshold``
        (default ``threshold=1000``). Indices are absolute within the full
        recording (offset by ``start``).

        Returns
        -------
        np.ndarray
            1-D int64 sample indices of detected onsets.
        """
        start = int(start)
        total = self.n_samples - start if n_samples is None else int(n_samples)
        if total <= 1:
            return np.zeros(0, dtype=np.int64)

        thr = int(threshold)
        onsets: List[int] = []
        # Carry one sample so edges on chunk boundaries are not missed.
        prev = None
        done = 0
        while done < total:
            take = min(chunk_samples, total - done)
            seg = self.get_ttl(start + done, take)
            if prev is not None:
                work = np.empty(take + 1, dtype=np.int16)
                work[0] = prev
                work[1:] = seg
                offset = start + done - 1
            else:
                work = seg
                offset = start + done
            below = work < -thr
            above = ~below
            edges = np.flatnonzero(below[:-1] & above[1:])
            if edges.size:
                onsets.append(edges.astype(np.int64) + offset)
            prev = seg[-1]
            done += take

        if not onsets:
            return np.zeros(0, dtype=np.int64)
        return np.concatenate(onsets)


def open_litke(path: Union[str, Path], drop_ttl: bool = True) -> LitkeRecording:
    """Convenience constructor matching lab default (TTL dropped for sorting)."""
    return LitkeRecording(path, drop_ttl=drop_ttl)
