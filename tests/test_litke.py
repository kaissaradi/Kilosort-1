"""Identity and smoke tests for native Litke IO (kilosort.litke)."""

import struct
from pathlib import Path

import numpy as np
import pytest
import torch

from kilosort import litke
from kilosort.io import BinaryRWFile


def _write_minimal_header(n_elec, n_samples, array_id=1551, freq=20000,
                          comment='', dsid='test'):
    """Build a valid Litke header bytes object (big-endian Vision tags)."""
    if len(comment) % 2:
        comment = comment + ' '
    parts = []
    # placeholder header length — fill after
    body = []
    body.append(struct.pack('>III', 0, 4, 0))  # HEADER_LENGTH_TAG + size + value placeholder
    body.append(struct.pack('>IIIQ', 1, 12, 1904, 0))  # TIME
    cbytes = comment.encode('utf-8')
    body.append(struct.pack('>II', 2, len(cbytes)) + cbytes)
    body.append(struct.pack('>III', 3, 4, 1))  # FORMAT
    body.append(struct.pack('>IIII', 4, 8, n_elec, array_id))
    body.append(struct.pack('>III', 5, 4, freq))
    dbytes = dsid.encode('utf-8')
    body.append(struct.pack('>II', 7, len(dbytes)) + dbytes)
    body.append(struct.pack('>III', 499, 4, n_samples))  # DATA_TAG
    raw = b''.join(body)
    # fix header_length field (bytes 8:12 of first tag payload)
    # layout: tag(4)+size(4)+header_length(4) ...
    hl = len(raw)
    fixed = struct.pack('>III', 0, 4, hl) + raw[12:]
    assert len(fixed) == hl
    return fixed


def _write_litke_file(path, data_with_ttl, array_id=1551, freq=20000):
    """Write one Litke .bin from (n_samples, n_elec) int16 including TTL."""
    data = np.asarray(data_with_ttl, dtype=np.int16)
    n_samples, n_elec = data.shape
    header = _write_minimal_header(n_elec, n_samples, array_id=array_id, freq=freq)
    packed = litke.pack_samples(data)
    path.write_bytes(header + packed.tobytes())
    return path


def test_bytes_per_sample_known_boards():
    assert litke.bytes_per_sample(520) == 780  # 519 + TTL
    assert litke.bytes_per_sample(513) == 770  # 512 + TTL


def test_pack_unpack_even_identity():
    rng = np.random.default_rng(0)
    # 12-bit-ish range after offset: keep values in [-2048, 2047]
    data = rng.integers(-500, 500, size=(17, 8), dtype=np.int16)
    packed = litke.pack_samples_even(data)
    out = litke.unpack_samples(packed, 17, 8)
    np.testing.assert_array_equal(out, data)
    # numba matches pure python oracle
    ref = litke.unpack_samples_python(packed, 17, 8)
    np.testing.assert_array_equal(out, ref)


def test_pack_unpack_odd_identity():
    rng = np.random.default_rng(1)
    # TTL is raw 16-bit; recording chans in 12-bit range
    ttl = rng.integers(-2000, 2000, size=(11, 1), dtype=np.int16)
    rec = rng.integers(-800, 800, size=(11, 6), dtype=np.int16)
    data = np.concatenate([ttl, rec], axis=1)
    packed = litke.pack_samples_odd(data)
    out = litke.unpack_samples(packed, 11, 7)
    np.testing.assert_array_equal(out, data)
    ref = litke.unpack_samples_python(packed, 11, 7)
    np.testing.assert_array_equal(out, ref)


def test_unpack_drop_ttl_matches_full_slice_even():
    rng = np.random.default_rng(0)
    data = rng.integers(-2000, 2000, size=(32, 8), dtype=np.int16)
    packed = litke.pack_samples(data)
    full = litke.unpack_samples(packed, 32, 8)
    drop = litke.unpack_samples_drop_ttl(packed, 32, 8)
    np.testing.assert_array_equal(drop, full[:, 1:])


def test_unpack_drop_ttl_matches_full_slice_odd():
    rng = np.random.default_rng(1)
    data = rng.integers(-2000, 2000, size=(20, 7), dtype=np.int16)
    # Odd pack uses raw 16-bit for ch0; keep in int16 range
    packed = litke.pack_samples(data)
    full = litke.unpack_samples(packed, 20, 7)
    drop = litke.unpack_samples_drop_ttl(packed, 20, 7)
    np.testing.assert_array_equal(drop, full[:, 1:])


def test_unpack_ttl_matches_full_unpack_even():
    """TTL-only path must be bit-identical to full unpack[:, 0] (even)."""
    rng = np.random.default_rng(42)
    data = rng.integers(-500, 500, size=(64, 8), dtype=np.int16)
    packed = litke.pack_samples_even(data)
    full = litke.unpack_samples(packed, 64, 8)
    ttl = litke.unpack_ttl(packed, 64, 8)
    np.testing.assert_array_equal(ttl, full[:, 0])
    # preallocated out
    out = np.empty(64, dtype=np.int16)
    litke.unpack_ttl(packed, 64, 8, out=out)
    np.testing.assert_array_equal(out, full[:, 0])


def test_unpack_ttl_matches_full_unpack_odd():
    """TTL-only path must be bit-identical to full unpack[:, 0] (odd)."""
    rng = np.random.default_rng(43)
    ttl0 = rng.integers(-3000, 3000, size=(40, 1), dtype=np.int16)
    rec = rng.integers(-800, 800, size=(40, 6), dtype=np.int16)
    data = np.concatenate([ttl0, rec], axis=1)
    packed = litke.pack_samples_odd(data)
    full = litke.unpack_samples(packed, 40, 7)
    ttl = litke.unpack_ttl(packed, 40, 7)
    np.testing.assert_array_equal(ttl, full[:, 0])


def test_get_ttl_identity_vs_full_decode(tmp_path):
    """Recording.get_ttl must match electrode 0 of a full (drop_ttl=False) decode.

    With drop_ttl=True, ``_read_raw_samples`` no longer includes TTL — compare
    against an explicit full-stream open instead.
    """
    rng = np.random.default_rng(44)
    n_samples, n_elec = 120, 8
    data = rng.integers(-400, 400, size=(n_samples, n_elec), dtype=np.int16)
    path = _write_litke_file(tmp_path / 'ttl_id.bin', data)
    with litke.LitkeRecording(path, drop_ttl=True) as rec:
        got = rec.get_ttl()
        got_slice = rec.get_ttl(10, 25)
    with litke.LitkeRecording(path, drop_ttl=False) as rec_full:
        full = rec_full._read_raw_samples(0, n_samples)
    np.testing.assert_array_equal(got, full[:, 0])
    np.testing.assert_array_equal(got_slice, full[10:35, 0])


def test_litke_recording_roundtrip(tmp_path):
    rng = np.random.default_rng(2)
    n_samples, n_elec = 200, 8  # even: no special TTL packing path vs 519
    data = rng.integers(-400, 400, size=(n_samples, n_elec), dtype=np.int16)
    path = _write_litke_file(tmp_path / 'chunk.bin', data)

    with litke.LitkeRecording(path, drop_ttl=False) as rec:
        assert rec.shape == (n_samples, n_elec)
        assert rec.dtype == np.dtype(np.int16)
        assert rec.fs == 20000.0
        got = rec[0:n_samples]
        np.testing.assert_array_equal(got, data)
        # partial slice + channel index
        np.testing.assert_array_equal(rec[10:30, 2:5], data[10:30, 2:5])
        np.testing.assert_array_equal(rec[5], data[5])


def test_litke_recording_drops_ttl(tmp_path):
    rng = np.random.default_rng(3)
    n_samples, n_elec = 50, 7  # odd → TTL is channel 0
    data = rng.integers(-300, 300, size=(n_samples, n_elec), dtype=np.int16)
    path = _write_litke_file(tmp_path / 'odd.bin', data, array_id=504)

    with litke.LitkeRecording(path, drop_ttl=True) as rec:
        assert rec.shape == (n_samples, n_elec - 1)
        np.testing.assert_array_equal(rec[:], data[:, 1:])


def test_litke_multifile_folder(tmp_path):
    rng = np.random.default_rng(4)
    n_elec = 8
    a = rng.integers(-100, 100, size=(40, n_elec), dtype=np.int16)
    b = rng.integers(-100, 100, size=(25, n_elec), dtype=np.int16)
    folder = tmp_path / 'data000'
    folder.mkdir()
    # First file carries the header; subsequent files are body-only.
    header = _write_minimal_header(n_elec, 40 + 25)
    (folder / 'data000000.bin').write_bytes(
        header + litke.pack_samples(a).tobytes()
    )
    (folder / 'data000001.bin').write_bytes(litke.pack_samples(b).tobytes())

    with litke.LitkeRecording(folder, drop_ttl=False) as rec:
        assert rec.shape == (65, n_elec)
        np.testing.assert_array_equal(rec[0:40], a)
        np.testing.assert_array_equal(rec[40:65], b)
        # Span the file boundary
        span = rec[35:45]
        np.testing.assert_array_equal(span, np.concatenate([a[35:], b[:5]], 0))


def test_litke_as_binaryrw_file_object(tmp_path):
    rng = np.random.default_rng(5)
    n_samples, n_elec = 500, 8
    data = rng.integers(-200, 200, size=(n_samples, n_elec), dtype=np.int16)
    path = _write_litke_file(tmp_path / 'rw.bin', data)

    rec = litke.LitkeRecording(path, drop_ttl=True)
    n_chan = rec.shape[1]
    bfile = BinaryRWFile(
        filename=str(path),
        n_chan_bin=n_chan,
        fs=int(rec.fs),
        NT=100,
        nt=10,
        device=torch.device('cpu'),
        file_object=rec,
    )
    assert bfile.n_samples == n_samples
    X = bfile.padded_batch_to_torch(0)
    assert X.shape == (n_chan, 100 + 2 * 10)
    # Interior samples (after left pad of nt) match unpacked data.
    interior = X[:, 10:10 + 100].cpu().numpy().T.astype(np.int16)
    np.testing.assert_array_equal(interior, data[:100, 1:])
    rec.close()


def test_parse_header_rejects_garbage(tmp_path):
    p = tmp_path / 'nope.bin'
    p.write_bytes(b'not a litke file' + b'\x00' * 64)
    with pytest.raises(ValueError, match='Litke|header'):
        litke.LitkeRecording(p)


def test_electrode0_is_ttl_dropped_from_sorting_stream(tmp_path):
    """Electrode 0 is stim TTL, not spikes — must not enter KS file_object.

    Lab contract: converter writes samples[:, 1:] only. drop_ttl=True must
    match that; get_ttl always returns electrode 0 even when drop_ttl=True.
    """
    rng = np.random.default_rng(7)
    n_samples, n_elec = 80, 7  # odd → TTL packed as 16-bit ch0
    # Distinct TTL waveform vs neural-looking noise
    ttl = np.zeros(n_samples, dtype=np.int16)
    ttl[10:20] = -2000
    ttl[20:30] = 0
    ttl[40:50] = -2000
    rec_ch = rng.integers(-100, 100, size=(n_samples, n_elec - 1), dtype=np.int16)
    data = np.concatenate([ttl[:, None], rec_ch], axis=1)
    path = _write_litke_file(tmp_path / 'ttl.bin', data, array_id=504)

    with litke.LitkeRecording(path, drop_ttl=True) as rec:
        assert rec.shape == (n_samples, n_elec - 1)
        # Sorting stream is neural only
        np.testing.assert_array_equal(rec[:], rec_ch)
        # TTL still available separately
        np.testing.assert_array_equal(rec.get_ttl(), ttl)
        np.testing.assert_array_equal(rec.get_ttl(5, 10), ttl[5:15])
        out = rec.save_ttl(tmp_path / 'ttl_chan0.npy')
        np.testing.assert_array_equal(np.load(out), ttl)
        # Lab-style rising edges: below[:-1] & above[1:] → index of last
        # sample still < -thr (convert_litke_to_kilosort convention).
        onsets = rec.detect_ttl_onsets(threshold=1000)
        # pulses end at 20 and 50 → reported indices 19 and 49
        np.testing.assert_array_equal(onsets, np.array([19, 49], dtype=np.int64))


def test_detect_ttl_onsets_chunk_boundary(tmp_path):
    """Onsets spanning read chunks must not be lost."""
    n_samples, n_elec = 50, 7
    ttl = np.full(n_samples, -2000, dtype=np.int16)
    ttl[25:] = 0  # sample 24 still low, 25 high → lab index 24
    rec_ch = np.zeros((n_samples, n_elec - 1), dtype=np.int16)
    data = np.concatenate([ttl[:, None], rec_ch], axis=1)
    path = _write_litke_file(tmp_path / 'edge.bin', data, array_id=504)
    with litke.LitkeRecording(path, drop_ttl=True) as rec:
        onsets = rec.detect_ttl_onsets(threshold=1000, chunk_samples=10)
        np.testing.assert_array_equal(onsets, np.array([24], dtype=np.int64))


# ---------------------------------------------------------------------------
# Real-data / lab-oracle fixtures (not self-pack roundtrips)
# ---------------------------------------------------------------------------
# these .npz files were produced once by decoding real Litke bytes (or
# packing with lab bin2py_cythonext) and freezing the lab ground truth. They
# catch self-consistent-but-wrong nibble/sign/interleave bugs that pack∋unpack
# identity tests cannot see.

_DATA = Path(__file__).resolve().parent / 'data'


def test_real_519_unpack_matches_bin2py_oracle():
    """Real 20251204A packed bytes must decode bit-exact to lab bin2py.

    Fixture: tests/data/litke_real_519_bin2py_oracle.npz
    packed_uint8 = exact mid-recording file bytes from data000;
    expected_int16 = bin2py_cythonext.unpack_bin_even_num_electrodes.
    """
    path = _DATA / 'litke_real_519_bin2py_oracle.npz'
    assert path.is_file(), f'missing oracle fixture {path}'
    z = np.load(path)
    packed = z['packed_uint8']
    expected = z['expected_int16']
    n_samples = int(z['n_samples'])
    n_elec = int(z['n_electrodes'])
    assert packed.size == n_samples * litke.bytes_per_sample(n_elec)
    assert expected.shape == (n_samples, n_elec)

    got = litke.unpack_samples(packed, n_samples, n_elec)
    np.testing.assert_array_equal(got, expected)

    # pure-Python path must agree too (catches numba-only skew)
    py = litke.unpack_samples_python(packed, n_samples, n_elec)
    np.testing.assert_array_equal(py, expected)

    # TTL-only path vs full oracle column 0 (real packed bytes)
    ttl = litke.unpack_ttl(packed, n_samples, n_elec)
    np.testing.assert_array_equal(ttl, expected[:, 0])


def test_odd_unpack_matches_bin2py_oracle():
    """Odd-electrode (512-board style) pack/unpack vs lab bin2py freeze."""
    path = _DATA / 'litke_odd_bin2py_oracle.npz'
    assert path.is_file(), f'missing oracle fixture {path}'
    z = np.load(path)
    packed = z['packed_uint8']
    expected = z['expected_int16']
    n_samples = int(z['n_samples'])
    n_elec = int(z['n_electrodes'])
    assert n_elec % 2 == 1

    got = litke.unpack_samples(packed, n_samples, n_elec)
    np.testing.assert_array_equal(got, expected)
    py = litke.unpack_samples_python(packed, n_samples, n_elec)
    np.testing.assert_array_equal(py, expected)
