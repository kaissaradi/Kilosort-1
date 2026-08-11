"""Identity and smoke tests for native Litke IO (kilosort.litke)."""

import struct

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
