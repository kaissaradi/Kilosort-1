import pytest
import tempfile
from pathlib import Path

import numpy as np
import torch

from kilosort import io


def test_fwav_cache_bit_identical_across_batches(torch_device):
    """Fourier high-pass cache must match recompute-per-batch results."""
    from kilosort.preprocessing import get_highpass_filter

    torch.manual_seed(0)
    n_chan, nt_len = 4, 512
    hp = get_highpass_filter(fs=20000, cutoff=300, device=torch_device)
    # Minimal BinaryFiltered without a real file: construct then inject tensors.
    bf = io.BinaryFiltered.__new__(io.BinaryFiltered)
    bf.chan_map = None
    bf.whiten_mat = None
    bf.hp_filter = hp
    bf.dshift = None
    bf.do_CAR = False
    bf.invert_sign = False
    bf.artifact_threshold = np.inf
    bf.device = torch_device
    bf._fwav_cache = None
    bf._fwav_cache_nt = None

    X0 = torch.randn(n_chan, nt_len, device=torch_device)
    # First call populates cache
    y1 = bf.filter(X0.clone())
    assert bf._fwav_cache is not None
    assert bf._fwav_cache_nt == nt_len
    # Second call reuses cache — bit-identical to first on same input
    y2 = bf.filter(X0.clone())
    assert torch.equal(y1, y2)
    # Bust cache and recompute explicitly
    bf._fwav_cache = None
    bf._fwav_cache_nt = None
    y3 = bf.filter(X0.clone())
    assert torch.equal(y1, y3)


def test_probe_io():
    # Create one-column probe with 5 contacts, spaced 1um apart.
    json_probe = {
        'chanMap': np.arange(5),
        'xc': np.ones(5),
        'yc': np.arange(5),
        'kcoords': np.zeros(5),
        'n_chan': 5
    }
    # Repeat in .prb format
    prb_probe = """
channel_groups = {
    0: {
            'channels' : [0,1,2,3,4],
            'geometry': {
                0: [1, 0],
                1: [1, 1],
                2: [1, 2],
                3: [1, 3],
                4: [1, 4]
            }
    }
}
"""
    
    # Save both to file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json_file = Path(f.name)
        print(json_file)
        io.save_probe(json_probe, json_file)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.prb', delete=False) as f:
        f.write(prb_probe)
        prb_file = Path(f.name)

    # Load both with kilosort.io
    probe1 = io.load_probe(json_file)
    probe2 = io.load_probe(prb_file)

    print('probe1:')
    print(probe1)
    print('probe2:')
    print(probe2)

    try:
        # Verify that loaded probes contain the same information
        for k in ['chanMap', 'xc', 'yc', 'kcoords', 'n_chan']:
            print(f'testing key {k}')
            assert (k in probe1) and (k in probe2)
            assert np.all(probe1[k] == probe2[k])
    finally:
        # Remove temporary files
        json_file.unlink()
        prb_file.unlink()


def test_bad_channels():
    probe = {
        'xc': np.zeros(5), 'yc': np.arange(5)*10, 'kcoords': np.zeros(5),
        'chanMap': np.arange(5), 'n_chan': 5
        }
    bad_channels = [2, 4]
    probe2 = io.remove_bad_channels(probe, bad_channels)
    assert probe2['n_chan'] == 3
    for k in ['xc', 'yc', 'kcoords']:
        assert probe2[k].size == 3
    assert np.all(probe2['chanMap'] == [0, 1, 3])
    assert np.all(probe['chanMap'] == [0, 1, 2, 3, 4])  # original unchanged

    with pytest.raises(IndexError):
        # These are not in the channel map.
        _ = io.remove_bad_channels(probe, [5, 6])


def test_bat_extension(torch_device, data_directory):
    # Create memmap, write to file, close the file again.
    path = data_directory / 'binary_test' / 'temp_memmap.bat'
    path.parent.mkdir(parents=True, exist_ok=True)
    N, C = (1000, 10)
    r = np.random.rand(N,C)*2 - 1    # scale to (-1,1)
    r = (r*(2**14)).astype(np.int16)     # scale up

    try:
        a = np.memmap(path, mode='w+', shape=(N,C), dtype=np.int16)
        a[:] = r[:]
        a.flush()
        del(a)

        directory = path.parent
        filename = io.find_binary(directory)
        assert filename == path
        bfile = io.BinaryFiltered(filename, C, device=torch_device)
        x = bfile[0:100]  # Test data retrieval

    finally:
        # Delete memmap file and re-raise exception
        path.unlink()


def test_dat_extension(torch_device, data_directory):
    # Create memmap, write to file, close the file again.
    path = data_directory / 'binary_test' / 'temp_memmap.dat'
    path.parent.mkdir(parents=True, exist_ok=True)
    N, C = (1000, 10)
    r = np.random.rand(N,C)*2 - 1    # scale to (-1,1)
    r = (r*(2**14)).astype(np.int16)     # scale up

    try:
        a = np.memmap(path, mode='w+', shape=(N,C), dtype=np.int16)
        a[:] = r[:]
        a.flush()
        del(a)

        directory = path.parent
        filename = io.find_binary(directory)
        assert filename == path
        bfile = io.BinaryFiltered(filename, C, device=torch_device)
        x = bfile[0:100]  # Test data retrieval

    finally:
        # Delete memmap file and re-raise exception
        path.unlink()


def test_tmin_tmax(torch_device, data_directory):
    N, C = (1000, 10)
    NT = 300
    nt = 61
    data = np.repeat(np.arange(N)[...,np.newaxis], repeats=C, axis=1)
    fs = 10
    path = data_directory / 'time_interval_test' / 'temp_memmap.dat'
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        a = np.memmap(path, mode='w+', shape=(N,C), dtype=np.int16)
        a[:] = data[:]
        a.flush()
        del(a)

        bfile = io.BinaryRWFile(path, n_chan_bin=C, fs=fs, device=torch_device,
                                tmin=10, tmax=85, NT=NT, nt=nt)

        assert bfile.imin == 100
        assert bfile.imax == 850
        assert bfile.n_samples == 750
        assert bfile[0:50].min() == 100
        assert bfile[700:750].max() == 849
        assert bfile.n_batches == 3

        X0 = bfile.padded_batch_to_torch(ibatch=0)
        assert X0.min() == 100
        assert X0.max() == 100 + NT + nt - 1

        X1 = bfile.padded_batch_to_torch(ibatch=1)
        assert X1.min() == 100 + NT - nt
        assert X1.max() == 100 + 2*NT + nt - 1

        X2 = bfile.padded_batch_to_torch(ibatch=2)
        assert X2.min() == 100 + 2*NT - nt
        assert X2.max() == 849

    finally:
        # Delete memmap file and re-raise exception
        path.unlink()


def test_tmin_only(torch_device, data_directory):
    N, C = (1000, 10)
    NT = 300
    nt = 61
    data = np.repeat(np.arange(N)[...,np.newaxis], repeats=C, axis=1)
    fs = 10
    path = data_directory / 'time_interval_test' / 'temp_memmap2.dat'
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        a = np.memmap(path, mode='w+', shape=(N,C), dtype=np.int16)
        a[:] = data[:]
        a.flush()
        del(a)

        bfile = io.BinaryRWFile(path, n_chan_bin=C, fs=fs, device=torch_device,
                                tmin=43, NT=NT, nt=nt)

        assert bfile.imin == 430
        assert bfile.imax == 1000
        assert bfile.n_samples == 570
        assert bfile[0:10].min() == 430
        assert bfile[400:].max() == 999
        assert bfile.n_batches == 2

    finally:
        # Delete memmap file and re-raise exception
        path.unlink()


def test_tmax_only(torch_device, data_directory):
    N, C = (1000, 10)
    NT = 300
    nt = 61
    data = np.repeat(np.arange(N)[...,np.newaxis], repeats=C, axis=1)
    fs = 10
    path = data_directory / 'time_interval_test' / 'temp_memmap3.dat'
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        a = np.memmap(path, mode='w+', shape=(N,C), dtype=np.int16)
        a[:] = data[:]
        a.flush()
        del(a)

        bfile = io.BinaryRWFile(path, n_chan_bin=C, fs=fs, device=torch_device,
                                tmax=78, NT=NT, nt=nt)

        assert bfile.imin == 0
        assert bfile.imax == 780
        assert bfile.n_samples == 780
        assert bfile[0:100].min() == 0
        assert bfile[700:].max() == 779
        assert bfile.n_batches == 3

    finally:
        # Delete memmap file and re-raise exception
        path.unlink()


def test_file_group(torch_device, data_directory, bfile):
    file = data_directory / 'ZFM-02370_mini.imec0.ap.short.bin'
    fs = bfile.fs
    n_chans = bfile.n_chan_bin

    # Test with file_objects option
    # Load as three 15-second files instead of one 45-second file.
    objs = [np.memmap(file, dtype='int16', shape=bfile.shape, mode='r')
            for _ in range(3)]
    objs[0] = objs[0][:int(15*fs),:]
    objs[1] = objs[1][int(15*fs):int(30*fs),:]
    objs[2] = objs[2][int(30*fs):,:]
    bfg = io.BinaryFileGroup(file_objects=objs)
    bfile2 = io.BinaryFiltered(
        filename='test', n_chan_bin=n_chans, fs=fs, chan_map=bfile.chan_map,
        device=torch_device, file_object=bfg, dtype='int16'
        )

    # First batch, overlapping batch, and last batch
    # (assumes 45s test dataset with 2s batch size)
    for i in [0, 7, 22]:
        b1 = bfile.padded_batch_to_torch(i, skip_preproc=True)
        b2 = bfile2.padded_batch_to_torch(i)
        assert torch.allclose(b1, b2)

    # Test with filenames option
    files = [file]*3  # Load the same data three times
    bfile3 = io.BinaryFiltered(
        filename=files, n_chan_bin=n_chans, fs=fs, chan_map=bfile.chan_map,
        device=torch_device, dtype='int16'
    )

    # First and first, last and last, last of original and last of concat
    for i,j in [(0,0), (21,21), (22,67)]:
        b1 = bfile.padded_batch_to_torch(i, skip_preproc=True)
        b2 = bfile3.padded_batch_to_torch(j)
        assert torch.allclose(b1, b2)


def test_downsampling(bfile):
    b0a = bfile.padded_batch_to_torch(0)
    b5a = bfile.padded_batch_to_torch(5)
    b15a = bfile.padded_batch_to_torch(15)
    nba = bfile.n_batches
    bfile.set_downsampling(3)
    b0b = bfile.padded_batch_to_torch(0)
    b5b = bfile.padded_batch_to_torch(5)
    nbb = bfile.n_batches

    # First batch should be the same for both.
    assert torch.allclose(b0a, b0b)
    # Same batch index should be different since the latter skips batches.
    assert not torch.allclose(b5a, b5b)
    # But batch(i) should be the same as batch(j*3)
    assert torch.allclose(b15a, b5b)
    assert nbb <= 3*nba


def test_short_last_batch_does_not_crash(tmp_path):
    """Last batch shorter than `nt` used to AttributeError on n_batches.

    BinaryRWFile adjusted `self.n_batches` before set_downsampling created
    that attribute. The drop must apply to n_batches_raw, then n_batches is
    derived. MEA short fixtures and truncated tmax runs hit this path.
    """
    n_chan = 4
    NT, nt = 100, 10
    # With these params, n_samples=209 yields 3 raw batches and a too-short
    # final batch (reproduced crash before the fix).
    n_samples = 209
    path = tmp_path / 'short_tail.bin'
    np.zeros((n_samples, n_chan), dtype=np.int16).tofile(path)

    bfile = io.BinaryRWFile(
        path, n_chan_bin=n_chan, fs=1000, NT=NT, nt=nt, device=torch.device('cpu'),
    )
    assert bfile.n_batches_raw == 2
    assert bfile.n_batches == 2
    # Dropped 9 residual samples so the last kept batch is valid.
    assert bfile.imax == 200
    assert bfile.n_samples == 200
    # Must be able to read every reported batch.
    for i in range(bfile.n_batches):
        X = bfile.padded_batch_to_torch(i)
        assert X.shape == (n_chan, NT + 2 * nt)


def test_empty_file_has_zero_batches(tmp_path):
    """Zero-sample file must not call _get_batch_edges(-1) / go negative."""
    path = tmp_path / 'empty.bin'
    path.write_bytes(b'')
    bfile = io.BinaryRWFile(
        path, n_chan_bin=4, fs=1000, NT=100, nt=10, device=torch.device('cpu'),
        dtype='int16',
    )
    assert bfile.n_batches_raw == 0
    assert bfile.n_batches == 0
    assert bfile.n_samples == 0


def test_single_batch_right_edge_replicate(tmp_path):
    """When n_batches==1, right pad must edge-replicate (not stay zeros).

    First-batch path left-pads only; without a single-batch branch the tail
    after real samples is torch.zeros — a hard discontinuity for filters.
    Multi-batch first/last behaviour must stay unchanged.
    """
    n_chan = 4
    NT, nt = 100, 10
    rng = np.random.default_rng(0)

    # --- single batch (n_samples < NT) ---
    n_samples = 80
    data = rng.integers(-200, 200, size=(n_samples, n_chan), dtype=np.int16)
    path = tmp_path / 'single.bin'
    data.tofile(path)
    b1 = io.BinaryRWFile(
        path, n_chan_bin=n_chan, fs=1000, NT=NT, nt=nt, device=torch.device('cpu'),
    )
    assert b1.n_batches == 1
    X = b1.padded_batch_to_torch(0)
    assert X.shape == (n_chan, NT + 2 * nt)
    # Interior matches file (placed at columns nt : nt+n_samples)
    interior = X[:, nt:nt + n_samples].cpu().numpy().T.astype(np.int16)
    np.testing.assert_array_equal(interior, data)
    # Left pad = edge replicate of first real sample
    assert torch.equal(X[:, :nt], X[:, nt:nt + 1].expand(-1, nt))
    # Right pad = edge replicate of last real sample (not zeros)
    end = nt + n_samples
    right = X[:, end:]
    assert right.shape[1] > 0
    assert not torch.all(right == 0)
    assert torch.equal(right, X[:, end - 1:end].expand_as(right))

    # --- multi-batch identity: first batch left-pads; last batch right-pads ---
    n_multi = NT * 3 + 50
    data_m = rng.integers(-200, 200, size=(n_multi, n_chan), dtype=np.int16)
    path_m = tmp_path / 'multi.bin'
    data_m.tofile(path_m)
    bm = io.BinaryRWFile(
        path_m, n_chan_bin=n_chan, fs=1000, NT=NT, nt=nt, device=torch.device('cpu'),
    )
    assert bm.n_batches > 1
    X0 = bm.padded_batch_to_torch(0)
    assert torch.equal(X0[:, :nt], X0[:, nt:nt + 1].expand(-1, nt))
    # First multi-batch still has real right context from NT+nt read — not zeros
    assert not torch.all(X0[:, -nt:] == 0)
    Xlast = bm.padded_batch_to_torch(bm.n_batches - 1)
    # Last batch right-pads by edge replicate of its last loaded sample
    # Find first all-zero-or-replicated tail: last column equals previous
    assert torch.equal(Xlast[:, -1:], Xlast[:, -2:-1])
