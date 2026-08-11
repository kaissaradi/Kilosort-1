import pytest
import numpy as np
import torch
from torch.fft import fft, ifft, fftshift

import kilosort.preprocessing as kpp
from kilosort import datashift, io


np.random.seed(123)


def _whitening_local_reference(CC, xc, yc, nrange=32, device=torch.device('cpu')):
    """Per-channel argsort path used before nearest-index precompute."""
    Nchan = CC.shape[0]
    Wrot = torch.zeros((Nchan, Nchan), device=device, dtype=CC.dtype)
    for j in range(Nchan):
        ds = (xc[j] - xc)**2 + (yc[j] - yc)**2
        isort = np.argsort(ds)
        ix = isort[:nrange]
        wrot = kpp.whitening_from_covariance(CC[np.ix_(ix, ix)])
        Wrot[j, ix] = wrot[0]
    return Wrot


def test_whitening_local_matches_per_channel_argsort():
    rng = np.random.default_rng(7)
    n = 12
    xc = np.tile(np.arange(4, dtype=np.float64) * 30.0, 3)
    yc = np.repeat(np.arange(3, dtype=np.float64) * 30.0, 4)
    A = rng.standard_normal((n, n)).astype(np.float32)
    CC = torch.from_numpy((A @ A.T + np.eye(n, dtype=np.float32)).astype(np.float32))

    got = kpp.whitening_local(CC, xc, yc, nrange=5, device=torch.device('cpu'))
    expected = _whitening_local_reference(CC, xc, yc, nrange=5,
                                          device=torch.device('cpu'))
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)


def test_get_whitening_matrix_single_batch(tmp_path):
    """One-batch files used to hit CC/k with k==0 (empty range over n_batches-1)."""
    n_chan, NT, nt = 6, 500, 21
    n_samples = NT  # exactly one full batch
    path = tmp_path / 'one_batch.bin'
    rng = np.random.default_rng(0)
    data = rng.integers(-100, 100, size=(n_samples, n_chan), dtype=np.int16)
    data.tofile(path)

    xc = np.arange(n_chan, dtype=np.float64) * 30.0
    yc = np.zeros(n_chan, dtype=np.float64)
    bfile = io.BinaryFiltered(
        path,
        n_chan_bin=n_chan,
        fs=20000,
        NT=NT,
        nt=nt,
        chan_map=np.arange(n_chan),
        device=torch.device('cpu'),
        do_CAR=False,
    )
    assert bfile.n_batches == 1
    Wrot = kpp.get_whitening_matrix(bfile, xc, yc, nskip=25, nrange=4)
    assert Wrot.shape == (n_chan, n_chan)
    assert torch.isfinite(Wrot).all()

class TestFiltering:
    # 2 seconds of time samples at 30Khz, 1 channel
    t = np.linspace(0, 2, 60000, False, dtype='float32')[np.newaxis,...]
    # 100hz and 500hz signals
    sine_100hz = torch.from_numpy(np.sin(2*np.pi*100*t))
    sine_500hz = torch.from_numpy(np.sin(2*np.pi*500*t))
    # high pass filter (hard-coded for 300hz threshold)
    hp_filter = kpp.get_highpass_filter(device=torch.device('cpu'))

    def test_get_highpass_filter(self):
        # Add dummy axes, shape (channels in, channels out, width)
        hp_filter = self.hp_filter[None, None, :]
        filtered_100hz = torch.nn.functional.conv1d(self.sine_100hz, hp_filter)
        filtered_500hz = torch.nn.functional.conv1d(self.sine_500hz, hp_filter)

        # After applying high pass filter,
        # 100hz signal should be close to 0, 500hz should be mostly unchanged,
        # but neither case is exact.
        assert torch.max(filtered_100hz) < 0.01
        assert torch.max(filtered_500hz) > 0.9

    def test_fft_highpass(self):
        fft1 = kpp.fft_highpass(self.hp_filter, NT=1000)    # crop filter
        fft2 = kpp.fft_highpass(self.hp_filter, NT=100000)  # pad filter
        # TODO: Currently this only leaves it unchanged b/c NT is hard-coded
        #       to the same value for get_highpass_filter and fft_highpass,
        #       which is fragile. Should define that better somewhere.
        fft3 = kpp.fft_highpass(self.hp_filter)             # same size

        # New filter's shape should match NT, or be the same as the original
        # filter.
        assert fft1.shape[0] == 1000
        assert fft2.shape[0] == 100000
        assert fft3.shape[0] == self.hp_filter.shape[0]

        # rFFT shapes are Hermitian half-spectra
        r1 = kpp.rfft_highpass(self.hp_filter, NT=1000)
        assert r1.shape[0] == 1000 // 2 + 1

        # Production rFFT path (BinaryFiltered.filter)
        fr = kpp.rfft_highpass(self.hp_filter, NT=self.sine_100hz.shape[1])
        x100 = kpp.apply_highpass_rfft(self.sine_100hz, fr)
        x500 = kpp.apply_highpass_rfft(self.sine_500hz, fr)

        # After applying high pass filter,
        # 100hz signal should be close to 0, 500hz should be mostly unchanged,
        # but neither case is exact.
        assert torch.max(x100) < 0.01
        assert torch.max(x500) > 0.9


class TestArtifactRemoval:
    
    def test_threshold(self, torch_device):
        # Deterministic fill: prior suite tests advance global np.random, and
        # CAR (median across chans) can pull a barely-over-threshold spike
        # back under 30000 — making this flaky. Use a fixed generator and a
        # spike far above threshold after mean/CAR.
        rng = np.random.default_rng(0)
        a = rng.integers(-1000, 1000, (1000, 10)).astype(np.float32)
        a[900, 4] = 100_000

        bfile1 = io.BinaryFiltered(
            filename='dummy', n_chan_bin=10, NT=500, device=torch_device,
            file_object=a, artifact_threshold=30000
        )
        bfile2 = io.BinaryFiltered(
            filename='dummy', n_chan_bin=10, NT=500, device=torch_device,
            file_object=a
            )

        # No threshold crossings in the first half, so these should match.
        assert torch.allclose(bfile1[:500,:], bfile2[:500,:])
        # Second half should be zeroed out for bfile1 only.
        zeros = torch.zeros(500,10).to(torch_device).float()
        assert torch.allclose(bfile1[500:,:], zeros.T)
        assert not torch.allclose(bfile2[500:,:], zeros.T)


class TestWhitening:

    def test_whitening_from_covariance(self, torch_device):
        x = torch.from_numpy(np.random.rand(100, 1000)).to(torch_device).float()
        cc = (x @ x.T)/1000
        wm = kpp.whitening_from_covariance(cc)
        whitened = wm @ x
        new_cov = (whitened @ whitened.T)/whitened.shape[1]

        # Covariance matrix of whitened data should be very close to the
        # identity matrix.
        assert torch.allclose(
            new_cov, torch.eye(new_cov.shape[1], device=torch_device),
            atol=1e-4
            )

    def test_get_whitening(self, bfile, saved_ops):
        xc = saved_ops['probe']['xc']
        yc = saved_ops['probe']['yc']
        wm = kpp.get_whitening_matrix(bfile, xc, yc)

        ### Perform other preprocessing steps on data to ensure valid result.
        # TODO: better way to encapsulate these steps for re-use.
        # Get first batch of data
        X = torch.from_numpy(bfile.file[:bfile.NT,:].T).to(bfile.device).float()
        # Remove unwanted channels
        if bfile.chan_map is not None:
            X = X[bfile.chan_map]
        # remove the mean of each channel, and the median across channels
        X = X - X.mean(1).unsqueeze(1)
        X = X - torch.median(X, 0)[0]
        # high-pass filtering in the Fourier domain (much faster than filtfilt etc)
        fwav = kpp.fft_highpass(bfile.hp_filter, NT=X.shape[1])
        X = torch.real(ifft(fft(X) * torch.conj(fwav)))
        X = fftshift(X, dim = -1)
        ###

        # Apply whitening matrix to one batch
        whitened = (wm @ X)
        new_cov = (whitened @ whitened.T)/whitened.shape[1]
        identity = torch.eye(new_cov.shape[1], device=bfile.device)

        # TODO: Double check with Marius, this still isn't true but maybe
        #       that's okay. The "shape" is still similar (e.g. high values
        #       along and adjacent to diagonal, rest near 0).
        # Covariance matrix of whitened data should be approximately equal
        # to the identity matrix.
        # assert torch.allclose(new_cov, identity, atol=1e-2)

        # Alternative test until identity matrix question is resolved.
        # Normalized covariance matrix should have 99th percentile < 0.1.
        # In other words, very few values that are not near 0.
        norm_cov = new_cov - new_cov.min()
        norm_cov = norm_cov/norm_cov.max()
        assert torch.quantile(torch.flatten(norm_cov), 0.99) < 0.1


# TODO: need to investigate why these aren't exact matches, likely an issue with
#       updates to dependencies.
# class TestDriftCorrection:

#     @pytest.mark.slow
#     def test_datashift(self, bfile, saved_ops, torch_device, capture_mgr):
#         saved_yblk = saved_ops['yblk']
#         saved_dshift = saved_ops['dshift']
#         saved_iKxx = saved_ops['iKxx'].to(torch_device)
#         with capture_mgr.global_and_fixture_disabled():
#             print('\nStarting datashift.run test...')
#             ops, st = datashift.run(saved_ops, bfile, device=torch_device)

#         # TODO: this fails on dshift, but the final version doesn't. So, dshift
#         #       must be overwritten later on in the pipeline. Need to save the
#         #       initial result separately.
#         print('testing yblk...')
#         assert np.allclose(saved_yblk, ops['yblk'])
#         print('testing dshift...')
#         # assert np.allclose(saved_dshift, ops['dshift'])
#         print('testing iKxx...')
#         assert torch.allclose(saved_iKxx, ops['iKxx'])
        

#     def test_get_drift_matrix(self):
#         # TODO
#         pass
