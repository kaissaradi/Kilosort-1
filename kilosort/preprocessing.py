import torch, os, scipy
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
from glob import glob
from torch.fft import fft, ifft, rfft, irfft, fftshift

def whitening_from_covariance(CC):
    """Whitening matrix for a covariance matrix CC.

    This is the so-called ZCA whitening matrix.

    """
    E,D,V =  torch.linalg.svd(CC)
    eps = 1e-6
    Wrot =(E / (D+eps)**.5) @ E.T
    return Wrot

# --- robust whitening covariance (KS4_ROBUST_COV) --------------------------
#
# Background: get_whitening_matrix's covariance CC = X @ X.T / T is a plain
# sample second moment over EVERY time sample, spikes included. A channel
# that carries a lot of real spikes reads as high-variance and gets whitened
# down harder than a dead channel, so the post-whitening noise floor ends up
# LOWER on the busiest channels -- see the whitening-inverts-the-threshold
# memory (measured 1.65x on one file/geometry). A prior attempt to fix this
# (KS4_MADW, in an earlier fork iteration -- not present in this tree) rescaled
# the finished Wrot by each channel's post-whitening MAD. That was a post-hoc
# patch on the OUTPUT of whitening_local, not a change to how CC was formed,
# and it was refuted on real ground truth on three separate axes (recall
# wash, a precision "win" that was really dropped real spikes, and worse
# refractory purity) -- see madw-whitening-fix-works (a retraction).
#
# This is a different, structural fix: downweight high-amplitude SAMPLES
# before they ever enter the covariance sum, so CC itself estimates the
# noise-only second moment instead of noise + a spike-inflated tail. It is an
# IRLS-style Huber M-estimator of the (zero-mean) scatter matrix:
#   1. Standardize each channel by a robust scale (median(|x|) as a
#      zero-median MAD proxy -- valid because the data is high-pass filtered).
#   2. Score every SAMPLE (a full array time-slice) by its RMS across
#      standardized channels -- this catches genuinely large multichannel
#      events (spikes) without singling out any one channel's baseline
#      variance, which is the failure mode of a per-channel-only rescale.
#   3. Huber-weight samples above a robust multiple of the typical score:
#      full weight below the cutoff, weight shrinking as 1/score above it.
#   4. Re-derive the channel scales using the new weights and repeat a few
#      times (2-3 iterations is enough for this to stabilize; it is not a
#      hot loop -- runs once per whitening batch, batches are subsampled by
#      nskip already).
#   5. Form CC as the WEIGHTED second moment, not X @ X.T unweighted.
#
# c_huber=3.0 and n_iters=3 are not derived from first principles -- they are
# a standard "moderate" Huber cutoff (real cortical/retinal spikes commonly
# exceed 3-5x the noise RMS; genuine Gaussian noise essentially never does)
# picked to be conservative rather than fit to any bench. Flag this in any
# report: it is a reasonable default, not a validated constant.
#
# Off by default. On (KS4_ROBUST_COV=1) it changes ONLY the covariance fed to
# whitening_local -- whitening_local itself, Wrot's shape, and every
# downstream consumer are untouched.

_ROBUST_COV_C_HUBER = 3.0
_ROBUST_COV_N_ITERS = 3
_ROBUST_COV_EPS = 1e-8


def _weighted_median_abs(X, w):
    """Weighted median of |X| along the time axis, per channel.

    X: (n_chan, T). w: (T,) nonnegative sample weights (broadcast across
    channels -- weight is a property of the time sample, not the channel).
    Returns (n_chan,). Assumes X is already ~zero-median (true for high-pass
    filtered MEA data), so median(|X|) is used directly as the MAD rather
    than subtracting a per-channel median first.
    """
    n_chan, T = X.shape
    absX = X.abs()
    order = torch.argsort(absX, dim=1)
    x_sorted = torch.gather(absX, 1, order)
    w_row = w.unsqueeze(0).expand(n_chan, -1)
    w_sorted = torch.gather(w_row, 1, order)
    cw = torch.cumsum(w_sorted, dim=1)
    total = cw[:, -1:].clamp_min(_ROBUST_COV_EPS)
    half = total / 2
    # index of the first sorted sample whose cumulative weight reaches half
    idx = torch.searchsorted(cw.contiguous(), half.contiguous())
    idx = idx.clamp(max=T - 1)
    return torch.gather(x_sorted, 1, idx).squeeze(1)


def _robust_sample_weights(X, n_iters=_ROBUST_COV_N_ITERS, c=_ROBUST_COV_C_HUBER):
    """IRLS Huber sample weights, in [0, 1], one per time sample of X.

    X: (n_chan, T) high-pass filtered batch. Low weight marks samples whose
    across-channel RMS (after robust per-channel standardization) is large
    relative to the bulk of the batch -- i.e. likely spikes, not noise.
    """
    T = X.shape[1]
    device = X.device
    w = torch.ones(T, device=device, dtype=X.dtype)
    for _ in range(max(1, n_iters)):
        sigma_c = _weighted_median_abs(X, w) / 0.6745
        sigma_c = sigma_c.clamp_min(_ROBUST_COV_EPS)
        Z = X / sigma_c[:, None]
        r = torch.sqrt((Z ** 2).mean(dim=0))  # (T,) RMS across channels
        med_r = r.median()
        mad_r = (r - med_r).abs().median() / 0.6745
        mad_r = mad_r.clamp_min(_ROBUST_COV_EPS)
        thresh = med_r + c * mad_r
        w = (thresh / r.clamp_min(_ROBUST_COV_EPS)).clamp(max=1.0)
    return w


def robust_batch_covariance(X, n_iters=_ROBUST_COV_N_ITERS, c=_ROBUST_COV_C_HUBER):
    """Huber-weighted covariance of a batch, robust to spike contamination.

    X: (n_chan, T). Returns (n_chan, n_chan), normalized to match the scale
    of the stock ``(X @ X.T) / X.shape[1]`` estimator (weights sum to T in
    expectation for an uncontaminated batch, so dividing by sum(w) keeps the
    two estimators on a comparable scale when there is nothing to downweight).
    """
    w = _robust_sample_weights(X, n_iters=n_iters, c=c)
    Xw = X * w[None, :]
    denom = w.sum().clamp_min(_ROBUST_COV_EPS)
    return (Xw @ X.T) / denom


def whitening_local(CC, xc, yc, nrange=32, device=torch.device('cuda')):
    """Compute whitening filter for each channel based on nearest channels."""
    Nchan = CC.shape[0]
    Wrot = torch.zeros((Nchan, Nchan), device=device, dtype=CC.dtype)

    # Precompute nearest-channel indices once. The old path re-sorted the full
    # Nchan×Nchan distance matrix on every center channel (O(N² log N) work).
    # Keep full argsort (not argpartition): regular MEA grids have many
    # equal distances, and partition tie-breaking would change which contacts
    # land in the whitening neighborhood.
    xc = np.asarray(xc, dtype=np.float64)
    yc = np.asarray(yc, dtype=np.float64)
    ds = (xc[:, None] - xc[None, :])**2 + (yc[:, None] - yc[None, :])**2
    nrange = int(min(nrange, Nchan))
    nearest = np.argsort(ds, axis=1)[:, :nrange]

    # for each channel, a local covariance matrix is extracted
    # the whitening matrix is computed for that local neighborhood
    for j in range(Nchan):
        ix = nearest[j]

        wrot = whitening_from_covariance(CC[np.ix_(ix, ix)])

        # the first row of wrot is a whitening vector for the center channel
        Wrot[j, ix] = wrot[0]
    return Wrot

def kernel2D_torch(x, y, sig = 1):
    """Simple Gaussian kernel for two sets of coordinates x and y."""
    ds = ((x.unsqueeze(1) - y)**2).sum(-1)
    Kn = torch.exp(-ds / (2*sig**2))
    return Kn


def get_drift_matrix(ops, dshift, device=torch.device('cuda')):
    """For given dshift drift, compute linear drift matrix for interpolation."""

    # first, interpolate drifts to every channel
    yblk = ops['yblk']
    if ops['nblocks'] == 1:
        shifts = dshift
    else:
        finterp = interp1d(yblk, dshift, fill_value="extrapolate", kind = 'linear')
        shifts = finterp(ops['probe']['yc'])

    # compute coordinates of desired interpolation
    xp = np.vstack((ops['probe']['xc'],ops['probe']['yc'])).T
    yp = xp.copy()
    yp[:,1] -= shifts

    xp = torch.from_numpy(xp).to(device)
    yp = torch.from_numpy(yp).to(device)

    # the kernel is radial symmetric based on distance
    Kyx = kernel2D_torch(yp, xp, ops['settings']['sig_interp'])
    
    # multiply with precomputed inverse kernel matrix of original channels
    M = Kyx @ ops['iKxx']

    return M


def get_fwav(NT = 30122, fs = 30000, device=torch.device('cuda')):
    """Precomputes a filter to use for high-pass filtering.
    
    To be used with fft in pytorch. Currently depends on NT,
    but it could get padded for larger NT.

    """

    # a butterworth filter is specified in scipy
    b,a = butter(3, 300, fs = fs, btype = 'high')
    
    # a signal with a single entry is used to compute the impulse response
    x = np.zeros(NT)
    x[NT//2] = 1
    
    # symmetric filter from scipy
    wav = filtfilt(b,a , x).copy()
    wav = torch.from_numpy(wav).to(device).float()

    # the filter will be used directly in the Fourier domain
    fwav = fft(wav)

    return fwav

def get_whitening_matrix(f, xc, yc, nskip=25, nrange=32):
    """Get the whitening matrix, use every nskip batches.

    KS4_ROBUST_COV=1 (opt-in, off by default): downweight high-amplitude
    samples before they enter the covariance sum, via an IRLS Huber
    M-estimator (see robust_batch_covariance above). This targets the
    documented whitening-inverts-the-threshold defect -- CC computed from raw
    spikes-included data makes busy channels read as high-variance and get
    whitened down harder than dead channels. Off by default: reproduces stock
    output exactly (verified byte-identical in tests/test_mea_fork.py).
    Synthetic-only validation so far -- no real-GT confirmation, see the
    commit message.
    """
    n_chan = len(f.chan_map)
    # collect the covariance matrix across channels
    CC = torch.zeros((n_chan, n_chan), device=f.device)
    k = 0
    _robust = os.environ.get('KS4_ROBUST_COV', '').strip().lower() not in (
        '', '0', 'false', 'no',
    )
    # Historical loop skipped the final batch (`range(0, n_batches-1, ...)`).
    # On single-batch fixtures that made the range empty and divided by k==0.
    # Use at least batch 0; when n_batches>1 keep the same upper bound as before.
    n_scan = max(1, int(f.n_batches) - 1)
    for j in range(0, n_scan, nskip):
        # load data with high-pass filtering (see the Binary file class)
        X = f.padded_batch_to_torch(j)

        # remove padding
        X = X[:, f.nt : -f.nt]

        # cumulative covariance matrix
        if _robust:
            CC = CC + robust_batch_covariance(X)
        else:
            CC = CC + (X @ X.T)/X.shape[1]

        k += 1

    if k == 0:
        raise ValueError(
            'get_whitening_matrix found no batches to average; '
            f'n_batches={f.n_batches}'
        )
    CC = CC / k

    # compute the local whitening filters and collect back into Wrot
    Wrot = whitening_local(CC, xc, yc, nrange=nrange, device=f.device)

    return Wrot

def get_highpass_filter(fs=30000, cutoff=300, device=torch.device('cuda')):
    """Filter to use for high-pass filtering."""
    NT = 30122
    
    # a butterworth filter is specified in scipy
    b,a = butter(3, cutoff, fs=fs, btype='high')

    # a signal with a single entry is used to compute the impulse response
    x = np.zeros(NT)
    x[NT//2] = 1

    # symmetric filter from scipy
    hp_filter = filtfilt(b, a , x).copy()
    
    hp_filter = torch.from_numpy(hp_filter).to(device).float()
    return hp_filter

def _pad_or_crop_filter(hp_filter, NT):
    """Pad/crop the time-domain high-pass impulse response to length NT."""
    device = hp_filter.device
    ft = int(hp_filter.shape[0])
    NT = int(NT)
    if ft < NT:
        pad = (NT - ft) // 2
        return torch.cat((
            torch.zeros(pad, device=device, dtype=hp_filter.dtype),
            hp_filter,
            torch.zeros(pad + (NT - pad * 2 - ft), device=device,
                        dtype=hp_filter.dtype),
        ))
    if ft > NT:
        crop = (ft - NT) // 2
        return hp_filter[crop: crop + NT]
    return hp_filter


def fft_highpass(hp_filter, NT=30122):
    """Convert filter to full (complex) Fourier domain.

    Prefer ``rfft_highpass`` + ``apply_highpass_rfft`` on the batch hot path:
    real FFTs are ~2x faster on CPU MEA batches with float32-level agreement.
    """
    return fft(_pad_or_crop_filter(hp_filter, NT))


def rfft_highpass(hp_filter, NT=30122):
    """Real-FFT of the high-pass impulse response (length NT//2+1 complex)."""
    return rfft(_pad_or_crop_filter(hp_filter, NT))


def apply_highpass_rfft(X, fwav_r, NT=None):
    """Apply a precomputed ``rfft_highpass`` filter along the last dim of X.

    Matches ``real(ifft(fft(X) * conj(fft(filter))))`` then ``fftshift`` to
    float32 noise (~1e-6 abs on unit-scale random data). Used by
    ``BinaryFiltered.filter`` for every detect/extract batch.
    """
    if NT is None:
        NT = int(X.shape[-1])
    Y = irfft(rfft(X) * torch.conj(fwav_r), n=NT)
    return fftshift(Y, dim=-1)
