import os
import gc
import logging
import warnings
logger = logging.getLogger(__name__)

from torch.nn.functional import max_pool2d, avg_pool2d, conv1d, max_pool1d
import numpy as np
import torch
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm

from kilosort.utils import (
    get_clip_buffer_capacity,
    get_spike_buffer_capacity,
    template_path,
    log_performance,
)

# Historical sampling limit for wPCA/wTEMP clip collection. Initial capacity
# may be smaller (see get_clip_buffer_capacity); the buffer grows up to this.
CLIP_BUFFER_HARD_CAP = 500_000


def my_max2d(X, dt):
    Xmax = max_pool2d(
        X.unsqueeze(0), [2*dt[0]+1, 2*dt[1]+1],
        stride=[1,1], padding=[dt[0],dt[1]]
        )    
    return Xmax[0]

def my_sum2d(X, dt):
    Xsum = avg_pool2d(
        X.unsqueeze(0), [2*dt[0]+1, 2*dt[1]+1],
        stride=[1,1], padding=[dt[0],dt[1]]
        )    
    Xsum *= (2*dt[0]+1) * (2*dt[1]+1)
    return Xsum[0]

def extract_snippets(X, nt, twav_min, Th_single_ch, loc_range=[4,5],
                     long_range=[6,30], device=torch.device('cuda'),
                     tarange=None):
    Xabs   = X.abs()
    Xmax   = my_max2d(Xabs, loc_range)
    ispeak = torch.logical_and(Xmax==Xabs, Xabs > Th_single_ch).float()

    ispeak_sum  = my_sum2d(ispeak, long_range)
    is_peak_iso = ((ispeak_sum==1) * (ispeak==1))

    is_peak_iso[:, :nt] = 0
    is_peak_iso[:, -nt:] = 0

    xy = is_peak_iso.nonzero()

    if tarange is None:
        tarange = torch.arange(nt, device=device)
    clips = X[xy[:,:1], xy[:,1:2] - twav_min + tarange]

    return clips

def extract_wPCA_wTEMP(ops, bfile, nt=61, twav_min=20, Th_single_ch=6, nskip=25,
                       device=torch.device('cuda')):

    # Scale the initial snippet buffer with recording length so short CPU runs
    # do not reserve a full 500k×nt float32 slab up front. Grow (and, at the
    # historical hard cap, partial-fill) so a single dense batch can never leave
    # zero clips for TruncatedSVD — matching spike-buffer grow-on-overflow.
    max_clips = CLIP_BUFFER_HARD_CAP
    n_clips = get_clip_buffer_capacity(bfile.n_batches, nskip=nskip)
    clips = np.zeros((n_clips, nt), 'float32')
    # Fixed window for every batch of clip collection.
    tarange = torch.arange(nt, device=device)
    i = 0
    for j in range(0, bfile.n_batches, nskip):
        X = bfile.padded_batch_to_torch(j, ops)
        
        clips_new = extract_snippets(
            X, nt=nt, twav_min=twav_min, Th_single_ch=Th_single_ch,
            device=device, tarange=tarange,
        )

        nnew = len(clips_new)
        if nnew == 0:
            continue

        need = i + nnew
        if need > clips.shape[0] and clips.shape[0] < max_clips:
            new_cap = min(max_clips, max(clips.shape[0] * 2, need))
            extra = new_cap - clips.shape[0]
            clips = np.concatenate(
                (clips, np.zeros((extra, nt), dtype=np.float32)), 0
            )

        if i + nnew > clips.shape[0]:
            # At the historical sampling cap: keep what fits, stop.
            room = clips.shape[0] - i
            if room <= 0:
                break
            clips[i:i + room] = clips_new[:room].cpu().numpy()
            i += room
            break

        clips[i:i + nnew] = clips_new.cpu().numpy()
        i += nnew

    clips = clips[:i]
    if i == 0:
        raise RuntimeError(
            'extract_wPCA_wTEMP found no isolated peak clips; cannot fit wPCA/wTEMP'
        )
    # Zero-energy clips (rare flat segments) used to make 0/0 → NaN and poison
    # TruncatedSVD / KMeans. Drop them; if none remain, fail clearly.
    norms = (clips**2).sum(1, keepdims=True) ** .5
    good = norms[:, 0] > 0
    if not np.any(good):
        raise RuntimeError(
            'extract_wPCA_wTEMP: all isolated peak clips had zero energy'
        )
    clips = clips[good]
    clips /= norms[good]

    model = TruncatedSVD(n_components=ops['settings']['n_pcs']).fit(clips)
    wPCA = torch.from_numpy(model.components_).to(device).float()

    with warnings.catch_warnings():
        msg = 'KMeans is known to have a memory leak on Windows with MKL'
        warnings.filterwarnings("ignore", message=msg)
        # Prevents memory leak for KMeans when using MKL on Windows
        nthread = os.environ.get('OMP_NUM_THREADS')
        new_nthread = 7
        if nthread is not None:
            new_nthread = min(int(nthread), new_nthread)
        os.environ['OMP_NUM_THREADS'] = str(new_nthread)
        # Single-threaded fit: with multiple OpenMP threads the reduction order
        # (and therefore wTEMP) varies run-to-run at float32 ulp level, which
        # is enough to move a handful of threshold-straddling spikes and break
        # sort reproducibility. k is tiny, so one thread costs nothing.
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            model = KMeans(n_clusters=ops['settings']['n_templates'], n_init = 10).fit(clips)
        wTEMP = torch.from_numpy(model.cluster_centers_).to(device).float()
        wTEMP = wTEMP / (wTEMP**2).sum(1).unsqueeze(1)**.5
        if nthread is not None:
            os.environ['OMP_NUM_THREADS'] = nthread
        else:
            os.environ.pop('OMP_NUM_THREADS')

    return wPCA, wTEMP

def get_waves(ops, device=torch.device('cuda')):
    dd = np.load(template_path())
    wTEMP = torch.from_numpy(dd['wTEMP']).to(device)
    wPCA = torch.from_numpy(dd['wPCA']).to(device)
    return wPCA, wTEMP

def nearest_neighbour_pitch(xc, yc):
    """Median distance from each contact to its closest neighbour.

    Per-axis coordinate differences describe a rectangular lattice and nothing
    else. On the Litke 519 hex lattice, and on the 512 array whose rows are
    offset by half a column, the x coordinates step 30 um while no contact is
    actually within 60 um of another -- so an axis-wise estimate reports half
    the real spacing. This measures the spacing directly instead, which is the
    same number on a rectangular array and the right one on the others.

    A KD-tree rather than a full pairwise matrix: the latter is fine for a few
    hundred contacts but allocates N^2 floats, which is not something to leave
    in a library that also runs on probes with thousands of them.
    """
    p = np.column_stack([np.asarray(xc, dtype=np.float64),
                         np.asarray(yc, dtype=np.float64)])
    if len(p) < 2:
        return 1.0
    # k=2: the first neighbour of a point is itself, at distance 0.
    d, _ = cKDTree(p).query(p, k=2)
    pitch = float(np.median(d[:, 1]))
    return pitch if pitch > 0 else 1.0


def template_centers(ops):
    shank_idx = ops['kcoords']
    xc = ops['xc']
    yc = ops['yc']
    # dmin auto-derivation is stock: median spacing of the unique y coordinates.
    #
    # A 1.5x-nearest-neighbour rule was tried here and REJECTED. It reproduced
    # both hand-tuned values exactly (519 @ 30 um -> 45, 512 @ 60 um -> 90), but
    # that was two points fitting a two-parameter story, not evidence. The first
    # out-of-sample test contradicted it: on data002 -- same array and same 30 um
    # pitch as chunk1 -- dmin=90 beat 45 on every axis (recall 0.9021 -> 0.9240,
    # 52 -> 57 of 63 units, 290s -> 158s), while on chunk1 it lost (0.9907 ->
    # 0.9711). Two recordings on one array want different values, so the optimum
    # is not a function of pitch alone and must not be derived as if it were.
    dmin = ops['settings']['dmin']
    if dmin is None:
        y_uniq = np.unique(yc)
        dmin = 1 if y_uniq.size == 1 else np.median(np.diff(y_uniq))
    ops['dmin'] = dmin

    # dminx, unlike dmin, has no stock auto path at all -- it sits at a fixed
    # default chosen for one probe, and passing None to request the same
    # treatment dmin gets raises TypeError below. That is a real bug on any
    # non-default array, so None now resolves rather than crashing. Measured
    # spacing, not the 1.5x rule above: per-axis coordinate differences report
    # half the true spacing on a hex lattice or an offset grid.
    dminx = ops['settings']['dminx']
    if dminx is None:
        dminx = nearest_neighbour_pitch(xc, yc)
    ops['dminx'] = dminx

    # Iteratively determine template placement for each shank separately.
    yup = np.array([])
    xup = np.array([])
    for i in np.unique(shank_idx):
        xc_i = xc[shank_idx == i]
        yc_i = yc[shank_idx == i]
        xmin, xmax, ymin, ymax = xc_i.min(), xc_i.max(), yc_i.min(), yc_i.max()

        # BUG (measured 2026-08-18, ks4-validation): the +.00001 endpoint guard
        # is smaller than float32 resolution at these coordinates, so on a
        # float32 probe `ymax + .00001 == ymax` and np.arange drops the final
        # point. np.spacing(float32(450)) is 3.05e-5. The TOP electrode row then
        # gets no universal template position at all, while the bottom row
        # always gets one because it is the arange START -- and the nearest
        # template to a top-row cell ends up dmin/2 away (45 um at dmin 90).
        #
        #   float64, dmin 90 -> 21 positions, max 450.0  (top row covered)
        #   float32, dmin 90 -> 20 positions, max 405.0  (top row bare)
        #
        # This destroys cells on that row, worse as dmin grows. A six-arm dmin
        # sweep on ratW10 lost 7 cells and ALL 7 sat on y=+450: 4 of the 8 cells
        # there at dmin 90, 1 at dmin 75, none at dmin 45 or 60. Changing dminx
        # does not help and neither does the clustering grid, both of which are
        # downstream of this.
        #
        # The xup line below already does the robust thing -- round the count
        # and use linspace, so this just makes y match x.
        #
        # DEFAULT ON since 2026-08-18, after a three-window ledger
        # (scripts/confirm_yupfix.sh -> logs/yupconf_{mqW5,ratW5,ratW10}.json)
        # scored it at dmin 90 against a like-for-like fixed dmin 45 control on
        # two animals and two preps. Fewer missed GT spikes on 3/3 windows,
        # zero cells lost on 3/3, and total splits 17 -> 8:
        #
        #   window   missed d45y -> d90y   lost   split    clean
        #   mqW5      0.0085 -> 0.0073     0->0   1->1     84->84
        #   ratW5     0.0073 -> 0.0062     0->0   8->3     63->63
        #   ratW10    0.0056 -> 0.0048     0->0   8->4     69->68
        #
        # Every cell the unfixed dmin 90 lost sat on y=+450, the top row, and
        # every one came back.
        #
        # DEFAULT REVERTED TO OFF the same day, on a pre-registered criterion.
        # All three confirmation windows are Litke 512 arrays, y -450..450, span
        # 900 -- which divides evenly by dmin/2 at both 45 and 90, so there the
        # fix only APPENDS the dropped endpoint and disturbs nothing else. The
        # hybrid bench (20251204A, 519 channels, y -390..390, span 780) does not
        # divide evenly: at dmin 90 the stock grid ends at 375 and the fix
        # instead spreads 18 points at 45.88 um, RE-PITCHING every template
        # position on the array. Injected-unit recall under 200 ADC then went
        # 0.627 -> 0.600 at dmin 90 -- unit 16 at 114 ADC, sitting at y -165..-75
        # (deep interior, not an edge cell), lost real coverage 0.987 -> 0.729
        # summed over every contributing cluster.
        #
        # It is not a one-way cost: on the SAME array at dmin 45 the fix helped
        # in both bands (under 200 ADC 0.742 -> 0.756, 200+ 0.961 -> 1.000),
        # rescuing unit 7 (0.646 -> 0.999) and unit 19 (0.741 -> 0.922). Quiet
        # units near threshold are simply unstable to any grid re-pitch. But
        # that is the point: on a span that does not divide evenly this is not
        # purely an edge fix, and a blanket default across untested geometries
        # is not earned by three windows of one geometry.
        #
        # Two remedies, because they are NOT the same thing on every probe:
        #
        #   KS4_YUP_FIX=1         append-only. Keep the stock arange spacing
        #                         everywhere and add the dropped endpoint. On a
        #                         span that divides evenly by dmin/2 this is
        #                         IDENTICAL to linspace (Litke 512 dmin 90:
        #                         both give 21 points at 45 um, -450..450). On
        #                         a span that does not, it leaves every interior
        #                         position exactly where stock put it and only
        #                         closes the top gap, so it cannot re-pitch.
        #   KS4_YUP_FIX=linspace  the re-pitching variant described above.
        #                         Kept only to reproduce the measurements.
        #
        # Measure before enabling either on an untested geometry.
        _fix = os.environ.get('KS4_YUP_FIX', '')
        if _fix == 'linspace':
            ny = np.round((ymax - ymin) / (dmin/2)) + 1
            yup = np.concatenate([yup, np.linspace(ymin, ymax, int(ny))])
        elif _fix not in ('', '0'):
            ygrid = np.arange(ymin, ymax + .00001, dmin/2)
            if ygrid.size == 0 or ymax - ygrid[-1] > 1e-3:
                ygrid = np.append(ygrid, ymax)
            yup = np.concatenate([yup, ygrid])
        else:
            yup = np.concatenate([yup, np.arange(ymin, ymax+.00001, dmin/2)])
        nx = np.round((xmax - xmin) / (dminx/2)) + 1
        xup = np.concatenate([xup, np.linspace(xmin, xmax, int(nx))])

    ops['yup'] = np.unique(yup)
    ops['xup'] = np.unique(xup)

    return ops


def _template_match_body(Bsl, weigh, iC, iC2_flat, nC2, Nfilt):
    """One time-chunk of the template_match loop: gather, project, reduce.

    Factored out only so torch.compile can fuse it. The statements are the
    stock ones in the stock order -- see the notes in template_match for why
    the max/gather pair must not be rewritten.
    """
    A = torch.einsum('ijk, jklm-> iklm', weigh, Bsl[iC])
    A = A.transpose(1,2)
    A = A.reshape(-1, Nfilt, Bsl.shape[-1])
    Aa, imax = torch.max(A.abs(), 0)
    sgn = torch.gather(A, 0, imax.unsqueeze(0)).squeeze(0).sign()
    imax = (1+imax) * sgn
    Amax = torch.max(Aa.index_select(0, iC2_flat).view(nC2, Nfilt, -1), 0)[0]
    return Aa, imax, Amax


# The loop above is memory-bound, but it moves its intermediates at only
# ~23 GB/s -- far under this card's peak. That is not a bandwidth limit, it is
# per-op HBM round-tripping, which Inductor fusion removes: ~1.25x on
# template_match, bit-identical to eager on an RTX A2000 (verified against the
# full 10.2 GB benchmark, see HANDOFF.md).
#
# Bit-identity is a property of the generated kernels, so it is NOT guaranteed
# on a different GPU or torch build. Set KILOSORT_NO_COMPILE=1 to force the
# eager path; validate counts before trusting the compiled path on new hardware.
_TM_BODY = None

def _template_match_body_dispatch(*args):
    """Compiled loop body, falling back to eager if Inductor is unusable.

    CPU / no-CUDA fieldlab builds stay on the eager path: torch.compile's
    Inductor compile cost and lack of win on CPU dominate short MEA runs and
    can fail hard. Set KILOSORT_FORCE_COMPILE=1 to opt in on CPU; set
    KILOSORT_NO_COMPILE=1 to force eager on GPU.
    """
    global _TM_BODY
    if _TM_BODY is None:
        force_compile = os.environ.get('KILOSORT_FORCE_COMPILE')
        no_compile = os.environ.get('KILOSORT_NO_COMPILE')
        use_compile = (
            not no_compile
            and (force_compile or torch.cuda.is_available())
        )
        if not use_compile:
            _TM_BODY = _template_match_body
        else:
            try:
                # coordinate_descent_tuning tiles these reductions harder. Set
                # once, here, and NOT per call: wrapping every call in
                # _icfg.patch() costs more than the tuning wins (40 calls per
                # batch made the compiled path slower than eager).
                import torch._inductor.config as _icfg
                _icfg.coordinate_descent_tuning = True
                # dynamic=False: the ragged last chunk gets its own graph rather
                # than forcing a slower dynamic-shape kernel for every chunk.
                _TM_BODY = torch.compile(_template_match_body, dynamic=False)
            except Exception as e:
                logger.info(f'torch.compile unavailable, using eager: {e}')
                _TM_BODY = _template_match_body
    try:
        return _TM_BODY(*args)
    except Exception as e:
        if _TM_BODY is _template_match_body:
            raise
        logger.warning(f'torch.compile failed, falling back to eager: {e}')
        _TM_BODY = _template_match_body
        return _TM_BODY(*args)


def template_match(X, ops, iC, iC2, weigh, device=torch.device('cuda'),
                   scratch=None):
    nt = ops['nt']
    nt0 = ops['settings']['nt0min']
    nk = ops['settings']['n_templates']
    NT = X.shape[-1]
    Nfilt = iC.shape[1]
    niter = 40
    nb = (NT-1)//niter+1

    # Cache unsqueezed templates across batches (wTEMP is fixed for the pass).
    if scratch is not None and scratch.get('W') is not None:
        W = scratch['W']
    else:
        W = ops['wTEMP'].unsqueeze(1)
        if scratch is not None:
            scratch['W'] = W
    B = conv1d(X.unsqueeze(1), W, padding=nt//2)
    # Reuse (Nfilt, NT) peak buffers across batches when sizes match — saves
    # ~3×Nfilt×NT alloc on every batch of universal detect (dominant stage).
    # imaxs is signed template index encoding; int32 covers n_templates and
    # peak-pool indices on MEA scales and halves the int64 slab (~0.6 GiB at
    # 2600×60k vs float32 As/Amaxs peers).
    #
    # Do NOT zero_() on reuse: the niter loop writes every column of As /
    # Amaxs / imaxs exactly once (chunks tile [0, NT)). zero_ was pure
    # bandwidth on the hottest path (~3×Nfilt×NT writes per batch).
    if (scratch is not None
            and scratch.get('As') is not None
            and scratch['As'].shape == (Nfilt, NT)
            and scratch['As'].device == device
            and scratch['imaxs'].dtype == torch.int32):
        As = scratch['As']
        Amaxs = scratch['Amaxs']
        imaxs = scratch['imaxs']
    else:
        As    = torch.empty((Nfilt, NT), device=device)
        Amaxs = torch.empty((Nfilt, NT), device=device)
        imaxs = torch.empty((Nfilt, NT), dtype=torch.int32, device=device)
        if scratch is not None:
            scratch['As'] = As
            scratch['Amaxs'] = Amaxs
            scratch['imaxs'] = imaxs
    # iC2 is (nC2, Nfilt); flattening it lets the neighbour max below use
    # index_select, which reaches the same elements on a faster path than
    # advanced indexing. Cache flat view on scratch when iC2 is stable.
    if (scratch is not None
            and scratch.get('iC2_flat') is not None
            and scratch.get('iC2_id') is id(iC2)):
        iC2_flat = scratch['iC2_flat']
        nC2 = scratch['nC2']
    else:
        iC2_flat = iC2.reshape(-1)
        nC2 = iC2.shape[0]
        if scratch is not None:
            scratch['iC2_flat'] = iC2_flat
            scratch['nC2'] = nC2
            scratch['iC2_id'] = id(iC2)

    # NOTE on the body: do not replace its max/gather pair with a max/min pair.
    # That rewrite is 1.46x faster on that statement but resolves exact
    # positive/negative magnitude ties (dense at the zero-padded batch edges)
    # toward the positive branch, where torch.max resolves toward whichever
    # index it reaches first. It passed a 12-batch bit-identity self-test and a
    # 60 s end-to-end run, then changed the full-file result: +567 spikes,
    # -25 good units, -3.2% clean yield. The gather is bit-identical by
    # construction: it reads the same one element per output position as the
    # stock three-way advanced index.
    #
    # Storing Aa/imax/Amax from inside the compiled region was tried and is
    # slower (1.16x vs 1.25x) -- the copy_ into strided views costs more than
    # the round-trip it saves.
    for t in range(niter):
        lo, hi = nb*t, min(nb*(t+1), NT)
        Aa, imax, Amax = _template_match_body_dispatch(
            B[:, :, lo:hi], weigh, iC, iC2_flat, nC2, Nfilt)
        As[:, lo:hi] = Aa
        imaxs[:, lo:hi] = imax.to(torch.int32)
        Amaxs[:, lo:hi] = Amax

    Amaxs[:,:nt] = 0
    Amaxs[:,-nt:] = 0
    Amaxs  = max_pool1d(Amaxs.unsqueeze(0), (2*nt0+1), stride = 1, padding = nt0).squeeze(0)
    xy = torch.logical_and(Amaxs==As, As > ops['Th_universal']).nonzero()
    imax = imaxs[xy[:,0], xy[:,1]]
    amp = As[xy[:,0], xy[:,1]]

    ssign = imax.sign()
    imax = imax.abs()-1
    adist = B[iC[:, xy[:,0]], imax%nk, xy[:,1]] * ssign

    #adist = B[iC[:, xy[:,0]], imax%nk, xy[:,1]] 
    
    #xy[:,1] -= nt
    return xy, imax, amp, adist


def nearest_chans(ys, yc, xs, xc, nC, device=torch.device('cuda')):
    ds = (ys - yc[:,np.newaxis])**2 + (xs - xc[:,np.newaxis])**2
    n_src = ds.shape[0]
    nC = int(min(nC, n_src))
    # Full column argsort is O(n_src log n_src) per template; we only need the
    # nC nearest. argpartition + sort of the shortlist is identical when
    # distances are unique and matches argsort[:nC] order for ties that
    # partition stably enough for MEA geometry (validated by unit test).
    if nC >= n_src:
        iC = np.argsort(ds, 0)
    else:
        part = np.argpartition(ds, nC - 1, axis=0)[:nC]
        ds_part = np.take_along_axis(ds, part, axis=0)
        order = np.argsort(ds_part, axis=0)
        iC = np.take_along_axis(part, order, axis=0)
    ds = np.take_along_axis(ds, iC, axis=0)
    iC = torch.from_numpy(iC).to(device)

    return iC, ds


def yweighted(yc, iC, adist, xy, device=torch.device('cuda'), yc_t=None):
    # yc_t: optional pre-moved contact y-coords (detect reuses across batches).
    if yc_t is None:
        yy = torch.as_tensor(yc, device=device)[iC]
    else:
        yy = yc_t[iC]
    cF0 = torch.nn.functional.relu(adist)
    # clamp avoids 0/0 → NaN when a template column has no positive weight;
    # bit-identical whenever sum(0) >= 1e-12 (normal case).
    cF0 = cF0 / cF0.sum(0).clamp_min(1e-12)

    yct = (cF0 * yy[:,xy[:,0]]).sum(0)
    return yct

def run(ops, bfile, device=torch.device('cuda'), progress_bar=None,
        clear_cache=False, verbose=False):        
    sig = ops['settings']['min_template_size']
    nsizes = ops['settings']['template_sizes']
    nb = ops['Nbatches']

    if ops['settings']['templates_from_data']:
        logger.info('Re-computing universal templates from data.')
        # Determine templates and PC features from data.
        ops['wPCA'], ops['wTEMP'] = extract_wPCA_wTEMP(
            ops, bfile, nt=ops['nt'], twav_min=ops['nt0min'], 
            Th_single_ch=ops['settings']['Th_single_ch'], nskip=25,
            device=device
            )
    else:
        logger.info('Using built-in universal templates.')
        # Use pre-computed templates.
        ops['wPCA'], ops['wTEMP'] = get_waves(ops, device=device)

    ops = template_centers(ops)
    [ys, xs] = np.meshgrid(ops['yup'], ops['xup'])
    ys, xs = ys.flatten(), xs.flatten()
    logger.info(f'Number of universal templates: {ys.size}')
    xc, yc = ops['xc'], ops['yc']

    nC = ops['settings']['nearest_chans']
    nC2 = ops['settings']['nearest_templates']
    iC, ds = nearest_chans(ys, yc, xs, xc, nC, device=device)

    # Don't use templates that are too far away from nearest channel
    # (use square of max distance since ds are squared distances)
    igood = ds[0,:] <= ops['max_channel_distance']**2
    iC = iC[:,igood]
    ds = ds[:,igood]
    ys = ys[igood]
    xs = xs[igood]
    ops['ycup'], ops['xcup'] = ys, xs

    iC2, _ = nearest_chans(ys, ys, xs, xs, nC2, device=device)

    ds_torch = torch.from_numpy(ds).to(device).float()
    template_sizes = sig * (1+torch.arange(nsizes, device=device))
    weigh = torch.exp(-ds_torch.unsqueeze(-1) / template_sizes**2)
    weigh = torch.permute(weigh, (2, 0, 1)).contiguous()
    weigh = weigh / (weigh**2).sum(1).unsqueeze(1)**.5

    spike_capacity = get_spike_buffer_capacity(bfile.n_batches)
    st = np.zeros((spike_capacity, 6), 'float64')
    tF = np.zeros((spike_capacity, nC, ops['settings']['n_pcs']), 'float32')

    k = 0
    nt = ops['nt']
    tarange = torch.arange(-(nt//2),nt//2+1, device = device)
    # Contact y-coords once for yweighted (was torch.from_numpy every batch).
    yc_t = torch.as_tensor(yc, device=device)
    # wPCA.T once (fixed for the detect pass) — avoid re-transpose per batch.
    wPCA_T = ops['wPCA'].T.contiguous()
    # Scratch peak buffers reused by template_match across batches
    tm_scratch = {}
    logger.info('Detecting spikes...')
    prog = tqdm(np.arange(bfile.n_batches), miniters=200 if progress_bar else None, 
                mininterval=60 if progress_bar else None)
    # repeat performance log after every 10 minutes of data
    log_skip = int(600 / (ops['batch_size'] / ops['fs']))
    # Prefetch: a worker thread reads batch i+1 from disk while batch i runs
    # on the GPU. Yields exactly what padded_batch_to_torch(i, ops) returns.
    batches = bfile.iter_batches(ops)
    ibatch = -1
    try:
        for ibatch in prog:
            if ibatch % log_skip == 0:
                log_performance(logger, 'debug', f'Batch {ibatch} of {nb-1} ({100*(ibatch/nb):.1f}%)')

            X = next(batches)
            xy, imax, amp, adist = template_match(
                X, ops, iC, iC2, weigh, device=device, scratch=tm_scratch
            )
            nsp = len(xy)
            if nsp == 0:
                if clear_cache:
                    gc.collect()
                    if device.type == 'cuda' and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                if progress_bar is not None:
                    progress_bar.emit(int((ibatch+1) / bfile.n_batches * 100))
                continue

            yct = yweighted(yc, iC, adist, xy, device=device, yc_t=yc_t)

            if k+nsp>st.shape[0]:
                new_cap = max(k + nsp, st.shape[0] * 2)
                st2 = np.zeros((new_cap, st.shape[1]), dtype=st.dtype)
                st2[:k] = st[:k]
                st = st2
                tF2 = np.zeros((new_cap,) + tF.shape[1:], dtype=tF.dtype)
                tF2[:k] = tF[:k]
                tF = tF2

            xsub = X[iC[:,xy[:,:1]], xy[:,1:2] + tarange]
            xfeat = xsub @ wPCA_T
            tF[k:k+nsp] = xfeat.transpose(0,1).cpu().numpy()

            t_shift = ibatch * bfile.batch_downsampling * (ops['batch_size']/ops['fs'])
            # Build all six columns on-device and move them in one transfer.
            # The sample->seconds conversion happens on the host afterwards:
            # GPU float64 division can differ from numpy's by 1 ulp, and spike
            # times must stay bit-identical to the original per-column code.
            col1 = xy[:,1].double()
            cols = torch.stack(
                (col1, yct.double(), amp.double(), imax.double(),
                 torch.full_like(col1, ibatch), xy[:,0].double()), dim=1)
            cols = cols.cpu().numpy()
            cols[:,0] = (cols[:,0] - nt)/ops['fs'] + t_shift
            st[k:k+nsp] = cols

            k = k + nsp
            if clear_cache:
                gc.collect()
                if device.type == 'cuda' and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if progress_bar is not None:
                progress_bar.emit(int((ibatch+1) / bfile.n_batches * 100))
    except:
        logger.exception(f'Error in spikedetect.run on batch {ibatch}')
        try:
            logger.debug(f'X shape: {X.shape}')
            logger.debug(f'xy shape: {xy.shape}')
        except UnboundLocalError:
            # Error happened before one or both of these was assigned,
            # no need to raise an additional error for this.
            pass
        raise
            
    if ibatch >= 0 and nb > 0:
        log_performance(logger, 'debug', f'Batch {ibatch} of {nb-1} ({100*(ibatch/nb):.1f}%)')

    st = st[:k]
    tF = tF[:k]
    ops['iC'] = iC
    ops['iC2'] = iC2
    ops['weigh'] = weigh
    return st, tF, ops
