import os
import gc
import logging
import warnings
logger = logging.getLogger(__name__)

from torch.nn.functional import max_pool2d, avg_pool2d, conv1d, max_pool1d
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm

from kilosort.utils import template_path, log_performance


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
                     long_range=[6,30], device=torch.device('cuda')):
    Xabs   = X.abs()
    Xmax   = my_max2d(Xabs, loc_range)
    ispeak = torch.logical_and(Xmax==Xabs, Xabs > Th_single_ch).float()

    ispeak_sum  = my_sum2d(ispeak, long_range)
    is_peak_iso = ((ispeak_sum==1) * (ispeak==1))

    is_peak_iso[:, :nt] = 0
    is_peak_iso[:, -nt:] = 0

    xy = is_peak_iso.nonzero()

    clips = X[xy[:,:1], xy[:,1:2] - twav_min + torch.arange(nt, device=device)]

    return clips

def extract_wPCA_wTEMP(ops, bfile, nt=61, twav_min=20, Th_single_ch=6, nskip=25,
                       device=torch.device('cuda')):

    clips = np.zeros((500000,nt), 'float32')
    i = 0
    for j in range(0, bfile.n_batches, nskip):
        X = bfile.padded_batch_to_torch(j, ops)
        
        clips_new = extract_snippets(X, nt=nt, twav_min=twav_min,
                                     Th_single_ch=Th_single_ch, device=device)

        nnew = len(clips_new)

        if i+nnew>clips.shape[0]:
            break

        clips[i:i+nnew] = clips_new.cpu().numpy()
        i+= nnew 

    clips = clips[:i]
    clips /= (clips**2).sum(1, keepdims=True)**.5

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

def template_centers(ops):
    shank_idx = ops['kcoords']
    xc = ops['xc']
    yc = ops['yc']
    dmin = ops['settings']['dmin']
    if dmin is None:
        # Try to determine a good value automatically based on contact positions.
        y_uniq = np.unique(yc)
        if y_uniq.size == 1:
            dmin = 1
        else:
            dmin = np.median(np.diff(np.unique(y_uniq)))
    ops['dmin'] = dmin
    ops['dminx'] = dminx = ops['settings']['dminx']

    # Iteratively determine template placement for each shank separately.
    yup = np.array([])
    xup = np.array([])
    for i in np.unique(shank_idx):
        xc_i = xc[shank_idx == i]
        yc_i = yc[shank_idx == i]
        xmin, xmax, ymin, ymax = xc_i.min(), xc_i.max(), yc_i.min(), yc_i.max()

        yup = np.concatenate([yup, np.arange(ymin, ymax+.00001, dmin/2)])
        nx = np.round((xmax - xmin) / (dminx/2)) + 1
        xup = np.concatenate([xup, np.linspace(xmin, xmax, int(nx))])

    ops['yup'] = np.unique(yup)
    ops['xup'] = np.unique(xup)

    return ops


def template_match(X, ops, iC, iC2, weigh, device=torch.device('cuda')):
    nt = ops['nt']
    nt0 = ops['settings']['nt0min']
    nk = ops['settings']['n_templates']
    NT = X.shape[-1]
    Nfilt = iC.shape[1]
    niter = 40
    nb = (NT-1)//niter+1

    W = ops['wTEMP'].unsqueeze(1)
    B = conv1d(X.unsqueeze(1), W, padding=nt//2)
    As    = torch.zeros((Nfilt, NT), device=device)
    Amaxs = torch.zeros((Nfilt, NT), device=device)
    imaxs = torch.zeros((Nfilt, NT), dtype = torch.int64, device=device)
    # iC2 is (nC2, Nfilt); flattening it lets the neighbour max below use
    # index_select, which reaches the same elements on a faster path than
    # advanced indexing.
    iC2_flat = iC2.reshape(-1)
    nC2 = iC2.shape[0]

    for t in range(niter):
        A = torch.einsum('ijk, jklm-> iklm', weigh, B[iC,:, nb*t:nb*(t+1)])
        A = A.transpose(1,2)
        A = A.reshape(-1, Nfilt, A.shape[-1])
        w = A.shape[-1]

        # NOTE: do not replace this with a max/min pair. That rewrite is 1.46x
        # faster on this statement but resolves exact positive/negative
        # magnitude ties (dense at the zero-padded batch edges) toward the
        # positive branch, where torch.max resolves toward whichever index it
        # reaches first. It passed a 12-batch bit-identity self-test and a 60 s
        # end-to-end run, then changed the full-file result: +567 spikes,
        # -25 good units, -3.2% clean yield.
        Aa, imax = torch.max(A.abs(), 0)
        # gather reads the same one element per output position as the stock
        # three-way advanced index, so this is bit-identical by construction.
        sgn = torch.gather(A, 0, imax.unsqueeze(0)).squeeze(0).sign()
        imax = (1+imax) * sgn

        As[:, nb*t:nb*(t+1)] = Aa
        imaxs[:, nb*t:nb*(t+1)] = imax
        Amax = torch.max(Aa.index_select(0, iC2_flat).view(nC2, Nfilt, w), 0)[0]
        Amaxs[:, nb*t:nb*(t+1)] = Amax

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
    iC = np.argsort(ds, 0)[:nC]
    iC = torch.from_numpy(iC).to(device)
    ds = np.sort(ds, 0)[:nC]

    return iC, ds


def yweighted(yc, iC, adist, xy, device=torch.device('cuda')):    

    yy = torch.from_numpy(yc).to(device)[iC]
    cF0 = torch.nn.functional.relu(adist)
    cF0 = cF0/cF0.sum(0)

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

    st = np.zeros((10**6, 6), 'float64')
    tF = np.zeros((10**6, nC , ops['settings']['n_pcs']), 'float32')

    k = 0
    nt = ops['nt']
    tarange = torch.arange(-(nt//2),nt//2+1, device = device)
    logger.info('Detecting spikes...')
    prog = tqdm(np.arange(bfile.n_batches), miniters=200 if progress_bar else None, 
                mininterval=60 if progress_bar else None)
    # repeat performance log after every 10 minutes of data
    log_skip = int(600 / (ops['batch_size'] / ops['fs']))
    # Prefetch: a worker thread reads batch i+1 from disk while batch i runs
    # on the GPU. Yields exactly what padded_batch_to_torch(i, ops) returns.
    batches = bfile.iter_batches(ops)
    try:
        for ibatch in prog:
            if ibatch % log_skip == 0:
                log_performance(logger, 'debug', f'Batch {ibatch} of {nb-1} ({100*(ibatch/nb):.1f}%)')

            X = next(batches)
            xy, imax, amp, adist = template_match(X, ops, iC, iC2, weigh, device=device)
            yct = yweighted(yc, iC, adist, xy, device=device)
            nsp = len(xy)

            if k+nsp>st.shape[0]:
                st = np.concatenate((st, np.zeros_like(st)), 0)
                tF = np.concatenate((tF, np.zeros_like(tF)), 0)

            xsub = X[iC[:,xy[:,:1]], xy[:,1:2] + tarange]
            xfeat = xsub @ ops['wPCA'].T
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
            
    log_performance(logger, 'debug', f'Batch {ibatch} of {nb-1} ({100*(ibatch/nb):.1f}%)')

    st = st[:k]
    tF = tF[:k]
    ops['iC'] = iC
    ops['iC2'] = iC2
    ops['weigh'] = weigh
    return st, tF, ops
