Litke MEA binary data
=====================

This fork can stream **Litke / Vision packed ``.bin``** recordings into Kilosort
without an offline multi‑GB conversion step.

Electrode 0 is TTL (stim sync), not spikes
------------------------------------------

In every Litke bin, **electrode index 0 is the digital TTL / visual‑stimulus
trigger channel**. It is used to align stimulus frames and other lab events.
It is **not** a recording electrode for spike sorting.

| Role | Index in packed bin | Used for spike sorting? |
| ---- | ------------------- | ----------------------- |
| TTL / visual stim triggers | ``0`` | **No** (dropped by default) |
| Neural electrodes | ``1 … N-1`` | Yes → 512 or 519 channels |

Lab converters (``convert_litke_to_kilosort``, MEA‑fieldlab join scripts) write
only ``samples[:, 1:]`` to the int16 file Kilosort sorts. Native IO matches that
contract with ``drop_ttl=True`` (the default).

**Do not** pass electrode 0 into Kilosort as a neural channel. If you need the
stim stream, save it separately (see below).

Quick start
-----------

.. code-block:: python

    from pathlib import Path
    from kilosort.litke import LitkeRecording
    from kilosort import run_kilosort

    litke_path = Path('/path/to/EXP/data000')  # folder of data000000.bin, …
    # or a single: Path('/path/to/data000000.bin')

    rec = LitkeRecording(litke_path)  # drop_ttl=True → neural channels only
    print(rec.shape, rec.fs, rec.array_id, rec.n_chan)
    # e.g. (17860000, 519), 20000.0, 1551, 519

    # Optional: keep stim TTL for alignment (not used by sorting)
    rec.save_ttl(Path('results') / 'ttl_chan0.npy')
    onset_samples = rec.detect_ttl_onsets()  # lab threshold 1000
    np.save(Path('results') / 'ttl_onsets.npy', onset_samples)

    settings = {
        'n_chan_bin': rec.n_chan,   # 512 or 519 — must match probe
        'fs': int(rec.fs),          # usually 20000
        'results_dir': 'results/kilosort4_litke',
        # 'tmin': 0, 'tmax': 10,    # short smoke on CPU
        # … plus your production thresholds / dmin / probe …
    }
    run_kilosort(
        settings,
        filename=str(rec.paths[0]),  # still required for bookkeeping
        file_object=rec,
        # probe=your_litke_probe,
    )
    rec.close()

Using ``BinaryRWFile`` only
---------------------------

.. code-block:: python

    import torch
    from kilosort.litke import LitkeRecording
    from kilosort.io import BinaryRWFile

    rec = LitkeRecording('/path/to/data000')
    bfile = BinaryRWFile(
        filename=str(rec.paths[0]),
        n_chan_bin=rec.n_chan,
        fs=int(rec.fs),
        file_object=rec,
        device=torch.device('cpu'),  # or 'cuda'
        tmin=0,
        tmax=5,  # seconds — optional short window
    )
    X = bfile.padded_batch_to_torch(0)  # shape (n_chan, NT + 2*nt)

Saving TTL separately
---------------------

.. code-block:: python

    from kilosort.litke import LitkeRecording

    with LitkeRecording('/path/to/data000') as rec:
        # Full int16 waveform of electrode 0
        rec.save_ttl('ttl_chan0.npy')

        # Or a window
        ttl = rec.get_ttl(start=0, n_samples=rec.fs * 60)  # first minute

        # Rising edges (same rule as convert_litke_to_kilosort):
        # transition from < -threshold to >= -threshold, default threshold=1000
        onsets = rec.detect_ttl_onsets(threshold=1000)
        # onsets are sample indices; times in seconds: onsets / rec.fs

``get_ttl`` always returns electrode 0, even when ``drop_ttl=True``. You do
**not** need ``drop_ttl=False`` (which would put TTL into the sorting matrix).

What the files look like
------------------------

* **Folder mode:** ``data000/data000000.bin``, ``data000001.bin``, …  
  Header lives in the first file; later files are packed sample bodies only.
* **Single file:** one ``.bin`` with header + body.
* **Sample rate:** usually 20 kHz (from the Vision header).
* **Channel counts after dropping TTL:**
  * 519‑electrode (30 µm) boards → **519** neural channels (520 incl. TTL).
  * 512‑electrode (60 µm) boards → **512** neural channels (513 incl. TTL).

``n_chan_bin`` and your probe geometry must match those neural counts.

What not to do
--------------

* Do **not** open a Litke ``.bin`` as a plain int16 ``BinaryRWFile`` filename.
  The file is packed 12‑bit Vision data, not interleaved int16 samples.
* Do **not** set ``drop_ttl=False`` for sorting unless you intentionally change
  channel count and probe maps (almost never).
* Do **not** treat electrode 0 as a spike channel in Phy or analysis.

Correctness
-----------

* Unpack layout matches lab bin2py
  (``unpack_bin_even_num_electrodes`` / ``unpack_bin_odd_num_electrodes``).
* **Frozen lab-oracle fixtures** (not self‑pack roundtrips):
  ``tests/data/litke_real_519_bin2py_oracle.npz`` — real mid‑recording bytes
  from ``20251204A/data000`` with expected int16 from
  ``bin2py_cythonext``; ``tests/data/litke_odd_bin2py_oracle.npz`` for the
  odd/512 path. ``tests/test_litke.py`` asserts bit‑exact unpack against
  these. That catches self‑consistent‑but‑wrong nibble/sign bugs that
  pack∋unpack identity tests cannot see.
* Other unit tests: multi‑file, TTL drop/save/onsets, ``BinaryRWFile`` smoke.
* Live field recheck (optional): import the lab ``.so`` and compare windows;
  last run bit‑exact at start / mid / 10 M / multi‑file boundary.
* Short CPU e2e (non‑golden): ``LitkeRecording`` + ``tmax=5`` + 30 µm 519
  probe + Th 8/6 on ``20251204A/data000`` completed (~80 s): 110 units
  (14 good), ~38k spikes. Runtime dominated by peel spike detection, not IO.

API summary
-----------

* ``LitkeRecording(path, drop_ttl=True)`` — array‑like ``(n_samples, n_chan)`` int16
* ``rec.n_chan``, ``rec.fs``, ``rec.array_id``, ``rec.paths``
* ``rec.get_ttl(...)`` / ``rec.save_ttl(path)`` / ``rec.detect_ttl_onsets(...)``
* ``open_litke(path)`` — same as the constructor
