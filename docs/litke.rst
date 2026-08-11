.. _litke:

Litke MEA data
==============

Kilosort can sort Litke multi-electrode array (MEA) recordings directly from
their native packed ``.bin`` files. You do not need an offline conversion to
int16 before sorting.


What Litke data looks like
--------------------------

A Litke recording is typically either:

* a **folder** of multi-part files named like ``data000000.bin``,
  ``data000001.bin``, … (for example ``/path/to/EXP/data000/``), or
* a **single** ``.bin`` file with the same format.

Each file begins with a **Vision / Litke binary header** (big-endian tags).
Sample data after the header is **packed 12-bit** values. When the electrode
count is odd, channel 0 is a 16-bit **TTL** channel packed separately from the
12-bit recording channels.

After unpacking with the default TTL handling, the array shape used for sorting
is ``(n_samples, n_channels)`` with ``dtype=int16``, where ``n_channels`` is
usually **512** or **519** to match lab probe maps.


Why use the native reader
-------------------------

Lab pipelines often convert Litke bins to a plain int16 binary with tools such
as ``bin2py`` before spike sorting. That rewrite:

* can produce multi-gigabyte intermediate files, and
* may require a Cython extension that is not always built.

``kilosort.litke.LitkeRecording`` streams samples on the fly: the header is
parsed in pure Python, and samples are unpacked with Numba (already a Kilosort
dependency). No separate convert step and no Cython build are required for
sorting.


Minimal working example
-----------------------

Pass a ``LitkeRecording`` as ``file_object`` to ``run_kilosort``. You still
provide a ``filename`` (Kilosort uses it for bookkeeping); data are read from
the recording object, not as raw int16 from that path.

.. code-block:: python

   from pathlib import Path
   from kilosort import run_kilosort
   from kilosort.litke import LitkeRecording

   # Folder of dataXXXXXX.bin parts, or a single .bin path
   data_path = Path('/path/to/EXP/data000')
   rec = LitkeRecording(data_path)  # drop_ttl=True by default

   settings = {
       'n_chan_bin': rec.n_chan,       # must match probe channel count
       'fs': int(rec.fs),              # often 20000 for Litke
       # Optional: limit duration for a smoke test (seconds)
       # 'tmax': 30,
   }

   ops, st, clu, tF, Wall, similar_templates, \
       is_ref, est_contam_rate, kept_spikes = run_kilosort(
           settings=settings,
           filename=str(rec.paths[0]),
           file_object=rec,
           probe=your_litke_probe,     # 512- or 519-channel Litke probe map
           # probe_name=...            # if you load probes by name in your setup
           results_dir=data_path / 'kilosort4',
       )
   rec.close()


Using ``BinaryRWFile`` with ``file_object``
------------------------------------------

The same object works with the lower-level IO wrapper:

.. code-block:: python

   from kilosort.io import BinaryRWFile
   from kilosort.litke import LitkeRecording

   rec = LitkeRecording('/path/to/EXP/data000')
   bfile = BinaryRWFile(
       filename=str(rec.paths[0]),
       n_chan_bin=rec.n_chan,
       fs=int(rec.fs),
       file_object=rec,
       device='cpu',   # or a CUDA device string / torch.device
   )
   # bfile.padded_batch_to_torch(batch_index) → torch tensor for one batch
   rec.close()


Required settings and probe
---------------------------

Match these to the recording and your probe layout:

+------------------+----------------------------------------------------------+
| Setting          | Value                                                    |
+==================+==========================================================+
| ``n_chan_bin``   | ``rec.n_chan`` (channels after TTL handling; typically   |
|                  | 512 or 519)                                              |
+------------------+----------------------------------------------------------+
| ``fs``           | ``int(rec.fs)``, commonly ``20000`` for Litke            |
+------------------+----------------------------------------------------------+
| Probe            | Your Litke **512** or **519** probe map; ``n_chan_bin``  |
|                  | must match the number of channels in that probe          |
+------------------+----------------------------------------------------------+
| ``drop_ttl``     | ``True`` (default on ``LitkeRecording``): drop channel 0 |
|                  | so the layout matches lab converter output and probe maps|
+------------------+----------------------------------------------------------+

Do not invent probe paths. Use the Litke 512/519 probe file you already use for
converted data so channel order and geometry stay consistent.


Smoke tests with ``tmax``
-------------------------

For a short end-to-end check without sorting a full multi-hour experiment, set
``tmax`` in seconds (for example ``30`` or ``60``) in ``settings``. That limits
how much of the recording Kilosort processes while still exercising the native
reader and the rest of the pipeline.


TTL channel
-----------

On odd electrode counts, **TTL is channel 0** in the packed layout. By default
(``drop_ttl=True``), ``LitkeRecording`` omits that channel so ``shape[1]``
matches the lab converter and standard Litke probe maps (512 or 519 recording
channels). Only set ``drop_ttl=False`` if you intentionally want the TTL
channel included and have adjusted ``n_chan_bin`` and the probe accordingly.


What not to do
--------------

* **Do not** pass a raw Litke ``.bin`` path to ``BinaryRWFile`` or
  ``run_kilosort`` as a plain int16 binary **without** ``file_object``. The
  packed 12-bit layout is not row-major int16; results will be wrong or the
  load will fail.
* **Do not** drop the TTL twice (for example convert offline with TTL removed
  and also use another custom drop). With the native reader, leave
  ``drop_ttl=True`` and use the same probe as for converter output.
* **Do not** set ``n_chan_bin`` to the wrong count (header electrode count
  including TTL, or a Neuropixels default). Always use ``rec.n_chan`` with the
  matching 512- or 519-channel Litke probe.


Correctness tests
-----------------

The unpack layout and ``file_object`` integration are covered by
``tests/test_litke.py`` (header parse, even/odd pack–unpack identity, TTL drop,
multi-file folders, and ``BinaryRWFile`` smoke). Treat that suite as the
correctness contract when changing Litke IO.
