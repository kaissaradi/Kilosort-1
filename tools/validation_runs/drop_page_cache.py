#!/usr/bin/env python
"""Reclaim page cache for this run's big files without root.

The background-task monitor kills on low MemFree, but MemFree was low only
because streaming a 34 GB .bin (and the 25.5 GB raw it was built from) filled
the page cache with fully reclaimable pages -- MemAvailable never dropped
below 190 of 197 GB and kilosort's own peak was 9.3 GB.

posix_fadvise(POSIX_FADV_DONTNEED) drops a file's clean pages from cache and
needs no privileges, unlike /proc/sys/vm/drop_caches.
"""
import glob
import os
import sys

TARGETS = []
for pat in sys.argv[1:]:
    TARGETS.extend(sorted(glob.glob(pat)))


def meminfo():
    d = {}
    with open('/proc/meminfo') as f:
        for line in f:
            k, _, v = line.partition(':')
            d[k] = int(v.split()[0]) // 1024 // 1024  # GB
    return d


before = meminfo()
dropped = 0
for path in TARGETS:
    if not os.path.isfile(path):
        continue
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        continue
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        dropped += os.fstat(fd).st_size
    finally:
        os.close(fd)

after = meminfo()
print(f'fadvised {len(TARGETS)} paths, {dropped / 1e9:.1f} GB of file data')
print(f'  MemFree      {before["MemFree"]:4d} -> {after["MemFree"]:4d} GB')
print(f'  Cached       {before["Cached"]:4d} -> {after["Cached"]:4d} GB')
print(f'  MemAvailable {before["MemAvailable"]:4d} -> {after["MemAvailable"]:4d} GB')
