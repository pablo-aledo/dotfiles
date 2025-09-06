#!/usr/bin/env python3
import os
import errno
import shutil
import logging
from fuse import FUSE, Operations

class CacheFS(Operations):
    def __init__(self, source, cache):
        self.source = os.path.abspath(source)
        self.cache = os.path.abspath(cache)
        logging.info(f"Initialized with source={self.source}, cache={self.cache}")

    def _source_path(self, path):
        return os.path.join(self.source, path.lstrip("/"))

    def _cache_path(self, path):
        return os.path.join(self.cache, path.lstrip("/"))

    def _ensure_cached(self, path):
        src = self._source_path(path)
        dst = self._cache_path(path)

        if not os.path.exists(dst):
            if not os.path.exists(src):
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            logging.info(f"Cached: {src} -> {dst}")

    def getattr(self, path, fh=None):
        full_path = self._cache_path(path)
        if os.path.exists(full_path):
            st = os.lstat(full_path)
        else:
            self._ensure_cached(path)
            st = os.lstat(self._cache_path(path))
        return dict((key, getattr(st, key)) for key in (
            'st_atime', 'st_ctime', 'st_gid', 'st_mode',
            'st_mtime', 'st_nlink', 'st_size', 'st_uid'))

    def readdir(self, path, fh):
        src_dir = self._source_path(path)
        cache_dir = self._cache_path(path)
        entries = set(['.', '..'])

        if os.path.isdir(src_dir):
            entries.update(os.listdir(src_dir))
        if os.path.isdir(cache_dir):
            entries.update(os.listdir(cache_dir))

        return list(entries)

    def readlink(self, path):
        return os.readlink(self._cache_path(path))

    def open(self, path, flags):
        try:
            self._ensure_cached(path)
        except FileNotFoundError:
            # Allow creation if opening for writing
            if not (flags & os.O_WRONLY or flags & os.O_RDWR or flags & os.O_CREAT):
                raise
        return os.open(self._cache_path(path), flags)

    def create(self, path, mode, fi=None):
        full_path = self._cache_path(path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        return os.open(full_path, os.O_WRONLY | os.O_CREAT, mode)

    def read(self, path, size, offset, fh):
        os.lseek(fh, offset, os.SEEK_SET)
        return os.read(fh, size)

    def write(self, path, data, offset, fh):
        os.lseek(fh, offset, os.SEEK_SET)
        return os.write(fh, data)

    def truncate(self, path, length):
        full_path = self._cache_path(path)
        with open(full_path, 'r+') as f:
            f.truncate(length)

    def flush(self, path, fh):
        return os.fsync(fh)

    def release(self, path, fh):
        return os.close(fh)

    def unlink(self, path):
        try:
            os.unlink(self._cache_path(path))
            logging.info(f"Deleted: {path}")
        except FileNotFoundError:
            raise

    def rename(self, old, new):
        os.makedirs(os.path.dirname(self._cache_path(new)), exist_ok=True)
        os.rename(self._cache_path(old), self._cache_path(new))
        logging.info(f"Renamed: {old} -> {new}")

    def statfs(self, path):
        stv = os.statvfs(self._cache_path(path))
        return dict((key, getattr(stv, key)) for key in (
            'f_bavail', 'f_bfree', 'f_blocks', 'f_bsize',
            'f_favail', 'f_ffree', 'f_files', 'f_flag', 'f_frsize', 'f_namemax'))

if __name__ == '__main__':
    import sys
    import argparse

    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

    parser = argparse.ArgumentParser(description='FUSE CacheFS - local cache layer for slow network filesystems')
    parser.add_argument('source', help='Path to source (slow) filesystem, e.g. /mnt/redfs')
    parser.add_argument('cache', help='Path to local cache directory, e.g. ~/.cache/redfs')
    parser.add_argument('mountpoint', help='Path where the virtual filesystem will be mounted, e.g. ~/mnt/cachefs')
    args = parser.parse_args()

    os.makedirs(args.mountpoint, exist_ok=True)
    os.makedirs(args.cache, exist_ok=True)

    FUSE(CacheFS(args.source, args.cache), args.mountpoint, nothreads=True, foreground=True)

