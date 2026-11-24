"""Extract an Info-ZIP split archive (name.z01, name.z02, ..., name.zip) with the standard library.

Python's zipfile refuses multi-disk archives. Here the parts are exposed as one concatenated
read-only stream; the multi-disk check is bypassed, and each entry's local-header offset is
rebased to the start of the disk (part) it lives on.

  py -3 -I unzip_split.py D:/.../TestImage.zip --out D:/.../test [--check 50]
"""
import argparse
import bisect
import glob
import io
import os
import struct
import zipfile


class Concat(io.RawIOBase):
    def __init__(self, paths):
        self.fs = [open(p, "rb") for p in paths]
        self.sizes = [os.path.getsize(p) for p in paths]
        self.starts = [sum(self.sizes[:i]) for i in range(len(paths))]
        self.total, self.pos = sum(self.sizes), 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.total + off
        return self.pos

    def readinto(self, b):
        n, done = len(b), 0
        while done < n and self.pos < self.total:
            i = bisect.bisect_right(self.starts, self.pos) - 1
            f = self.fs[i]
            f.seek(self.pos - self.starts[i])
            chunk = f.read(min(n - done, self.sizes[i] - (self.pos - self.starts[i])))
            if not chunk:
                break
            b[done:done + len(chunk)] = chunk
            done += len(chunk)
            self.pos += len(chunk)
        return done


def _end_rec_64_any_disk(fpin, offset, endrec):
    """zipfile._EndRecData64 without the 'spans multiple disks' refusal."""
    try:
        fpin.seek(offset - zipfile.sizeEndCentDir64Locator, 2)
    except OSError:
        return endrec
    data = fpin.read(zipfile.sizeEndCentDir64Locator)
    if len(data) != zipfile.sizeEndCentDir64Locator:
        return endrec
    sig, diskno, reloff, disks = struct.unpack(zipfile.structEndArchive64Locator, data)
    if sig != zipfile.stringEndArchive64Locator:
        return endrec
    fpin.seek(offset - zipfile.sizeEndCentDir64Locator - zipfile.sizeEndCentDir64, 2)
    data = fpin.read(zipfile.sizeEndCentDir64)
    (sig, sz, create_version, read_version, disk_num, disk_dir,
     dircount, dircount2, dirsize, diroffset) = struct.unpack(zipfile.structEndArchive64, data)
    if sig != zipfile.stringEndArchive64:
        return endrec
    endrec[zipfile._ECD_SIGNATURE] = sig
    endrec[zipfile._ECD_DISK_NUMBER] = disk_num
    endrec[zipfile._ECD_DISK_START] = disk_dir
    endrec[zipfile._ECD_ENTRIES_THIS_DISK] = dircount
    endrec[zipfile._ECD_ENTRIES_TOTAL] = dircount2
    endrec[zipfile._ECD_SIZE] = dirsize
    endrec[zipfile._ECD_OFFSET] = diroffset
    return endrec


def open_split(last_part):
    stem = os.path.splitext(last_part)[0]
    parts = sorted(glob.glob(stem + ".z[0-9][0-9]")) + [last_part]
    zipfile._EndRecData64 = _end_rec_64_any_disk
    stream = Concat(parts)
    z = zipfile.ZipFile(io.BufferedReader(stream, buffer_size=1 << 20))
    # zipfile shifted every offset by 'concat' (= start of the disk holding the central directory);
    # rebase each entry to the start of its own disk instead.
    cd_disk = len(parts) - 1
    shift = stream.starts[cd_disk]
    for zi in z.infolist():
        zi.header_offset = zi.header_offset - shift + stream.starts[zi.volume]
    return z, parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("archive", help="the final .zip part")
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", type=int, default=0, help="only CRC-check the first N and last N entries")
    a = ap.parse_args()
    z, parts = open_split(a.archive)
    names = z.namelist()
    bad = [n for n in names if n.startswith("/") or ".." in n.split("/")]
    if bad:
        raise SystemExit(f"unsafe paths: {bad[:3]}")
    print(f"parts={[os.path.basename(p) for p in parts]} entries={len(names)} "
          f"volumes={sorted({zi.volume for zi in z.infolist()})}")
    if a.check:
        sample = [zi for zi in z.infolist() if not zi.is_dir()]
        sample = sample[: a.check] + sample[-a.check:]
        for zi in sample:
            z.read(zi)                          # raises on bad CRC / bad offset
        print(f"CRC ok for {len(sample)} sampled entries across volumes "
              f"{sorted({zi.volume for zi in sample})}")
        return
    z.extractall(a.out)
    print(f"extracted {len(names)} entries -> {a.out}")


if __name__ == "__main__":
    main()
