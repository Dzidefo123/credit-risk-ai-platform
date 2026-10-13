"""Unpack provider zip archives into ``<raw_dir>/extracted`` without touching the archives.

The full-volume SFLLD download is a yearly zip of quarterly zips, each holding
``orig_YYYYQn.txt`` and ``perf_YYYYQn.txt``. Archives stay as delivered; text members are
extracted (zip CRC-checked by ``zipfile``) and an unpack manifest records archive checksums.
Re-running skips members whose extracted size already matches.
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from crisk.config import DatasetConfig
from crisk.utils import sha256_file, utc_now, write_json


def unpack_archives(cfg: DatasetConfig, pattern: str = "**/*Q[1-4].zip") -> dict:
    out_dir = cfg.raw_dir / "extracted"
    out_dir.mkdir(parents=True, exist_ok=True)
    archives = sorted(p for p in cfg.raw_dir.glob(pattern) if out_dir not in p.parents)
    if not archives:
        raise FileNotFoundError(f"No archives matching {pattern!r} under {cfg.raw_dir}")
    entries = []
    for z in archives:
        with zipfile.ZipFile(z) as zf:
            for info in zf.infolist():
                if not info.filename.endswith(".txt"):
                    continue
                target = out_dir / Path(info.filename).name
                status = "skipped_exists"
                if not target.exists() or target.stat().st_size != info.file_size:
                    tmp = target.with_suffix(".txt.part")
                    with zf.open(info) as src, tmp.open("wb") as dst:
                        shutil.copyfileobj(src, dst, length=1 << 22)  # zipfile verifies the CRC at EOF
                    tmp.replace(target)
                    status = "extracted"
                entries.append({"archive": str(z.relative_to(cfg.raw_dir)), "member": info.filename,
                                "bytes": info.file_size, "crc32": f"{info.CRC:08x}", "status": status})
    manifest = {"unpacked_at": utc_now(), "out_dir": str(out_dir),
                "archives": {str(z.relative_to(cfg.raw_dir)): sha256_file(z) for z in archives}, "members": entries}
    write_json(cfg.raw_dir / "extracted" / "_unpack_manifest.json", manifest)
    return manifest
