"""Inventory project storage; never remove, move, or open file contents."""
import ctypes
import datetime
import json
import os
import time
from pathlib import Path
from collections import defaultdict


def main():
    repo = Path(__file__).resolve().parents[1]
    roots = [repo, Path(r'C:\Users\jsowa\benchmark-translations')]
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    compressed = kernel.GetCompressedFileSizeW
    compressed.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_uint32)]
    compressed.restype = ctypes.c_uint32
    seen = set()
    groups = defaultdict(lambda: dict(files=0, logical_bytes=0, allocated_unique_bytes=0, allocation_unknown_files=0))
    largest, junctions, errors, resumes, archives = [], [], [], [], []
    next_progress = time.monotonic() + 25
    for root in roots:
        stack = [root]
        while stack:
            directory = stack.pop()
            try:
                with os.scandir(directory) as entries:
                    for entry in entries:
                        try:
                            # DirEntry.stat returns zero file IDs on Windows; os.stat
                            # is required to identify real hardlinks correctly.
                            stat = os.stat(entry.path, follow_symlinks=False)
                            if (getattr(stat, 'st_file_attributes', 0) & 0x400 and entry.is_dir(follow_symlinks=False)) or entry.is_symlink():
                                try:
                                    target = os.readlink(entry.path)
                                except (OSError, ValueError):
                                    target = 'unresolved reparse point; not traversed'
                                junctions.append(dict(path=entry.path, target=target))
                                continue
                            if entry.is_dir(follow_symlinks=False):
                                stack.append(Path(entry.path))
                                continue
                            high = ctypes.c_uint32()
                            ctypes.set_last_error(0)
                            low = compressed(entry.path, ctypes.byref(high))
                            allocation_error = ctypes.get_last_error() if low == 0xffffffff else 0
                            allocated = None if allocation_error else (high.value << 32) | low
                            identity = (stat.st_dev, stat.st_ino)
                            unique = identity not in seen
                            seen.add(identity)
                            relative = Path(entry.path).relative_to(root)
                            for depth in range(min(4, len(relative.parts))):
                                group = str(root.joinpath(*relative.parts[:depth]))
                                summary = groups[group]
                                summary['files'] += 1
                                summary['logical_bytes'] += stat.st_size
                                summary['allocated_unique_bytes'] += allocated if unique and allocated is not None else 0
                                summary['allocation_unknown_files'] += int(allocation_error != 0)
                            item = dict(path=entry.path, logical_bytes=stat.st_size,
                                        allocated_bytes=allocated, links=stat.st_nlink,
                                        modified_utc=datetime.datetime.fromtimestamp(stat.st_mtime, datetime.timezone.utc).isoformat())
                            if stat.st_size >= 20 * 1024**2:
                                largest.append(item)
                            if entry.name == 'last.pt':
                                resumes.append(item)
                            if entry.name.lower().endswith(('.zip', '.tar', '.gz', '.7z', '.vhdx')):
                                archives.append(item)
                            if time.monotonic() >= next_progress:
                                print(json.dumps(dict(scanned_unique_files=len(seen), current_directory=str(directory))), flush=True)
                                next_progress = time.monotonic() + 25
                        except OSError as exc:
                            errors.append(dict(path=entry.path, error=str(exc)))
            except OSError as exc:
                errors.append(dict(path=str(directory), error=str(exc)))
    result = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  policy='read-only metadata scan; no file contents read; no junction traversal; no deletions',
                  roots=[str(r) for r in roots],
                  groups=[dict(path=k, **v) for k, v in sorted(groups.items(), key=lambda p: -p[1]['allocated_unique_bytes'])],
                  large_files=sorted(largest, key=lambda p: -(p['allocated_bytes'] or 0)),
                  resume_files=resumes, archives=archives, junctions=junctions, errors=errors)
    result['supersedes'] = 'v1 had zero file IDs; v2 failed to exclude GetCompressedFileSizeW error sentinels. Do not use their aggregate allocation sums.'
    result['allocation_note'] = 'GetCompressedFileSizeW with unique file IDs; directory metadata and cluster rounding excluded; live directories can change during scan'
    output = repo / 'reports/series_2026-10-23/storage_readonly_inventory_v3.json'
    with output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(dict(report=str(output), groups=result['groups'][:25], large_files=result['large_files'][:12],
                          error_count=len(errors), junctions=junctions, resume_count=len(resumes)), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
