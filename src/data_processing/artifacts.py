"""Replay session roster, artifact hashing, and atomic receipt writes."""
import csv
import hashlib
import json
import os
import stat
from pathlib import Path

from . import storage

TARGETS = ('odor', 'position')
SESSIONS = ('080718_mitt', '081106_barat', '090212_stella', '090212_superchris', '090420_buchanan')
ALIASES = dict(zip(SESSIONS, ('Mitt', 'Barat', 'Stella', 'Superchris', 'Buchanan'), strict=True))
CLASSES = (1, 2, 3, 4, 5)
MODELS = ('historical_anchor', 'cebra_combined_mean')


def job_key(target, model, session, split):
    return f'{target}/{model}/{session}/{split}'


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def plain(path):
    path = Path(os.path.abspath(path))
    for part in (*reversed(path.parents), path):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        require(not stat.S_ISLNK(info.st_mode), f'Symlink forbidden: {part}')
        if part == path and stat.S_ISREG(info.st_mode):
            require(info.st_nlink == 1, f'Hard-linked artifact forbidden: {part}')
    return path


def within(root, name):
    rel = Path(name)
    require(not rel.is_absolute() and '..' not in rel.parts, f'Noncanonical relative path: {name}')
    return plain(root / rel)


def sha(path):
    with plain(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    return json.loads(plain(path).read_bytes())

def commit(path, payload):
    path = plain(path)
    if path.exists():
        require(canonical(read(path)) == canonical(payload), f'Immutable receipt differs: {path}')
    else:
        storage.write_json(path, payload)
    return sha(path)


def csv_write(path, rows, fields=()):
    if not fields:
        fields = list(dict.fromkeys(k for row in rows for k in row))
    with storage.temporary_for(plain(path)) as tmp, tmp.open('w', newline='') as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows({k: canonical(v).decode() if isinstance(v, (dict, list, tuple)) else v
                          for k, v in row.items()} for row in rows)
