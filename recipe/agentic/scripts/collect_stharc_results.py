#!/usr/bin/env python3
"""Snapshot experiment evidence without modifying sources; standard library only."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tarfile
from datetime import datetime, timezone


TEXT_EXTENSIONS = {'.log', '.yaml', '.yml', '.json', '.csv', '.tsv', '.txt', '.md', '.sh'}
PRUNE = {'.git', '__pycache__', 'node_modules', 'wandb', '.venv', 'venv'}
WEIGHTS = {'.pt', '.pth', '.bin', '.safetensors', '.ckpt', '.onnx', '.npy', '.npz'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home', type=Path, default=Path('/mnt/workspace/stharc'))
    parser.add_argument('--root', type=Path, action='append', default=[],
                        help='Additional result tree; repeatable (runs is always scanned).')
    parser.add_argument('--repo', type=Path, help='Optional training code checkout.')
    parser.add_argument('--include-rollouts', action='store_true')
    parser.add_argument('--max-file-mb', type=float, default=64)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.max_file_mb < float('inf'):
        parser.error('--max-file-mb must be positive and finite')
    limit = int(args.max_file_mb * 1024 * 1024)
    output = args.output.expanduser().absolute()
    home = args.home.expanduser().absolute()
    roots = [home / 'runs'] + [p.expanduser().absolute() for p in args.root]
    records, errors, missing = [], [], []
    seen = set()
    included_bytes = 0
    started = datetime.now(timezone.utc).isoformat()
    output.parent.mkdir(parents=True, exist_ok=True)

    def error(exc):
        errors.append(str(exc))

    def candidates():
        for index, root in enumerate(roots):
            if not root.is_dir():
                missing.append(str(root))
                continue
            for directory, dirs, files in os.walk(root, onerror=error, followlinks=False):
                dirs[:] = sorted(d for d in dirs if d not in PRUNE)
                for name in sorted(files):
                    path = Path(directory) / name
                    yield path, 'sources/root_%02d/%s' % (index, path.relative_to(root))
        if home.is_dir():
            for path in sorted(home.glob('*.log')):
                yield path, 'sources/home_logs/' + path.name
        else:
            missing.append(str(home))
        if args.repo:
            scripts = args.repo.expanduser().absolute() / 'recipe/agentic/scripts'
            if not scripts.is_dir():
                missing.append(str(scripts))
            else:
                for path in sorted(scripts.glob('*stharc*.sh')):
                    yield path, 'sources/launchers/' + path.name

    def add_bytes(archive, name, data, mtime=0):
        info = tarfile.TarInfo(name)
        info.size = len(data)
        info.mtime = int(mtime)
        info.mode = 0o600
        archive.addfile(info, io.BytesIO(data))

    # Exclusive creation prevents accidental replacement of a previous evidence bundle.
    with output.open('xb') as raw:
        os.chmod(output, 0o600)
        with tarfile.open(fileobj=raw, mode='w:gz') as archive:
            for path, member in candidates():
                if path == output or path in seen:
                    continue
                seen.add(path)
                record = {'source': str(path), 'archive_path': member}
                records.append(record)
                try:
                    if path.is_symlink():
                        record['status'] = 'skipped_symlink'
                        continue
                    before = path.stat()
                    record.update(size=before.st_size, mtime_ns=before.st_mtime_ns)
                    parts = {part.lower() for part in path.parts}
                    suffix = path.suffix.lower()
                    in_checkpoint = bool(parts & {'checkpoints', 'checkpoint'})
                    rollout = bool(parts & {'rollout_logs', 'traces'}) or suffix == '.jsonl'
                    if suffix in WEIGHTS or (in_checkpoint and suffix not in TEXT_EXTENSIONS):
                        record['status'] = 'checkpoint_or_binary_inventory_only'
                        continue
                    config_or_log = '.hydra' in parts or suffix in {'.yaml', '.yml', '.log'}
                    if rollout and not config_or_log and not args.include_rollouts:
                        record['status'] = 'rollout_inventory_only'
                        continue
                    if suffix not in TEXT_EXTENSIONS | {'.jsonl'} and not path.name.startswith('events.out.tfevents'):
                        record['status'] = 'unsupported_inventory_only'
                        continue
                    if before.st_size > limit:
                        record['status'] = 'skipped_size_limit'
                        continue
                    with path.open('rb') as source:
                        data = source.read(limit + 1)
                    if len(data) > limit:
                        record['status'] = 'skipped_size_limit'
                        continue
                    after = path.stat()
                    changed = (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)
                    record['status'] = 'included_changing' if changed else 'included'
                    record['snapshot_bytes'] = len(data)
                    record['sha256'] = hashlib.sha256(data).hexdigest()
                    add_bytes(archive, member, data, before.st_mtime)
                    included_bytes += len(data)
                except OSError as exc:
                    record['status'] = 'read_error'
                    record['error'] = str(exc)
                    error(exc)
            counts = {}
            for record in records:
                status = record['status']
                counts[status] = counts.get(status, 0) + 1
            metadata = {
                'format_version': 1,
                'started_utc': started,
                'finished_utc': datetime.now(timezone.utc).isoformat(),
                'roots': [str(p) for p in roots],
                'home': str(home),
                'include_rollouts': args.include_rollouts,
                'max_file_bytes': limit,
                'missing_paths': missing,
                'errors': errors,
                'counts': counts,
                'included_bytes': included_bytes,
                'notes': [
                    'No run selected by score; names do not establish dataset or seed.',
                    'Modified, recovered and original evidence must be distinguished during analysis.',
                    'This is not an atomic snapshot; changing files are marked when detected.',
                    'Checkpoint weights are excluded; other exclusions appear in manifest.',
                    'Only specified roots, top-level home logs and optional launchers are scanned.',
                    'No process status or training completion is inferred by this collector.',
                ],
            }
            add_bytes(archive, 'collection.json', json.dumps(metadata, indent=2).encode())
            manifest = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records)
            add_bytes(archive, 'manifest.jsonl', manifest.encode())
    print(json.dumps({'output': str(output), 'archive_bytes': output.stat().st_size,
                      **metadata}, ensure_ascii=False, indent=2))
    return 2 if errors or missing or not included_bytes else 0


if __name__ == '__main__':
    sys.exit(main())