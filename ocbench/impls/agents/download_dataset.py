"""Download an OCBench dataset shard by shard, keeping free disk space for checkpoints.

python agents/download_dataset.py --env_name visual-block-double-task1-v0 --num_val 100 --reserve_gb 30
  - downloads the first `--num_val` validation shards (the held-out set) first,
  - then as many training shards as fit while leaving `--reserve_gb` free (all of them if possible),
  - skips files that are already complete. Prints how many training shards are on disk at the end.
"""
import argparse
import json
import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download
from ocbench.dataset_utils import DATASET_REPO, DATASET_REVISION


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', required=True)
    parser.add_argument('--num_train', type=int, default=None, help='max training shards (default: as many as fit)')
    parser.add_argument('--num_val', type=int, default=100, help='validation shards = held-out set')
    parser.add_argument('--reserve_gb', type=float, default=30.0, help='disk to keep free (checkpoints, logs, margin)')
    args = parser.parse_args()

    root = Path(os.environ.get('XDG_CACHE_HOME') or '~/.cache').expanduser() / 'ocbench' / 'datasets'
    category = 'visual' if args.env_name.startswith('visual-') else ('lite' if '-lite-' in args.env_name else 'full')
    subdir = f'{category}/{args.env_name}'
    manifest_dir = root / '.cache' / 'ocbench' / DATASET_REVISION
    if not (manifest_dir / 'manifest.jsonl').is_file():
        hf_hub_download(DATASET_REPO, 'manifest.jsonl', repo_type='dataset', revision=DATASET_REVISION,
                        local_dir=manifest_dir)
    manifest = [json.loads(line) for line in (manifest_dir / 'manifest.jsonl').read_text().splitlines() if line]
    files = {e['path']: e for e in manifest if e['path'].rsplit('/', 1)[0] == subdir}
    train = sorted(p for p, e in files.items() if e['kind'] == 'data' and e['split'] == 'train')
    val = [p.removesuffix('.npz') + '-val.npz' for p in train][: args.num_val]
    if args.num_train is not None:
        train = train[: args.num_train]
    print(f'{args.env_name}: {len(files)} files in the release; up to {len(train)} train + {len(val)} val shards')

    def have(path):
        local = root / path
        return local.is_file() and local.stat().st_size == files[path]['bytes']

    def fetch(path):
        if not have(path):
            hf_hub_download(DATASET_REPO, path, repo_type='dataset', revision=DATASET_REVISION, local_dir=root)

    for i, path in enumerate(val):
        fetch(path)
        if (i + 1) % 25 == 0:
            print(f'  val shards: {i + 1}/{len(val)}', flush=True)

    n_train = 0
    for path in train:
        if not have(path):
            free_gb = shutil.disk_usage(root).free / 1e9
            if free_gb - files[path]['bytes'] / 1e9 < args.reserve_gb:
                print(f'  stopping: {free_gb:.1f} GB free, keeping {args.reserve_gb:.0f} GB reserved', flush=True)
                break
            fetch(path)
        n_train += 1
        if n_train % 25 == 0:
            print(f'  train shards: {n_train}/{len(train)} ({shutil.disk_usage(root).free / 1e9:.0f} GB free)', flush=True)

    print(f'DONE: {n_train} train shards + {len(val)} val shards on disk; '
          f'{shutil.disk_usage(root).free / 1e9:.1f} GB free')


if __name__ == '__main__':
    main()
