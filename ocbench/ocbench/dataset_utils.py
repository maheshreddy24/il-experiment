"""OCBench dataset utilities."""

import json
import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np


DATASET_REPO = 'seohongpark/ocbench'
DATASET_REVISION = 'fd771640f7a47879d8b06657fe05389641b5bff9'
DATASET_KEYS = ('observations', 'observation_interval', 'actions', 'rewards', 'masks', 'terminals')


def get_episode_offsets(dataset):
    """Get episode boundaries in transition and observation arrays.

    Args:
        dataset: Dataset dictionary with 'terminals', 'actions', 'observations', and 'observation_interval'.

    Returns:
        Transition offsets and observation offsets, each an array of length num_episodes + 1.
        Consecutive offsets give the start and end indices of each episode.
    """
    offsets = np.concatenate([[0], np.flatnonzero(dataset['terminals']) + 1])
    interval = int(dataset['observation_interval'])
    observation_counts = (np.diff(offsets) + interval - 1) // interval + 1
    observation_offsets = np.concatenate([[0], np.cumsum(observation_counts)])
    if offsets[-1] != len(dataset['actions']) or observation_offsets[-1] != len(dataset['observations']):
        raise ValueError('Observations must include each episode start, scheduled observations, and final endpoint.')
    return offsets, observation_offsets


def load_dataset(paths, episode_mask=None):
    """Load OCBench dataset files into one dictionary.

    Args:
        paths: Path to a local .npz file, or a list of paths to concatenate in order.
        episode_mask: Optional boolean mask with one entry per episode, in file order. Only episodes with
            True entries are loaded. If None, load all episodes.

    Returns:
        Dictionary of NumPy arrays with 'observations', 'observation_interval', 'actions', 'rewards', 'masks',
        and 'terminals'. Observations are stored every 'observation_interval' steps, including each episode's
        initial and final observations. Actions, rewards, masks, and terminals contain one entry per control
        step. 'terminals' marks the last step of each episode. 'masks' is 0 for environment termination and 1
        otherwise, including time-limit truncation. Use get_episode_offsets to find the episode boundaries.
    """
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    paths = [Path(path).expanduser() for path in paths]
    if episode_mask is not None:
        episode_mask = np.asarray(episode_mask, dtype=bool)

    def get_array_info(path, key):
        with zipfile.ZipFile(path) as file, file.open(f'{key}.npy') as array_file:
            version = np.lib.format.read_magic(array_file)
            assert version == (1, 0), f'Unsupported NPY format: {version}'
            shape, _, dtype = np.lib.format.read_array_header_1_0(array_file)
        return shape, dtype

    # Find episode ranges before allocating the merged arrays.
    path_ranges = []
    interval = None
    episode_offset = 0
    for path in paths:
        with np.load(path) as data:
            cur_interval = int(data['observation_interval'])
            terminals = data['terminals']
            offsets = np.concatenate([[0], np.flatnonzero(terminals) + 1])
            observation_counts = (np.diff(offsets) + cur_interval - 1) // cur_interval + 1
            observation_offsets = np.concatenate([[0], np.cumsum(observation_counts)])
        if interval is not None and interval != cur_interval:
            raise ValueError('Cannot combine datasets with different observation intervals.')
        interval = cur_interval
        if offsets[-1] != len(terminals) or observation_offsets[-1] != get_array_info(path, 'observations')[0][0]:
            raise ValueError(f'Invalid episode/observation boundaries in {path}.')
        num_episodes = len(offsets) - 1
        keep = range(num_episodes)
        if episode_mask is not None:
            keep = np.flatnonzero(episode_mask[episode_offset : episode_offset + num_episodes])
        episode_offset += num_episodes
        ranges = []
        for i in keep:
            start, end = offsets[i : i + 2]
            ob_start, ob_end = observation_offsets[i : i + 2]
            if ranges and ranges[-1][1] == start:
                ranges[-1] = (ranges[-1][0], end, ranges[-1][2], ob_end)
            else:
                ranges.append((start, end, ob_start, ob_end))
        path_ranges.append((path, ranges))
    if episode_mask is not None and episode_mask.shape != (episode_offset,):
        raise ValueError('Episode mask must have one entry per episode.')

    total_size = sum(end - start for _, ranges in path_ranges for start, end, _, _ in ranges)
    total_observations = sum(end - start for _, ranges in path_ranges for _, _, start, end in ranges)
    dataset = {'observation_interval': np.asarray(interval, dtype=np.int32)}
    keys = [key for key in DATASET_KEYS if key != 'observation_interval']
    for key in keys:
        shape, dtype = get_array_info(paths[0], key)
        size = total_observations if key == 'observations' else total_size
        dataset[key] = np.empty((size, *shape[1:]), dtype=dtype)

    # Load one array at a time to limit peak memory usage.
    offset = observation_offset = 0
    for path, ranges in path_ranges:
        with np.load(path) as data:
            for key in keys:
                values = data[key]
                dst = observation_offset if key == 'observations' else offset
                for start, end, ob_start, ob_end in ranges:
                    if key == 'observations':
                        start, end = ob_start, ob_end
                    count = end - start
                    dataset[key][dst : dst + count] = values[start:end]
                    dst += count
                del values
        offset += sum(end - start for start, end, _, _ in ranges)
        observation_offset += sum(end - start for _, _, start, end in ranges)
    return dataset


def download_datasets(env_name, dataset_root=None, num_shards=None):
    """Download OCBench datasets.

    Each training shard is downloaded with its validation data and metadata. Files are saved under
    `<dataset_root>/<lite|full|visual>/<env_name>/`.

    Args:
        env_name: Environment name (e.g., 'block-lite-single-task1-v0'). CPU and MJWarp environments share datasets.
        dataset_root: Root directory to save the datasets. Defaults to '$XDG_CACHE_HOME/ocbench/datasets' if
            XDG_CACHE_HOME is set, or '~/.cache/ocbench/datasets' otherwise.
        num_shards: Number of training shards to download, in filename order. If None, download all shards.

    Returns:
        Two lists of local training and validation dataset paths, paired in filename order.
    """
    env_name = env_name.replace('-cpu-', '-', 1)
    if dataset_root is None:
        dataset_root = Path(os.environ.get('XDG_CACHE_HOME') or '~/.cache') / 'ocbench' / 'datasets'
    dataset_root = Path(dataset_root).expanduser().resolve()

    # Load the release manifest.
    manifest_path = dataset_root / '.cache' / 'ocbench' / DATASET_REVISION / 'manifest.jsonl'
    if not manifest_path.is_file():
        from huggingface_hub import hf_hub_download

        hf_hub_download(
            DATASET_REPO,
            'manifest.jsonl',
            repo_type='dataset',
            revision=DATASET_REVISION,
            local_dir=manifest_path.parent,
        )
    manifest = [json.loads(line) for line in manifest_path.read_text().splitlines() if line]

    # Select training shards and their validation data and metadata.
    if env_name.startswith('visual-'):
        category = 'visual'
    elif '-lite-' in env_name:
        category = 'lite'
    else:
        category = 'full'
    dataset_subdir = f'{category}/{env_name}'
    files = [entry for entry in manifest if entry['path'].rsplit('/', 1)[0] == dataset_subdir]
    train_paths = sorted(entry['path'] for entry in files if entry['kind'] == 'data' and entry['split'] == 'train')
    if not train_paths:
        raise ValueError(f'No released dataset is available for {env_name}. Use load_dataset for custom data.')
    if num_shards is not None:
        train_paths = train_paths[:num_shards]
    if not train_paths:
        raise ValueError(f'No training shards selected for {env_name}.')
    selected_paths = set()
    for path in train_paths:
        stem = path.removesuffix('.npz')
        for suffix in ['.npz', '-val.npz', '-metadata.pkl', '-val-metadata.pkl']:
            selected_paths.add(stem + suffix)
    files = [entry for entry in files if entry['path'] in selected_paths]
    if {entry['path'] for entry in files} != selected_paths:
        raise ValueError(f'Incomplete release manifest for {env_name}.')
    train_paths = [str(dataset_root / path) for path in train_paths]
    val_paths = [path.removesuffix('.npz') + '-val.npz' for path in train_paths]

    # Check cached files against the sizes and modification times saved after the last download.
    receipt_path = dataset_root / '.cache' / 'ocbench' / f'{dataset_subdir}.json'
    receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else {}
    cached_files = receipt.get('files', {}) if receipt.get('revision') == DATASET_REVISION else {}
    files_to_download = []
    force_download = False
    for entry in files:
        path = dataset_root / entry['path']
        if not path.is_file():
            files_to_download.append(entry)
            continue
        stat = path.stat()
        if stat.st_size != entry['bytes']:
            files_to_download.append(entry)
            force_download = True
        elif cached_files.get(entry['path']) != [stat.st_size, stat.st_mtime_ns]:
            files_to_download.append(entry)
    if not files_to_download:
        print(f'Using cached dataset: {dataset_root / dataset_subdir} ({len(train_paths)} training shards)')
        return train_paths, val_paths

    # Download missing or changed files.
    from huggingface_hub import snapshot_download

    size = sum(entry['bytes'] for entry in files_to_download)
    print(
        f'Dataset: {env_name}\nLocation: {dataset_root / dataset_subdir}\nSelected: {len(train_paths)} training shards'
    )
    print(
        f'Checking/downloading {len(files_to_download)} files ({size / 1e9:.2f} GB); existing HF downloads are reused.'
    )
    snapshot_download(
        DATASET_REPO,
        repo_type='dataset',
        revision=DATASET_REVISION,
        local_dir=dataset_root,
        allow_patterns=[entry['path'] for entry in files_to_download],
        max_workers=4,
        force_download=force_download,
    )

    # Verify downloaded files and save the cache receipt.
    for entry in files:
        path = dataset_root / entry['path']
        if not path.is_file() or path.stat().st_size != entry['bytes']:
            raise OSError(f'Dataset file is missing or has the wrong size: {path}')
        stat = path.stat()
        cached_files[entry['path']] = [stat.st_size, stat.st_mtime_ns]
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=receipt_path.parent, delete=False) as file:
        json.dump(dict(revision=DATASET_REVISION, files=cached_files), file)
        temporary_path = file.name
    os.replace(temporary_path, receipt_path)
    print('Dataset ready.')
    return train_paths, val_paths
