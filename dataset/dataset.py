import torch
import os
import h5py
import json

NORM_STATS_FILE = "norm_stats.json"
# field -> (h5 dataset key, channel slice). pos/normals are 3-ch, values is 6-ch.
_STATS_FIELDS = {"pos": ("pos", slice(0, 3)),
                 "norm": ("normals", slice(0, 3)),
                 "out": ("values", slice(0, 6))}


def _streaming_mean_std(files):
    """Per-channel mean/std over all points in all files (single pass, low memory)."""
    count = 0
    sums = {k: None for k in _STATS_FIELDS}
    sqsums = {k: None for k in _STATS_FIELDS}
    for fp in files:
        with h5py.File(fp, "r") as h5:
            n = h5["pos"].shape[0]
            for k, (key, sl) in _STATS_FIELDS.items():
                arr = torch.from_numpy(h5[key][:, sl][:]).double()
                s, sq = arr.sum(0), (arr * arr).sum(0)
                sums[k] = s if sums[k] is None else sums[k] + s
                sqsums[k] = sq if sqsums[k] is None else sqsums[k] + sq
        count += n
    stats = {}
    for k in _STATS_FIELDS:
        mean = sums[k] / count
        var = (sqsums[k] / count) - mean * mean
        std = var.clamp_min(0).sqrt()
        stats[f"{k}_mean"] = mean.tolist()
        stats[f"{k}_std"] = std.clamp_min(1e-8).tolist()
    return stats


def load_or_compute_norm_stats(save_dir, f_list, recompute=False):
    """Return normalization tensors, computing them from the training h5 once and
    caching to <save_dir>/norm_stats.json. This makes the pipeline dataset-agnostic:
    migrating to a new dataset auto-derives the right statistics instead of relying
    on hardcoded constants.

    Returns a dict of (1,1,C) float tensors: pos_mean/std, norm_mean/std, out_mean/std.
    """
    stats_path = os.path.join(save_dir, NORM_STATS_FILE)
    if recompute or not os.path.exists(stats_path):
        stats = _streaming_mean_std(f_list)
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=4)
    else:
        with open(stats_path, "r") as f:
            stats = json.load(f)
    return {key: torch.tensor(val).view(1, 1, -1).float() for key, val in stats.items()}


class AirplaneDataset(torch.utils.data.Dataset):
    def __init__(self, path, train=True, split=None, train_set=None, split_size=600000):
        self.split_size = split_size
        # `split` ('train'|'val'|'test') selects the manifest key; when omitted we fall
        # back to the legacy `train` bool ('train' / 'test') for backward compatibility.
        if split is None:
            split = 'train' if train else 'test'
        # Resolve the split manifest: prefer one inside `path`, else the repo-root copy,
        # else an explicit override via the AIRPLANE_JSON env var.
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidates = [
            os.environ.get('AIRPLANE_JSON'),
            os.path.join(path, 'airplane_dataset.json'),
            os.path.join(repo_root, 'airplane_dataset.json'),
        ]
        filename = next((c for c in candidates if c and os.path.exists(c)), None)
        if filename is None:
            raise FileNotFoundError(
                f"airplane_dataset.json not found in {path!r}, repo root, or AIRPLANE_JSON")
        with open(filename, 'r') as f:
            data = json.load(f)
        key = f'{split}_set'
        if key in data:
            self.train_set = data[key]
        elif split == 'val' and 'test_set' in data:
            # legacy 2-way split has no val_set: reuse the test split for validation.
            print("WARNING: no 'val_set' in split manifest; falling back to 'test_set' for validation")
            self.train_set = data['test_set']
        else:
            raise KeyError(f"split key {key!r} not found in {filename}")
        if train_set is not None:
            self.train_set = train_set

        self.f_list = [os.path.join(path, f) for f in self.train_set]
        self.num_points_list = []
        for f in self.f_list:
            with h5py.File(f, "r") as h5:
                num_points = h5["pos"].shape[0]
            self.num_points_list.append(num_points)

    def __len__(self):
        return len(self.f_list)
    
    def __getitem__(self, idx):
        return self.f_list[idx], self.num_points_list[idx]

class AirplaneDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1, sampler=None):
        super(AirplaneDataLoader, self).__init__(dataset, batch_size=batch_size, sampler=sampler)
        self.split_size = dataset.split_size
    
    def __iter__(self):
        for idx in self.sampler:
            file, num_points = self.dataset[idx]
            start = 0
            end = num_points
            with h5py.File(file, "r") as f:
                pos = torch.from_numpy(f["pos"][start:end]).unsqueeze(0).float()
                normals = torch.from_numpy(f["normals"][start:end]).unsqueeze(0).float()
                values = torch.from_numpy(f["values"][start:end]).unsqueeze(0).float()
                # attributes
                mach = torch.tensor(f.attrs["Ma"]).unsqueeze(0).float()
                alpha = torch.tensor(f.attrs["alpha"]).unsqueeze(0).float()
                beta = torch.tensor(f.attrs["beta"]).unsqueeze(0).float()
                
            sdf = torch.zeros((1, num_points, 1))
            x = torch.cat([pos, sdf, normals], dim=-1)
            condition = torch.cat([mach, alpha, beta], dim=-1).unsqueeze(0)
            yield x, values, pos, condition, None