import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset


class Dataset_UEA(Dataset):
    """Time-series classification dataset loader for UEA style archives."""

    _CACHE: Dict[Tuple[str, str, int, bool, float, int], Dict] = {}

    def __init__(
        self,
        root_path: str,
        data_path: str,
        split: str = "train",
        seq_len: int = None,
        normalize: bool = True,
        val_ratio: float = 0.1,
        random_seed: int = 42,
    ):
        super().__init__()
        split = split.lower()
        cache_key = (
            os.path.abspath(root_path),
            data_path,
            -1 if seq_len is None else int(seq_len),
            bool(normalize),
            float(val_ratio),
            int(random_seed),
        )
        if cache_key not in Dataset_UEA._CACHE:
            Dataset_UEA._CACHE[cache_key] = self._prepare_splits(
                root_path=root_path,
                data_path=data_path,
                seq_len=seq_len,
                normalize=normalize,
                val_ratio=val_ratio,
                random_seed=random_seed,
            )
        cache = Dataset_UEA._CACHE[cache_key]
        if split not in cache["splits"]:
            raise ValueError(f"Unknown split '{split}'. Available: {list(cache['splits'].keys())}")
        tensors = cache["splits"][split]
        self.data = tensors["data"]
        self.targets = tensors["labels"]
        self.seq_len = cache["meta"]["seq_len"]
        self.n_vars = cache["meta"]["n_vars"]
        self.n_classes = cache["meta"]["n_classes"]
        self.n_inp = 1

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

    @classmethod
    def _prepare_splits(cls, root_path, data_path, seq_len, normalize, val_ratio, random_seed):
        train_x, train_y, test_x, test_y = cls._load_raw_arrays(root_path, data_path)
        train_x = cls._format_batch(train_x)
        test_x = cls._format_batch(test_x)
        train_y = cls._reshape_labels(train_y)
        test_y = cls._reshape_labels(test_y)

        label_encoder = LabelEncoder()
        label_encoder.fit(np.concatenate([train_y, test_y], axis=0))
        train_y = label_encoder.transform(train_y)
        test_y = label_encoder.transform(test_y)

        if normalize:
            train_x = cls._normalize_per_sample(train_x)
            test_x = cls._normalize_per_sample(test_x)

        if seq_len is None:
            seq_len = int(max(train_x.shape[1], test_x.shape[1]))
        train_x = cls._resize_batch(train_x, seq_len)
        test_x = cls._resize_batch(test_x, seq_len)

        if val_ratio > 0:
            val_ratio = min(max(val_ratio, 1e-6), 0.5)
            stratify = train_y if len(np.unique(train_y)) > 1 else None
            train_x, val_x, train_y, val_y = train_test_split(
                train_x,
                train_y,
                test_size=val_ratio,
                random_state=random_seed,
                stratify=stratify,
            )
        else:
            val_x = np.empty((0, seq_len, train_x.shape[2]), dtype=train_x.dtype)
            val_y = np.empty((0,), dtype=train_y.dtype)

        meta = {"seq_len": seq_len, "n_vars": train_x.shape[2], "n_classes": len(label_encoder.classes_)}
        splits = {
            "train": cls._to_tensors(train_x, train_y),
            "val": cls._to_tensors(val_x, val_y),
            "test": cls._to_tensors(test_x, test_y),
        }
        return {"splits": splits, "meta": meta}

    @staticmethod
    def _to_tensors(data, labels):
        data = data.astype(np.float32, copy=False)
        labels = labels.astype(np.int64, copy=False)
        return {
            "data": torch.from_numpy(data),
            "labels": torch.from_numpy(labels),
        }

    @staticmethod
    def _reshape_labels(labels):
        labels = np.asarray(labels)
        if labels.ndim > 1:
            labels = labels.squeeze()
        return labels

    @staticmethod
    def _format_batch(data):
        data = np.asarray(data)
        if data.ndim == 2:
            data = data[:, :, None]
        if data.ndim != 3:
            raise ValueError(f"Expected 3D array (samples, length, channels), got shape {data.shape}")
        # ensure layout [N, L, C]
        if data.shape[1] < data.shape[2]:
            data = np.transpose(data, (0, 2, 1))
        return data

    @staticmethod
    def _normalize_per_sample(data):
        mean = data.mean(axis=1, keepdims=True)
        std = data.std(axis=1, keepdims=True)
        std[std < 1e-5] = 1.0
        return (data - mean) / std

    @staticmethod
    def _resize_batch(data, target_len):
        if data.shape[1] == target_len:
            return data
        if data.shape[1] > target_len:
            return data[:, -target_len:, :]
        pad_len = target_len - data.shape[1]
        pad = np.zeros((data.shape[0], pad_len, data.shape[2]), dtype=data.dtype)
        return np.concatenate([pad, data], axis=1)

    @classmethod
    def _load_raw_arrays(cls, root_path, data_path):
        root = Path(root_path).expanduser()
        base = root / data_path
        if base.is_file():
            return cls._read_combined_file(base)
        if base.with_suffix(".npz").is_file():
            return cls._read_combined_file(base.with_suffix(".npz"))
        if base.is_dir():
            combined = base / f"{base.name}.npz"
            if combined.is_file():
                return cls._read_combined_file(combined)
            train_file = base / "train.npz"
            test_file = base / "test.npz"
            if train_file.is_file() and test_file.is_file():
                return cls._read_split_files(train_file, test_file)
            upper_train = base / f"{base.name}_TRAIN.npz"
            upper_test = base / f"{base.name}_TEST.npz"
            if upper_train.is_file() and upper_test.is_file():
                return cls._read_split_files(upper_train, upper_test)
        # fall back to parent directory based matches
        stem = base.name if base.is_dir() else base.stem
        parent = base.parent
        if parent is None or not parent.exists():
            parent = root
        train_candidates = [
            parent / f"{stem}_train.npz",
            parent / f"{stem}_TRAIN.npz",
        ]
        test_candidates = [
            parent / f"{stem}_test.npz",
            parent / f"{stem}_TEST.npz",
        ]
        for t_file, v_file in zip(train_candidates, test_candidates):
            if t_file.is_file() and v_file.is_file():
                return cls._read_split_files(t_file, v_file)
        raise FileNotFoundError(
            f"Could not locate dataset '{data_path}' under '{root_path}'. "
            "Expected either a combined NPZ file or train/test NPZ pairs."
        )

    @staticmethod
    def _read_combined_file(path):
        with np.load(path, allow_pickle=True) as handler:
            train_x = Dataset_UEA._extract_array(handler, ["X_train", "x_train", "train_x", "trainX", "train"])
            train_y = Dataset_UEA._extract_array(handler, ["y_train", "Y_train", "train_y", "trainY", "label_train"])
            test_x = Dataset_UEA._extract_array(handler, ["X_test", "x_test", "test_x", "testX", "test"])
            test_y = Dataset_UEA._extract_array(handler, ["y_test", "Y_test", "test_y", "testY", "label_test"])
        return train_x, train_y, test_x, test_y

    @staticmethod
    def _read_split_files(train_file, test_file):
        with np.load(train_file, allow_pickle=True) as train_handler:
            train_x = Dataset_UEA._extract_array(train_handler, ["x", "X", "data", "samples"])
            train_y = Dataset_UEA._extract_array(train_handler, ["y", "Y", "labels", "target"])
        with np.load(test_file, allow_pickle=True) as test_handler:
            test_x = Dataset_UEA._extract_array(test_handler, ["x", "X", "data", "samples"])
            test_y = Dataset_UEA._extract_array(test_handler, ["y", "Y", "labels", "target"])
        return train_x, train_y, test_x, test_y

    @staticmethod
    def _extract_array(handler, keys):
        for key in keys:
            if key in handler:
                return handler[key]
        raise ValueError(f"None of the keys {keys} found in NPZ file {list(handler.keys())}")
