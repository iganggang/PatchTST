import os

from src.data.datamodule import DataLoaders
from src.data.pred_dataset import Dataset_UEA


def _resolve_data_path(root_path: str, data_path: str) -> str:
    candidate = os.path.join(root_path, data_path)
    if os.path.exists(candidate):
        return data_path
    if not os.path.splitext(data_path)[1]:
        npz_name = f"{data_path}.npz"
        candidate = os.path.join(root_path, npz_name)
        if os.path.exists(candidate):
            return npz_name
    return data_path


def get_dls(params):
    root_path = getattr(params, "root_path", "./data/UEA")
    data_path = getattr(params, "data_path", None) or params.dset
    data_path = _resolve_data_path(root_path, data_path)

    dataset_kwargs = {
        "root_path": root_path,
        "data_path": data_path,
        "val_ratio": getattr(params, "val_ratio", 0.2),
        "scale": bool(getattr(params, "scale", False)),
    }

    dls = DataLoaders(
        datasetCls=Dataset_UEA,
        dataset_kwargs=dataset_kwargs,
        batch_size=params.batch_size,
        workers=getattr(params, "num_workers", 0),
        shuffle_train=True,
        shuffle_val=False,
    )

    if dls.train is None or len(dls.train.dataset) == 0:
        raise RuntimeError("Training dataset is empty. Check dataset path and splits.")

    sample_x, _ = dls.train.dataset[0]
    dls.vars = dls.train.dataset.n_vars if hasattr(dls.train.dataset, "n_vars") else sample_x.shape[-1]
    dls.len = dls.train.dataset.seq_len if hasattr(dls.train.dataset, "seq_len") else sample_x.shape[0]
    dls.c = dls.train.dataset.n_classes if hasattr(dls.train.dataset, "n_classes") else None
    return dls


if __name__ == "__main__":
    class Params:
        dset = "BasicMotions"
        batch_size = 16
        num_workers = 0
        val_ratio = 0.2
        scale = False
        root_path = "./data/UEA"

    params = Params()
    dls = get_dls(params)
    if dls.valid:
        for i, batch in enumerate(dls.valid):
            print(i, len(batch), batch[0].shape, batch[1].shape)
            if i > 2:
                break
