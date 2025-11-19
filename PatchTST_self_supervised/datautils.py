from src.data.datamodule import DataLoaders
from src.data.uea_dataset import Dataset_UEA


UEA_DATASETS = [
    "ArticularyWordRecognition",
    "BasicMotions",
    "Cricket",
    "EthanolConcentration",
    "ERing",
    "HandMovementDirection",
    "Handwriting",
    "Heartbeat",
    "Libras",
    "MotorImagery",
    "NATOPS",
    "PenDigits",
    "PhonemeSpectra",
    "RacketSports",
    "SelfRegulationSCP1",
    "SelfRegulationSCP2",
    "SpokenArabicDigits",
    "StandWalkJump",
]


def get_dls(params):
    if hasattr(params, "dset") and UEA_DATASETS and params.dset not in UEA_DATASETS:
        print(f"Warning: dataset {params.dset} not in predefined UEA list. Proceeding regardless.")

    dataset_kwargs = {
        "root_path": getattr(params, "root_path", "./data/UEA"),
        "data_path": params.dset,
        "seq_len": getattr(params, "context_points", None),
        "normalize": bool(getattr(params, "normalize", True)),
        "val_ratio": getattr(params, "val_ratio", 0.1),
        "random_seed": getattr(params, "split_seed", 42),
    }

    dls = DataLoaders(
        datasetCls=Dataset_UEA,
        dataset_kwargs=dataset_kwargs,
        batch_size=params.batch_size,
        workers=params.num_workers,
        shuffle_train=True,
        shuffle_val=False,
    )

    if dls.train is None:
        raise RuntimeError("Training dataloader could not be created. Please check dataset path.")

    train_dataset = dls.train.dataset
    dls.vars = train_dataset.n_vars
    dls.c = train_dataset.n_classes
    dls.len = train_dataset.seq_len
    return dls
