from operator import xor

from torch.utils.data import (
    ConcatDataset,
    DataLoader,
)

import src.datasets
from src.collate_fn.collate import (
    collate_fn,
)


def get_dataloaders(configs):
    dataloaders = {}
    for split, params in configs["data"].items():
        num_workers = params.get("num_workers", 1)

        if split == "train":
            drop_last = True
            shuffle = True
        else:
            drop_last = False
            shuffle = False

        datasets = []
        for ds in params["datasets"]:
            datasets.append(
                configs.init_obj(
                    ds,
                    src.datasets,
                    config_parser=configs,
                )
            )
        assert len(datasets)
        if len(datasets) > 1:
            dataset = ConcatDataset(datasets)
        else:
            dataset = datasets[0]

        assert xor(
            "batch_size" in params,
            "batch_sampler" in params,
        ), "You must provide batch_size or batch_sampler for each split"
        if "batch_size" in params:
            bs = params["batch_size"]
            batch_sampler = None
        else:
            raise Exception()
        if batch_sampler is None:
            dataloader = DataLoader(
                dataset,
                batch_size=bs,
                collate_fn=collate_fn,
                shuffle=shuffle,
                pin_memory=True,
                num_workers=num_workers,
                batch_sampler=batch_sampler,
                drop_last=split == "train",
            )
        else:
            dataloader = DataLoader(
                dataset,
                collate_fn=collate_fn,
                pin_memory=True,
                num_workers=num_workers,
                batch_sampler=batch_sampler,
            )
        dataloaders[split] = dataloader
    return dataloaders
