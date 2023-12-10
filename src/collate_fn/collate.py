import torch
from torch.nn.utils.rnn import (
    pad_sequence,
)


def collate_fn(batch):
    out = {"targets": torch.tensor([item["target"] for item in batch])}
    if "mel" in batch[0].keys():
        out["mels"] = pad_sequence(
            [item["mel"] for item in batch],
            batch_first=True,
            padding_value=0,
        )
    elif "wave" in batch[0].keys():
        out["waves"] = pad_sequence(
            [item["wave"] for item in batch], batch_first=True, padding_value=0
        )
    return out
