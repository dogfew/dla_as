import torch
from sklearn.metrics import accuracy_score
from torch import Tensor
from torch.nn import CrossEntropyLoss


class CrossEntropyLossWrapper(CrossEntropyLoss):
    def __init__(self, weights=(0.1, 1), *args, **kwargs):
        super().__init__(weight=torch.tensor(weights))

    def forward(self, logits: Tensor, targets: Tensor, **kwargs) -> Tensor:
        loss = super().forward(logits, targets)
        with torch.no_grad():
            probabilities = torch.softmax(logits, dim=1)
            predicted_classes = torch.argmax(probabilities, dim=1)
            accuracy = accuracy_score(
                targets.cpu().numpy(), predicted_classes.cpu().numpy()
            )
        return {
            "loss": loss,
            "probs": probabilities,
            "accuracy": accuracy,
        }
