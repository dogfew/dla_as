from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class AngularSoftmax(nn.Module):
    def __init__(self, in_features=2, out_features=16, margin=2, eps=1e-9, **kwargs):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.margin = margin
        self.W = nn.Parameter(torch.Tensor(out_features, in_features))
        self.eps = eps
        nn.init.xavier_uniform_(self.W)

    def forward(self, logits: Tensor, targets: Tensor, **kwargs):
        W = F.normalize(self.W, p=2, dim=1, eps=self.eps)
        x = F.normalize(logits, p=2, dim=1, eps=self.eps)
        cosine = x @ W.T
        numerator = torch.cos(
            torch.acos(
                torch.diagonal(
                    cosine.transpose(0, 1)[targets]
                               ).clamp(-1. + self.eps, 1. - self.eps)
            ) * self.margin
        )
        mask = torch.ones_like(cosine, dtype=torch.bool).scatter_(
            1, targets.unsqueeze(1), False
        )
        other = cosine[mask].view(cosine.size(0), -1)
        denom = numerator.exp() + other.exp().sum(dim=1)
        denom = torch.clamp(denom, self.eps)
        L = numerator - torch.log(denom)
        with torch.no_grad():
            probabilities = torch.softmax(logits, dim=1)
            predicted_classes = torch.argmax(probabilities, dim=1)
            accuracy = accuracy_score(
                targets.cpu().numpy(), predicted_classes.cpu().numpy()
            )
        loss = -torch.mean(L)
        return {
            "loss": loss,
            "probs": probabilities,
            "accuracy": accuracy,
        }
