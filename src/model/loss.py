from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
from src.model.utils import BatchedData


class Similarity(nn.Module):
    def __init__(self, metric="cosine", chunk_size=64):
        super(Similarity, self).__init__()
        self.metric = metric
        self.chunk_size = chunk_size

    def forward(self, query, reference):
        query = F.normalize(query, dim=-1)
        reference = F.normalize(reference, dim=-1)
        similarity = F.cosine_similarity(query, reference, dim=-1)
        return similarity.clamp(min=0.0, max=1.0)


class PairwiseSimilarity(nn.Module):
    def __init__(self, metric="cosine"):
        super(PairwiseSimilarity, self).__init__()
        self.metric = metric

    def forward(self, query: torch.Tensor, reference: torch.Tensor, rx=None, rt=None) -> torch.Tensor:

        query_norm = F.normalize(query, dim=1)
        reference_norm = F.normalize(reference, dim=1)
        S = query_norm @ reference_norm.T

        if self.metric == "csls":
            assert rx is not None and rt is not None

            S = 2 * S - rx[:, None] - rt[None, :]

            return S

        return S.clamp(min=0.0, max=1.0)
