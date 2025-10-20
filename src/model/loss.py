import torch
from torch import nn
import torch.nn.functional as F


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


def compute_csls_terms(proposal_descriptors, template_descriptors, k=10):
    objs = sorted(template_descriptors.keys())

    splits = [0]
    for o in objs:
        splits.append(splits[-1] + template_descriptors[o].size(0))

    template = torch.cat([template_descriptors[o] for o in objs], dim=0)  # [Nt, D]
    prop = proposal_descriptors  # [Nq, D]

    prop = F.normalize(prop, dim=1)
    template = F.normalize(template, dim=1)

    S = prop @ template.T  # [Nq, Nt]

    kx = min(k, max(1, template.size(0) - 1))
    rx = torch.topk(S, k=kx, dim=1).values.mean(dim=1)  # [Nq]

    T = template @ template.T  # [Nt, Nt]
    T.fill_diagonal_(-float('inf'))  # exclude self
    kt = min(k, max(1, template.size(0) - 1))
    rt = torch.topk(T, k=kt, dim=1).values.mean(dim=1)  # [Nt]

    return rx, rt, splits


def cosine_similarity(query: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:

    query_norm = F.normalize(query, dim=1)
    reference_norm = F.normalize(reference, dim=1)
    S = query_norm @ reference_norm.T

    return S.clamp(min=-1.0, max=1.0)


def csls_score(query: torch.Tensor, reference: torch.Tensor, rx: torch.Tensor, rt: torch.Tensor) -> torch.Tensor:

    query_norm = F.normalize(query, dim=1)
    reference_norm = F.normalize(reference, dim=1)
    S = cosine_similarity(query_norm, reference_norm)

    S_csls = 2 * S - rx[:, None] - rt[None, :]

    return S_csls


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
