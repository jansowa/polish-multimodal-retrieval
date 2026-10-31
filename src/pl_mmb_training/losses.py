"""Loss-only ablations; every known caption/image association is a positive."""
import math
from collections import defaultdict

import torch
import torch.nn.functional as F


class PositivePairs:
    def __init__(self, records):
        self.records = records
        self.targets = [r['target_id'] for r in records]
        self.known = defaultdict(set)
        for r in records:
            self.known[r['text_pl']].add(r['target_id'])

    def mask(self, indices):
        columns = defaultdict(list)
        for col, index in enumerate(indices):
            columns[self.targets[index]].append(col)
        row_indices, col_indices = [], []
        for row, index in enumerate(indices):
            for target in self.known[self.records[index]['text_pl']]:
                positions = columns.get(target, ())
                row_indices.extend([row] * len(positions))
                col_indices.extend(positions)
        result = torch.zeros((len(indices), len(indices)), dtype=torch.bool)
        result[row_indices, col_indices] = True
        return result


def symmetric_infonce(logits, positives):
    if positives.shape != logits.shape or not positives.any(0).all() or not positives.any(1).all():
        raise ValueError('Both contrastive directions require positives')
    masked = logits.masked_fill(~positives, -torch.inf)
    # Total probability assigned to any known positive, in both directions.
    return .5 * ((torch.logsumexp(logits, 1) - torch.logsumexp(masked, 1)).mean()
                 + (torch.logsumexp(logits, 0) - torch.logsumexp(masked, 0)).mean())


def sigmoid_loss(logits, positives):
    if positives.shape != logits.shape or logits.shape[0] == 0:
        raise ValueError('Invalid sigmoid pair matrix')
    labels = positives.to(logits.dtype) * 2 - 1
    # SigLIP: sum pair losses and divide by batch size, not by N**2.
    return F.softplus(-labels * logits).sum() / logits.shape[0]


class PairLoss(torch.nn.Module):
    def __init__(self, kind, temperature=.07, sigmoid_bias=-10.):
        super().__init__()
        if kind not in ('symmetric_infonce', 'siglip_sigmoid') or not 0 < temperature <= 1:
            raise ValueError('Invalid pair loss or temperature')
        self.kind = kind
        self.logit_scale = torch.nn.Parameter(torch.tensor(math.log(1 / temperature)))
        self.logit_bias = torch.nn.Parameter(torch.tensor(float(sigmoid_bias))) if kind == 'siglip_sigmoid' else None

    def forward(self, images, texts, positives):
        logits = F.normalize(texts, dim=-1) @ F.normalize(images, dim=-1).T * self.logit_scale.exp()
        if self.logit_bias is not None:
            return sigmoid_loss(logits + self.logit_bias, positives)
        return symmetric_infonce(logits, positives)

    def constrain(self):
        with torch.no_grad():
            self.logit_scale.clamp_(0., math.log(100.))
            if self.logit_bias is not None:
                self.logit_bias.clamp_(-30., 30.)

    def scalars(self):
        return dict(logit_scale=float(self.logit_scale.exp().detach()),
                    temperature=float(self.logit_scale.neg().exp().detach()),
                    logit_bias=float(self.logit_bias.detach()) if self.logit_bias is not None else None)
