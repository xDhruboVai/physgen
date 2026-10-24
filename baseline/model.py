"""Trainable temporal head over frozen backbone features (doc section 10)."""
import torch
import torch.nn as nn


class TemporalHead(nn.Module):
    """Per-slot logits over {0 normal, 1 shadow/lighting, 2 mechanics, 3 causality}."""

    def __init__(self, d_in, slots, d=256, layers=2, heads=4, dropout=0.1, n_classes=4):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d))
        self.pos = nn.Parameter(torch.zeros(1, slots, d))
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.out = nn.Linear(d, n_classes)

    def forward(self, x):
        return self.out(self.encoder(self.inp(x) + self.pos))


def clip_scores(logits):
    """Clip violation score (max over slots of log-odds of 'any violation') and per-slot violation prob."""
    lp = logits.log_softmax(-1)
    viol = torch.logsumexp(lp[..., 1:], -1) - lp[..., 0]  # log-odds of violation per slot
    return viol.max(-1).values, viol.sigmoid()
