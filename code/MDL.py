
import torch
import torch.nn as nn
import torch.nn.functional as F


class MatryoshkaClassificationLoss(nn.Module):
    def __init__(self, dims=(96, 192, 384, 768)):
        super().__init__()
        self.dims = dims

    def forward(
        self,
        image_features,  # [B, D]
        text_features,   # [C, D]
        labels,          # [B]
        logit_scale,     # positive scalar
    ):
        assert image_features.shape[-1] == text_features.shape[-1]

        full_dim = image_features.shape[-1]
        assert max(self.dims) <= full_dim

        losses = []
        logits_by_dim = {}

        for dim in self.dims:
            # Slice BEFORE normalization
            image = image_features[:, :dim]
            text = text_features[:, :dim]

            image = F.normalize(image, dim=-1)
            text = F.normalize(text, dim=-1)

            # B x C classification logits
            if text.dim() == 3:   # [B, C, D] image-conditioned text (after CMT)
                logits = logit_scale * torch.einsum("bd,bcd->bc", image, text)
            else:                 # [C, D]
                logits = logit_scale * (image @ text.T)

            loss = F.cross_entropy(logits, labels)

            losses.append(loss)
            logits_by_dim[dim] = logits

        total_loss = torch.stack(losses).mean()

        return total_loss, logits_by_dim
