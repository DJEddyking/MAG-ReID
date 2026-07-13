import torch
import torch.nn as nn
import torch.nn.functional as F

def pairwise_euclidean_dist(x, y=None):
    if y is None:
        y = x
    xx = (x**2).sum(dim=1, keepdim=True)
    yy = (y**2).sum(dim=1, keepdim=True)
    dist = xx + yy.t() - 2.0 * torch.matmul(x, y.t())
    return torch.clamp(dist, min=0.0)

def cross_modal_hard_mining_for_sum(dist_mat, labels, modalities):
    """
    返回 per-anchor:
      dist_ap_same: hardest positive distance among same-modality positives (or -1 if none)
      dist_ap_cross: hardest positive distance among cross-modality positives (or -1 if none)
      dist_an: hardest negative distance (smallest distance among negatives; fallback to any negative)
      valid_mask: whether anchor has at least one positive (same or cross)
    """
    device = dist_mat.device
    B = dist_mat.size(0)
    labels = labels.view(-1)
    modalities = modalities.view(-1)

    mask_self = torch.eye(B, dtype=torch.bool, device=device)
    same_label = labels.unsqueeze(0) == labels.unsqueeze(1)
    diff_label = ~same_label
    same_mod = modalities.unsqueeze(0) == modalities.unsqueeze(1)
    diff_mod = ~same_mod

    # masks
    pos_same_mask = same_label & same_mod & (~mask_self)
    pos_cross_mask = same_label & diff_mod & (~mask_self)
    pos_any_mask = same_label & (~mask_self)
    neg_cross_mask = diff_label & diff_mod
    neg_any_mask = diff_label

    neg_inf = -1e9
    pos_inf = 1e9

    # hardest positive same-modality: max distance among pos_same_mask
    pos_same_dist = dist_mat.clone()
    pos_same_dist[~pos_same_mask] = neg_inf
    pos_same_max, _ = pos_same_dist.max(dim=1)  # -inf if none

    # fallback: if none, set to neg_inf sentinel
    has_pos_same = pos_same_max > neg_inf/2
    pos_same_max[~has_pos_same] = -1.0  # sentinel for "no same positive"

    # hardest positive cross-modality
    pos_cross_dist = dist_mat.clone()
    pos_cross_dist[~pos_cross_mask] = neg_inf
    pos_cross_max, _ = pos_cross_dist.max(dim=1)
    has_pos_cross = pos_cross_max > neg_inf/2
    pos_cross_max[~has_pos_cross] = -1.0  # sentinel

    # fallback positives: if cross missing but any positive exists, we can use any positive
    # but we keep semantics: we prefer same and cross separately; if one missing we still may use any positive
    # compute any positive max (for fallback)
    pos_any_dist = dist_mat.clone()
    pos_any_dist[~pos_any_mask] = neg_inf
    pos_any_max, _ = pos_any_dist.max(dim=1)
    has_pos_any = pos_any_max > neg_inf/2

    # if same missing but any exists, set pos_same to pos_any (optional behavior)
    # here we keep separate: only use fallback for sum if one of same/cross missing but any exists.
    # compute hardest negative (prefer cross-modality negatives)
    neg_cross_dist = dist_mat.clone()
    neg_cross_dist[~neg_cross_mask] = pos_inf
    neg_cross_min, _ = neg_cross_dist.min(dim=1)
    has_neg_cross = neg_cross_min < pos_inf/2

    neg_any_dist = dist_mat.clone()
    neg_any_dist[~neg_any_mask] = pos_inf
    neg_any_min, _ = neg_any_dist.min(dim=1)
    has_neg_any = neg_any_min < pos_inf/2

    # choose neg: prefer cross negative, else any negative
    neg_min = neg_cross_min.clone()
    fallback_neg_idx = ~has_neg_cross
    if fallback_neg_idx.any():
        neg_min[fallback_neg_idx] = neg_any_min[fallback_neg_idx]

    # finalize pos values: keep -1 sentinel for missing
    # For anchors with no positives at all, mark invalid later
    has_any_pos = has_pos_any  # whether there is any positive (same or cross)
    # If same missing but cross exists, pos_same stays -1; vice versa.

    return pos_same_max, pos_cross_max, neg_min, has_any_pos

class SumConstraintCrossModalTripletLoss(nn.Module):
    """
    Triplet loss variant:
      - pull anchor to same-modality hardest positive (if exists)
      - pull anchor to cross-modality hardest positive (if exists)
      - enforce (d_ap_same + d_ap_cross) + margin < d_an  (if both positives exist)
    Loss per anchor is sum of:
      L_same = relu(d_ap_same - d_an + margin)   (if same exists)
      L_cross = relu(d_ap_cross - d_an + margin) (if cross exists)
      L_sum = relu((d_ap_same + d_ap_cross) - d_an + margin_sum) (if both exist)
    You can weight these three terms via weights.
    """
    def __init__(self, margin=0.3, margin_sum=0.5, w_same=1.0, w_cross=1.0, w_sum=1.0, normalize_feature=False):
        super().__init__()
        self.margin = margin
        self.margin_sum = margin_sum
        self.w_same = w_same
        self.w_cross = w_cross
        self.w_sum = w_sum
        self.normalize_feature = normalize_feature

    def forward(self, feats, labels, modalities):
        """
        feats: (B,D) features (mixed modalities)
        labels: (B,) long
        modalities: (B,) long
        returns: scalar loss, dict of diagnostics
        """
        device = feats.device
        B = feats.size(0)
        if self.normalize_feature:
            feats = F.normalize(feats, p=2, dim=1)

        dist_mat = pairwise_euclidean_dist(feats)  # (B,B)

        pos_same, pos_cross, neg_min, has_any_pos = cross_modal_hard_mining_for_sum(dist_mat, labels, modalities)

        # prepare masks for existence
        has_same = pos_same >= 0.0
        has_cross = pos_cross >= 0.0
        has_both = has_same & has_cross
        valid_anchor = has_any_pos  # at least one positive exists

        # compute per-anchor losses
        # L_same = relu(d_same - d_neg + margin)
        L_same = torch.zeros(B, device=device)
        if has_same.any():
            d_same = pos_same.clone()
            d_same[~has_same] = 0.0
            L_same = F.relu(d_same - neg_min + self.margin) * has_same.float()

        # L_cross = relu(d_cross - d_neg + margin)
        L_cross = torch.zeros(B, device=device)
        if has_cross.any():
            d_cross = pos_cross.clone()
            d_cross[~has_cross] = 0.0
            L_cross = F.relu(d_cross - neg_min + self.margin) * has_cross.float()

        # L_sum = relu((d_same + d_cross) - d_neg + margin_sum) only when both exist
        L_sum = torch.zeros(B, device=device)
        if has_both.any():
            d_sum = pos_same + pos_cross
            L_sum = F.relu(d_sum - neg_min + self.margin_sum) * has_both.float()

        # weighted sum per anchor
        loss_per_anchor = self.w_same * L_same + self.w_cross * L_cross + self.w_sum * L_sum

        # average over valid anchors
        valid_count = valid_anchor.sum().clamp(min=1.0)
        loss = loss_per_anchor.sum() / valid_count

        # diagnostics
        stats = {
            'loss_same_mean': (L_same.sum() / valid_count).item(),
            'loss_cross_mean': (L_cross.sum() / valid_count).item(),
            'loss_sum_mean': (L_sum.sum() / valid_count).item(),
            'valid_anchors': int(valid_count.item())
        }
        return loss, stats
