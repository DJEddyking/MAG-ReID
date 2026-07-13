import torch
import torch.nn as nn
import torch.nn.functional as F

def pairwise_euclidean_dist(x, y=None):
    """Compute pairwise euclidean distance between rows of x and y.
       If y is None, compute between x and x.
       Returns (B_x, B_y) matrix.
    """
    if y is None:
        y = x
    xx = (x**2).sum(dim=1, keepdim=True)  # (B_x,1)
    yy = (y**2).sum(dim=1, keepdim=True)  # (B_y,1)
    dist = xx + yy.t() - 2.0 * torch.matmul(x, y.t())
    dist = torch.clamp(dist, min=0.0)
    return dist

def hard_example_mining_cross_modal(dist_mat, labels, modalities):
    """
    Cross-modal hard mining.
    dist_mat: (B,B) pairwise distances (anchor i vs sample j)
    labels: (B,) long tensor of IDs
    modalities: (B,) long tensor of modality ids (e.g., 0,1,2)
    Returns:
      dist_ap: (B,) hardest positive distance for each anchor
      dist_an: (B,) hardest negative distance for each anchor
    Behavior:
      - For anchor i with modality m:
        positives considered: samples j with labels[j]==labels[i] and modalities[j] != m
        negatives considered: samples j with labels[j]!=labels[i] and modalities[j] != m
      - If no positives in other modalities, fallback to positives with same label (any modality except self).
      - If no negatives in other modalities, fallback to negatives across all modalities.
    """
    device = dist_mat.device
    B = dist_mat.size(0)
    labels = labels.view(-1)
    modalities = modalities.view(-1)

    # masks
    mask_self = torch.eye(B, dtype=torch.bool, device=device)
    same_label = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B,B)  ## 找到哪些是相同id和不同id
    diff_label = ~same_label

    same_mod = modalities.unsqueeze(0) == modalities.unsqueeze(1)   ## 找到哪些是相同模态和不同模态
    diff_mod = ~same_mod

    # positives: same label & different modality
    pos_cross_mask = same_label & diff_mod & (~mask_self)  ### 去掉相同模态的id, 以及当前id其本身
    # fallback positives: same label & not self (any modality)
    pos_any_mask = same_label & (~mask_self)

    # negatives: different label & different modality
    neg_cross_mask = diff_label & diff_mod
    # fallback negatives: different label (any modality)
    neg_any_mask = diff_label

    # For each anchor i, select positives and negatives
    dist_ap = torch.zeros(B, device=device)
    dist_an = torch.zeros(B, device=device)

    # For positives: we want the hardest positive -> largest distance among positives
    # For negatives: we want the hardest negative -> smallest distance among negatives
    # We'll compute masked distances and then reduce with appropriate ops.

    # masked positive distances (set non-positives to -inf so max picks valid)
    neg_inf = -1e9
    pos_mask = pos_cross_mask.clone()
    pos_dist_masked = dist_mat.clone()
    pos_dist_masked[~pos_mask] = neg_inf
    pos_max_vals, _ = pos_dist_masked.max(dim=1)  # -inf if none

    # fallback where pos_max_vals == -inf -> use pos_any_mask
    no_cross_pos = (pos_max_vals <= neg_inf/2)
    if no_cross_pos.any():
        pos_dist_masked_any = dist_mat.clone()
        pos_dist_masked_any[~pos_any_mask] = neg_inf
        pos_max_any, _ = pos_dist_masked_any.max(dim=1)
        pos_max_vals[no_cross_pos] = pos_max_any[no_cross_pos]

    # If still -inf (i.e., no positive at all, e.g., unique ID in batch), set to 0 and mark invalid later
    pos_max_vals[pos_max_vals <= neg_inf/2] = 0.0
    dist_ap = pos_max_vals

    # masked negative distances (set non-negatives to +inf so min picks valid)
    pos_inf = 1e9
    neg_mask = neg_cross_mask.clone()
    neg_dist_masked = dist_mat.clone()
    neg_dist_masked[~neg_mask] = pos_inf
    neg_min_vals, _ = neg_dist_masked.min(dim=1)  # +inf if none

    # fallback where neg_min_vals == +inf -> use neg_any_mask
    no_cross_neg = (neg_min_vals >= pos_inf/2)
    if no_cross_neg.any():
        neg_dist_masked_any = dist_mat.clone()
        neg_dist_masked_any[~neg_any_mask] = pos_inf
        neg_min_any, _ = neg_dist_masked_any.min(dim=1)
        neg_min_vals[no_cross_neg] = neg_min_any[no_cross_neg]

    # If still +inf (unlikely: batch contains only same-label samples), set to large value
    neg_min_vals[neg_min_vals >= pos_inf/2] = dist_mat.max().item() if dist_mat.numel() > 0 else 1.0
    dist_an = neg_min_vals

    return dist_ap, dist_an

class CrossModalTripletLoss(object):
    """
    Cross-modal triplet loss with harder example mining.
    When mining, positives/negatives are preferentially chosen from other modalities.
    """

    def __init__(self, margin=None, hard_factor=0.0):
        self.margin = margin
        self.hard_factor = hard_factor
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, labels, modalities, normalize_feature=False):
        """
        global_feat: (B, D) tensor of features (can be mixed modalities)
        labels: (B,) long tensor of identity labels
        modalities: (B,) long tensor of modality ids (e.g., 0 for rgb, 1 for ir, ...)
        normalize_feature: whether to L2 normalize features before distance
        Returns: loss, dist_ap, dist_an
        """
        if normalize_feature:
            global_feat = F.normalize(global_feat, p=2, dim=1)

        # pairwise euclidean distance
        dist_mat = pairwise_euclidean_dist(global_feat)  # (B,B)

        # cross-modal hard mining
        dist_ap, dist_an = hard_example_mining_cross_modal(dist_mat, labels, modalities)

        # apply hard_factor scaling
        dist_ap = dist_ap * (1.0 + self.hard_factor)
        dist_an = dist_an * (1.0 - self.hard_factor)

        # prepare target for ranking loss
        y = dist_an.new_ones(dist_an.size())

        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)

        return loss, dist_ap, dist_an
