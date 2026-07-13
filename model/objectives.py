import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_tcmpm(image_fetures, text_fetures, pid,  logit_scale, image_id=None, factor=0.3, epsilon=1e-8):
    """
    Cross Modal Projection Matching 
    t2i_proj = ||t|| * cos(theta)
    i2j_proj = ||v|| * cos(theta)
    """
    batch_size = image_fetures.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()

    if image_id != None:
        # print("Mix PID and ImageID to create soft label.")
        image_id = image_id.reshape((-1, 1))
        image_id_dist = image_id - image_id.t()
        image_id_mask = (image_id_dist == 0).float()
        labels = (labels - image_id_mask) * factor + image_id_mask
        # labels = (labels + image_id_mask) / 2

    image_norm = image_fetures / image_fetures.norm(dim=1, keepdim=True)
    text_norm = text_fetures / text_fetures.norm(dim=1, keepdim=True)

    image_proj_text = logit_scale * torch.matmul(image_fetures, text_norm.t())
    text_proj_image = logit_scale * torch.matmul(text_fetures, image_norm.t())


    # normalize the true matching distribution
    labels_distribute = labels / labels.sum(dim=1) # original paper use sum, and use norm will lead minus loss
    # labels_distribute = F.softmax((labels * logit_scale), dim=1)
    # labels_distribute = F.softmax(labels, dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))

    # i2t2t2i_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - F.log_softmax(text_proj_image, dim=1))

    # loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1)) + torch.mean(torch.sum(i2t2t2i_loss, dim=1))
    loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return loss


def compute_sdm_teacher(image_features, text_features, logit_scale):
    """
    Similarity Distribution Matching: 返回teacher model计算获得的i2t, t2i的相似度, 用这个作为gt, 和student计算
    """
    ########## 去掉为0的特征不计算
    # _mask = (text_features.sum(dim=1) != 0)
    # image_features = image_features[_mask]
    # text_features = text_features[_mask]
    # pid = pid[_mask]
    nonzero_mask_A = (image_features.abs().sum(dim=1) > 1e-12)
    nonzero_mask_B = (text_features.abs().sum(dim=1) > 1e-12)

    # 2. 同时非零的行
    _mask = nonzero_mask_A & nonzero_mask_B
    if _mask.sum().item() == 0:
        return torch.zeros([], dtype=_mask.dtype, device=_mask.device)
    image_features = image_features[_mask]
    text_features = text_features[_mask]
    # pid = pid[_mask]
    ############### 

    image_norm = image_features / image_features.norm(dim=1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=1, keepdim=True)

    t2i_cosine_theta = text_norm @ image_norm.t()
    i2t_cosine_theta = t2i_cosine_theta.t()

    text_proj_image = logit_scale * t2i_cosine_theta
    image_proj_text = logit_scale * i2t_cosine_theta

    return image_proj_text.softmax(dim=1), text_proj_image.softmax(dim=1)


def compute_sdm(image_features, text_features, pid, logit_scale, image_id=None, factor=0.3, epsilon=1e-8, teacher_i2t=None, teacher_t2i=None):
    """
    Similarity Distribution Matching
    """
    ########## 去掉为0的特征不计算
    # _mask = (text_features.sum(dim=1) != 0)
    # image_features = image_features[_mask]
    # text_features = text_features[_mask]
    # pid = pid[_mask]
    nonzero_mask_A = (image_features.abs().sum(dim=1) > 1e-12)
    nonzero_mask_B = (text_features.abs().sum(dim=1) > 1e-12)

    # 2. 同时非零的行
    _mask = nonzero_mask_A & nonzero_mask_B
    if _mask.sum().item() == 0:
        return torch.zeros([], dtype=_mask.dtype, device=_mask.device)
    image_features = image_features[_mask]
    text_features = text_features[_mask]
    pid = pid[_mask]
    ############### 

    batch_size = image_features.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()

    if image_id != None:
        # print("Mix PID and ImageID to create soft label.")
        image_id = image_id.reshape((-1, 1))
        image_id_dist = image_id - image_id.t()
        image_id_mask = (image_id_dist == 0).float()
        labels = (labels - image_id_mask) * factor + image_id_mask
        # labels = (labels + image_id_mask) / 2

    image_norm = image_features / image_features.norm(dim=1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=1, keepdim=True)

    t2i_cosine_theta = text_norm @ image_norm.t()
    i2t_cosine_theta = t2i_cosine_theta.t()

    # text_proj_image = logit_scale * (text_norm_value * t2i_cosine_theta)
    # image_proj_text = logit_scale * (image_norm_value * i2t_cosine_theta)

    # mean_norm_value = (text_norm_value + image_norm_value) / 2
    # text_proj_image = logit_scale * (mean_norm_value * t2i_cosine_theta)
    # image_proj_text = logit_scale * (mean_norm_value * i2t_cosine_theta)

    # k_value = 8
    # text_proj_image = logit_scale * (k_value * t2i_cosine_theta)
    # image_proj_text = logit_scale * (k_value * i2t_cosine_theta)

    text_proj_image = logit_scale * t2i_cosine_theta
    image_proj_text = logit_scale * i2t_cosine_theta

    # normalize the true matching distribution
    labels_distribute = labels / labels.sum(dim=1) # original paper use sum, and use norm will lead minus loss
    # labels_distribute = F.softmax((labels * logit_scale), dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))

    loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    if teacher_i2t is not None and teacher_t2i is not None:
        i2t_loss_t = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(teacher_i2t + epsilon))
        t2i_loss_t = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(teacher_t2i + epsilon))
        loss += torch.mean(torch.sum(i2t_loss_t, dim=1)) + torch.mean(torch.sum(t2i_loss_t, dim=1))

    return loss

def compute_mcm_or_mlm(scores, labels):
    ce = nn.CrossEntropyLoss(ignore_index=0)
    return ce(scores, labels)


def compute_itc(image_features, text_features, logit_scale):
    """
    image-text contrastive (ITC) loss, InfoNCE
    """
    batch_size = image_features.shape[0]
    labels = torch.arange(start=0, end=batch_size, dtype=torch.int64)
    labels = labels.to(image_features.device)

    # normalized features
    image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_image = logit_scale * image_norm @ text_norm.t()
    logits_per_text = logits_per_image.t()

    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t =F.cross_entropy(logits_per_text, labels)
    loss = (loss_i +  loss_t)/2

    return loss


def multi_positive_itc_get_logits(image_features, text_features, logit_scale):
    """
    """
    # normalize
    image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=-1, keepdim=True)

    # similarity matrix (N, N)
    logits_per_image = logit_scale * image_norm @ text_norm.t()
    logits_per_text = logits_per_image.t()

    return logits_per_image, logits_per_text



def multi_positive_itc(image_features, text_features, pid, logit_scale=1.0):
    """
    多正样本版 image-text contrastive (ITC) loss, InfoNCE
    支持一个 batch 内同 pid 的多个正样本
    """
    ########## 去掉为0的特征不计算
    # _mask = (text_features.sum(dim=1) != 0)
    # image_features = image_features[_mask]
    # text_features = text_features[_mask]
    # pid = pid[_mask]

    # 判断每一行是否全为 0
    nonzero_mask_A = (image_features.abs().sum(dim=1) > 1e-12)
    nonzero_mask_B = (text_features.abs().sum(dim=1) > 1e-12)

    # 2. 同时非零的行
    _mask = nonzero_mask_A & nonzero_mask_B
    if _mask.sum().item() == 0:
        return torch.zeros([], dtype=_mask.dtype, device=_mask.device)
    image_features = image_features[_mask]
    text_features = text_features[_mask]
    pid = pid[_mask]
    ############### 

    # label mask: 1 if same pid else 0
    labels = (pid.view(-1,1) == pid.view(1,-1)).float().to(image_features.device)

    # normalize
    image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=-1, keepdim=True)

    # similarity matrix (N, N)
    logits_per_image = logit_scale * image_norm @ text_norm.t()
    logits_per_text = logits_per_image.t()


    # exp logits
    exp_logits_img = torch.exp(logits_per_image)
    exp_logits_txt = torch.exp(logits_per_text)

    # numerator: sum over positives
    numerator_img = (exp_logits_img * labels).sum(dim=1)
    numerator_txt = (exp_logits_txt * labels).sum(dim=1)

    # denominator: sum over all
    denominator_img = exp_logits_img.sum(dim=1)
    denominator_txt = exp_logits_txt.sum(dim=1)

    # avoid divide by zero
    loss_img = -torch.log(numerator_img / denominator_img + 1e-8)
    loss_txt = -torch.log(numerator_txt / denominator_txt + 1e-8)

    # average
    loss = (loss_img.mean() + loss_txt.mean()) / 2
    return loss



class CMFL(nn.Module):
    """
    Cross Modal Focal Loss
    """

    def __init__(self, alpha=1, gamma=2, binary=False, multiplier=2, sg=False):
        super(CMFL, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.binary = binary
        self.multiplier = multiplier
        self.sg = sg

    def forward(self, inputs_a, inputs_b, targets):

        # bce_loss_a = F.binary_cross_entropy(inputs_a, targets, reduce=False)
        # bce_loss_b = F.binary_cross_entropy(inputs_b, targets, reduce=False)

        bce_loss_a = F.cross_entropy(inputs_a, targets, reduce=False)
        bce_loss_b = F.cross_entropy(inputs_b, targets, reduce=False)

        pt_a = torch.exp(-bce_loss_a)
        pt_b = torch.exp(-bce_loss_b)

        eps = 0.000000001

        if self.sg:
            d_pt_a = pt_a.detach()
            d_pt_b = pt_b.detach()
            wt_a = ((d_pt_b + eps) * (self.multiplier * pt_a * d_pt_b)) / (pt_a + d_pt_b + eps)
            wt_b = ((d_pt_a + eps) * (self.multiplier * d_pt_a * pt_b)) / (d_pt_a + pt_b + eps)
        else:
            wt_a = ((pt_b + eps) * (self.multiplier * pt_a * pt_b)) / (pt_a + pt_b + eps)
            wt_b = ((pt_a + eps) * (self.multiplier * pt_a * pt_b)) / (pt_a + pt_b + eps)

        if self.binary:
            wt_a = wt_a * (1 - targets)
            wt_b = wt_b * (1 - targets)

        f_loss_a = self.alpha * (1 - wt_a) ** self.gamma * bce_loss_a
        f_loss_b = self.alpha * (1 - wt_b) ** self.gamma * bce_loss_b

        loss = 0.5 * torch.mean(f_loss_a) + 0.5 * torch.mean(f_loss_b)

        return loss

def focal_loss_two(inputs_a, inputs_b, alpha, gamma):

    pt_a = torch.exp(-inputs_a)
    pt_b = torch.exp(-inputs_b)

    eps = 0.000000001


    wt_a = ((pt_b + eps) * (2 * pt_a * pt_b)) / (pt_a + pt_b + eps)
    wt_b = ((pt_a + eps) * (2 * pt_a * pt_b)) / (pt_a + pt_b + eps)


    f_loss_a = alpha * (1 + wt_a) ** gamma * inputs_a
    f_loss_b = alpha * (1 + wt_b) ** gamma * inputs_b

    loss = torch.mean(f_loss_a) + torch.mean(f_loss_b)

    return loss


def compute_itc_focal3(image_features, text_features, simage_features,  cimage_features, nimage_features,
                       fusion_features, logit_scale, alpha, gamma, klp,
                       image_features_tse=None, text_features_tse=None, simage_features_tse=None, 
                        cimage_features_tse=None, nimage_features_tse=None, fusion_features_tse=None):
    """
    image-text contrastive (ITC) loss, InfoNCE

    增加两个feature, 同时增加对应的loss

    label有问题 
    """
    batch_size = image_features.shape[0]
    labels = torch.arange(start=0, end=batch_size, dtype=torch.int64)
    labels = labels.to(image_features.device)

    
    # normalized features
    image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    # fusion_norm = fusion_features / fusion_features.norm(dim=-1, keepdim=True)

    #### tse
    # image_norm_tse = image_features_tse / image_features_tse.norm(dim=-1, keepdim=True)
    # fusion_norm_tse = fusion_features_tse / fusion_features_tse.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    # rgb and text
    if text_features is not None:
        text_norm = text_features / text_features.norm(dim=-1, keepdim=True)
        logits_per_image0 = logit_scale * image_norm @ text_norm.t()
        logits_per_text0 = logits_per_image0.t()

        loss_i_0 = F.cross_entropy(logits_per_image0, labels)#, reduce=False)
        loss_t_0 =F.cross_entropy(logits_per_text0, labels)#, reduce=False)
        loss_it_0 = (loss_i_0 +  loss_t_0)/2

        ### tse
        # text_norm_tse = text_features_tse / text_features_tse.norm(dim=-1, keepdim=True)
        # logits_per_image0_tse = logit_scale * image_norm_tse @ text_norm_tse.t()
        # logits_per_text0_tse = logits_per_image0_tse.t()
        # loss_i_0_tse = F.cross_entropy(logits_per_image0_tse, labels)#, reduce=False)
        # loss_t_0_tse =F.cross_entropy(logits_per_text0_tse, labels)#, reduce=False)
        # loss_it_0_tse = (loss_i_0_tse +  loss_t_0_tse)/2

    # cosine similarity as logits
    # rgb and sketch
    if simage_features is not None:
        simage_norm = simage_features / simage_features.norm(dim=-1, keepdim=True)
        logits_per_image1 = logit_scale * image_norm @ simage_norm.t()
        logits_per_text1 = logits_per_image1.t()

        loss_i_1 = F.cross_entropy(logits_per_image1, labels)#, reduce=False)
        loss_t_1 =F.cross_entropy(logits_per_text1, labels)#, reduce=False)
        loss_is = (loss_i_1 +  loss_t_1)/2

        ### tse
        # simage_norm_tse = simage_features_tse / simage_features_tse.norm(dim=-1, keepdim=True)
        # logits_per_image1_tse = logit_scale * image_norm_tse @ simage_norm_tse.t()
        # logits_per_text1_tse = logits_per_image1_tse.t()
        # loss_i_1_tse = F.cross_entropy(logits_per_image1_tse, labels)#, reduce=False)
        # loss_t_1_tse =F.cross_entropy(logits_per_text1_tse, labels)#, reduce=False)
        # loss_is_tse = (loss_i_1_tse +  loss_t_1_tse)/2

    # cosine similarity as logits
    # rgb and fusion feature (4种images相加)
    if fusion_features is not None:
        fusion_norm = fusion_features / fusion_features.norm(dim=-1, keepdim=True)
        logits_per_image = logit_scale * image_norm @ fusion_norm.t()
        logits_per_text = logits_per_image.t()

        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t =F.cross_entropy(logits_per_text, labels)
        loss_if = (loss_i +  loss_t)/2

    #### tse
    # logits_per_image_tse = logit_scale * image_norm_tse @ fusion_norm_tse.t()
    # logits_per_text_tse = logits_per_image_tse.t()
    # loss_i_tse = F.cross_entropy(logits_per_image_tse, labels)
    # loss_t_tse =F.cross_entropy(logits_per_text_tse, labels)
    # loss_if_tse = (loss_i_tse +  loss_t_tse)/2

    # rgb and color pencil
    if cimage_features is not None:
        cimage_norm = cimage_features / cimage_features.norm(dim=-1, keepdim=True)
        logits_per_image2 = logit_scale * image_norm @ cimage_norm.t()
        logits_per_text2 = logits_per_image2.t()

        loss_i2 = F.cross_entropy(logits_per_image2, labels)#, reduce=False)
        loss_t2 =F.cross_entropy(logits_per_text2, labels)#, reduce=False)
        loss_ic = (loss_i2 +  loss_t2)/2

        ##### tse
        # cimage_norm_tse = cimage_features_tse / cimage_features_tse.norm(dim=-1, keepdim=True)
        # logits_per_image2_tse = logit_scale * image_norm_tse @ cimage_norm_tse.t()
        # logits_per_text2_tse = logits_per_image2_tse.t()
        # loss_i2_tse = F.cross_entropy(logits_per_image2_tse, labels)#, reduce=False)
        # loss_t2_tse =F.cross_entropy(logits_per_text2_tse, labels)#, reduce=False)
        # loss_ic_tse = (loss_i2_tse +  loss_t2_tse)/2


    # rgb and nir 
    if nimage_features is not None:
        nimage_norm = nimage_features / nimage_features.norm(dim=-1, keepdim=True)
        logits_per_image3 = logit_scale * image_norm @ nimage_norm.t()
        logits_per_text3 = logits_per_image3.t()

        loss_i3 = F.cross_entropy(logits_per_image3, labels)#, reduce=False)
        loss_t3 =F.cross_entropy(logits_per_text3, labels)#, reduce=False)
        loss_in = (loss_i3 +  loss_t3)/2


        #### tse
        # nimage_norm_tse = nimage_features_tse / nimage_features_tse.norm(dim=-1, keepdim=True)
        # logits_per_image3_tse = logit_scale * image_norm_tse @ nimage_norm_tse.t()
        # logits_per_text3_tse = logits_per_image3_tse.t()
        # loss_i3_tse = F.cross_entropy(logits_per_image3_tse, labels)#, reduce=False)
        # loss_t3_tse =F.cross_entropy(logits_per_text3_tse, labels)#, reduce=False)
        # loss_in_tse = (loss_i3_tse +  loss_t3_tse)/2

    # focal loss
    # kl = F.kl_div(logits_per_text1.softmax(dim=-1).log(), logits_per_text0.detach().softmax(dim=-1), reduction='sum') + F.kl_div(logits_per_text0.softmax(dim=-1).log(), logits_per_text1.detach().softmax(dim=-1), reduction='sum')

    # loss = focal_loss_two(loss_it, loss_is, alpha, gamma) + loss_if + klp*(CoRefineLoss(logits_per_text1, logits_per_text0.detach()))
 
    # 简单相加
    loss = torch.zeros(1)[0].to(image_features.device)
    if fusion_features is not None:
        loss += loss_if #+ loss_if_tse
    if text_features is not None:
        loss += loss_it_0 #+ loss_it_0_tse
    if simage_features is not None:
        loss += loss_is #+ loss_is_tse
    if cimage_features is not None:
        loss += loss_ic #+ loss_ic_tse
    if nimage_features is not None:
        loss += loss_in #+ loss_in_tse
    return loss


def compute_id_new(input_logits, labels, margin=0.35, scale=64.0):
    criterion = nn.CrossEntropyLoss(reduction="mean")

    def apply_margin(logits, labels, m, s):
        # 将目标类logit减去margin，再整体乘以scale
        one_hot = torch.zeros_like(logits).scatter_(1, labels.view(-1, 1), 1.0)
        # logits = logits / logits.norm(dim=1, keepdim=True)## 改为标准的arcface, cosface loss的操作, 直接归一化logits即可
        logits_m = logits - m * one_hot
        logits_s = s * logits_m
        return logits_s
    img_logits_s = apply_margin(input_logits, labels, margin, scale)
    loss = criterion(img_logits_s, labels) #+ criterion(text_logits, labels)  ## 12.22 torch1.10以后支持输入one hot的labels  ### TODO: 把加上Margin和scale之前的logits拿到, 与之后的logits对比
    
    return loss, img_logits_s


import math
class ArcFaceLoss(nn.Module):
    """
    ArcFace (additive angular margin) implementation following the original paper.
    This implementation uses the trig identity to compute cos(theta + m).
    Margin m and scale s are set to the requested values: m=0.35, s=64.
    Returns: (loss, logits, cosine)
    """
    def __init__(self, in_features, out_features, s=64.0, m=0.35):
        super(ArcFaceLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m

        # precompute cos(m) and sin(m)
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        # threshold and mm as used in many reference implementations (kept for optional easy margin)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

        # weight: shape (C, D)
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.normal_(self.weight, std=0.01)

    def forward(self, input_embeddings, labels):
        """
        input_embeddings: (N, D)
        labels: (N,) long tensor with values in [0, C-1]
        """
        # normalize features and weights
        norm_weight = F.normalize(self.weight, dim=1)        # (C, D)
        norm_embeddings = F.normalize(input_embeddings, dim=1)  # (N, D)

        # cosine similarity between embeddings and weights
        cosine = F.linear(norm_embeddings, norm_weight)  # (N, C)
        # numerical stability
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        # compute phi = cos(theta + m) using trig identity
        sin_theta = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sin_theta * self.sin_m

        # optional easy margin (commented out to match "original" behavior unless enabled)
        # phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # one-hot labels
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        # combine: for target classes use phi, otherwise use original cosine
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits = logits * self.s

        loss = F.cross_entropy(logits, labels, reduction='none')
        return loss, logits



class CosFaceLoss(nn.Module):
    """
    标准的cosine face loss: 权重和输入都进行了L2归一化;
    注意: 此时没有bias存在
    """
    def __init__(self, in_features, out_features, s=64.0, m=0.35):
        super(CosFaceLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        
        # 分类器权重，形状为 (num_classes, embedding_size)
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        # nn.init.xavier_uniform_(self.weight)
        nn.init.normal_(self.weight.data, std=0.001)
        # self.bias = nn.Parameter(torch.FloatTensor(out_features))
        # nn.init.constant_(self.bias.data, val=0.0)

    def forward(self, input_embeddings, labels):
        # 1. 对权重进行 L2 归一化: W / ||W||
        norm_weight = F.normalize(self.weight, dim=1)
        
        # 2. 对输入特征进行 L2 归一化: x / ||x||
        norm_embeddings = F.normalize(input_embeddings, dim=1)
        
        # 3. 计算余弦相似度 cos(theta) = x_norm * W_norm^T
        # logits 的范围在 [-1, 1]
        cosine = F.linear(norm_embeddings, norm_weight) #, self.bias)   #### 增加bias
        
        # 4. 加上 Margin: cos(theta) - m
        # 仅针对正确类别 (target class) 减去 m
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        
        # 核心公式: s * (cos(theta_yi) - m)
        output = self.s * (cosine - one_hot * self.m)  ## TODO: margin和scale变为课学习的 ??? 另外, 整个cosface loss是没有偏置的, ...
        
        # 5. 计算交叉熵损失
        loss = F.cross_entropy(output, labels)
        
        return loss, cosine



class MultiModalIDHead(nn.Module):
    def __init__(self, feat_dim, num_classes, margin=0.35, scale=64):
        super(MultiModalIDHead, self).__init__()
        
        # 1. 关键：针对不同模态的独立 BN 层
        # affine=True 允许学习 shift(beta) 和 scale(gamma)，这能自动拉齐分布
        self.img_bn = nn.BatchNorm1d(feat_dim)
        self.text_bn = nn.BatchNorm1d(feat_dim)
        
        # 2. 共享的分类权重 (无 Bias)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, feat_dim))
        # nn.init.xavier_uniform_(self.weight)
        nn.init.normal_(self.weight, std=0.001)
        
        # ArcFace/CosFace 参数
        self.m = margin
        self.s = scale
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, feats, labels, feat_type='img'):

        def apply_margin_loss(logits, targets):
            # 构建 one-hot
            one_hot = torch.zeros_like(logits)
            one_hot.scatter_(1, targets.view(-1, 1), 1.0)
            
            # 只有目标类减 margin
            logits_m = logits - (one_hot * self.m)
            
            # 缩放
            logits_scaled = logits_m * self.s
            
            # 计算 Loss
            return self.criterion(logits_scaled, targets)

        W_norm = F.normalize(self.weight, p=2, dim=1)
        # --- 步骤 1: 分别通过 BN (解决 Bias=False 效果差的问题) ---
        # 注意：这里假设 feats 形状为 (B, D)
        if feat_type == 'img':
            img_feats_bn = self.img_bn(feats)
            img_feats_norm = F.normalize(img_feats_bn, p=2, dim=1)
            img_logits = F.linear(img_feats_norm, W_norm)
            loss_img = apply_margin_loss(img_logits, labels)
            return loss_img, img_logits
        else:
            text_feats_bn = self.text_bn(feats)
            text_feats_norm = F.normalize(text_feats_bn, p=2, dim=1)
            text_logits = F.linear(text_feats_norm, W_norm)
            loss_text = apply_margin_loss(text_logits, labels)
            return loss_text, text_logits
        
        # --- 步骤 2: 强制 L2 归一化 (解决 Logits 尺度不可控问题) ---
        # --- 步骤 3: 计算 Cosine Logits ---
        # 此时已经是纯粹的角度余弦值了
        
        # --- 步骤 4: 应用 Margin (这里演示 CosFace/AM-Softmax 逻辑，更稳健) ---
        # CosFace: cos(theta) - m
        # 你的 compute_id_new 本质是 CosFace，这里将其正规化
        # 可以加权，通常 1:1
        # total_loss = loss_img + loss_text
        
        # return total_loss
        

def adaptive_margin_scale(logits, labels,
                          m0=0.3, s0=30.0,
                          g_target=0.5,
                          alpha=0.05, beta=1.0,
                          m_min=0.0, m_max=0.6,
                          s_min=8.0, s_max=64.0,
                          ema_decay=0.9,
                          state=None):
    """
    logits: [B, C] assumed cosine logits in [-1,1]
    labels: [B]
    state: dict to keep EMA of m and s across calls, e.g. {'m':m0,'s':s0}
    returns: logits_s after applying adaptive CosFace-style margin, updated state
    """
    # logits = logits / logits.norm(dim=1, keepdim=True)


    B, C = logits.shape
    if state is None:
        state = {'m': m0, 's': s0}

    # target logits and strongest negative per sample
    idx = torch.arange(B, device=logits.device)
    t = logits[idx, labels]                          # [B]
    # mask out true class to get max negative
    logits_masked = logits.clone()
    logits_masked[idx, labels] = -1e9
    neg = logits_masked.max(dim=1).values            # [B]

    # statistics
    gap = (t - neg).mean().item()                    # batch mean gap
    conf = F.softmax(logits * state['s'], dim=1).gather(1, labels.unsqueeze(1)).mean().item()

    # update rules (simple proportional controller)
    m_new = state['m'] + alpha * (g_target - gap)
    m_new = float(max(min(m_new, m_max), m_min))

    s_new = state['s'] * (1.0 + beta * (1.0 - conf))
    s_new = float(max(min(s_new, s_max), s_min))

    # EMA smoothing
    m_ema = ema_decay * state['m'] + (1 - ema_decay) * m_new
    s_ema = ema_decay * state['s'] + (1 - ema_decay) * s_new

    # apply CosFace-style additive margin on target logits
    logits_m = logits.clone()
    logits_m[idx, labels] = logits_m[idx, labels] - m_ema
    logits_s = logits_m * s_ema

    # update state
    state.update({'m': m_ema, 's': s_ema})
    return logits_s, state


def compute_id_no_scale(input_logits, labels):
    """
    Instance loss proposed at http://arxiv.org/abs/1711.05535
    """
    criterion = nn.CrossEntropyLoss(reduction="mean")
    loss = criterion(input_logits, labels)
    return loss


def contrastive_loss_batch(z, labels, tau=0.1, eps=1e-9):
    """
    以rgb作为anchor, 与相同id的其他模态或本模态样本计算相似度作为分子; 与其他所有样本除了其本身计算相似度作为分母; 
    所有正样本相加;
    z: 一个batch内把所有模态拼接在一起, bs * N 维度, N>=2
    """
    device = z.device
    B = z.size(0)

    z = F.normalize(z, p=2, dim=1)
    sim = torch.matmul(z, z.t()) / tau  # (B, B)

    mask_self = torch.eye(B, dtype=torch.bool, device=device)
    # 用 -inf 排除自身（logsumexp 能正确处理 -inf）
    neg_inf = -1e9  # 或 float('-inf') 但 -1e9 更兼容某些后端
    sim_masked = sim.masked_fill(mask_self, neg_inf)  # (B, B)

    # log denominator: logsumexp over a != i
    log_den = torch.logsumexp(sim_masked, dim=1)  # (B,)  # 分母: 去除对角线本身取对数求和

    # 正样本掩码（相同标签且不是自身）
    labels = labels.view(-1)
    eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B)
    pos_mask = eq & (~mask_self)  # (B, B)  ### 去掉本身的其他正样本
    pos_count = pos_mask.sum(dim=1).float()  # (B,)

    # 计算每个 (i,p) 的 log_prob = sim[i,p] - log_den[i]
    # 使用 sim_masked 保证对角已被排除（虽然 pos_mask 已排除 self，但统一用 sim_masked 更清晰）
    log_prob = sim_masked - log_den.unsqueeze(1)  # (B, B)  # 去掉本身后的相似度 减去 分母

    # 只保留正样本的 log_prob
    log_prob_pos = log_prob * pos_mask.float()  # (B, B)
    sum_log_prob_pos = log_prob_pos.sum(dim=1)  # (B,)  ### 所有正样本的对比loss相加

    # 对每个 i 求平均（跳过 pos_count==0）
    valid = pos_count > 0
    pos_count_safe = pos_count.clone()
    pos_count_safe[~valid] = 1.0

    loss_i = - sum_log_prob_pos / pos_count_safe  # (B,)  ## 除以第i个样本所有正样本的个数, 即取平均
    loss_i = loss_i * valid.float()  ## 对于没有正样本的情况做保护, (但实际上, 每个epoch必会选择至少两个模态数据)

    if valid.sum() > 0:
        loss = loss_i.sum() / valid.sum()
    else:
        loss = torch.tensor(0.0, device=device)

    return loss



class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs) 
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1) 
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss



def compute_id_with_margin(image_logits, text_logits, labels, margin=0.0, scale=30.0):
    """
    AM-Softmax 近似实现（在logit空间施加margin+scale）。
    注意：严格的CosFace/ArcFace需特征和权重归一化；此处采用近似形式以避免侵入式改动分类器结构。
    """
    if margin <= 0:
        return compute_id(image_logits, text_logits, labels)
    criterion = nn.CrossEntropyLoss(reduction="mean")

    def apply_margin(logits, labels, m, s):
        # 将目标类logit减去margin，再整体乘以scale
        one_hot = torch.zeros_like(logits).scatter_(1, labels.view(-1, 1), 1.0)
        # logits = logits / logits.norm(dim=1, keepdim=True)## 改为标准的arcface, cosface loss的操作, 直接归一化logits即可
        logits_m = logits - m * one_hot
        logits_s = s * logits_m
        return logits_s

    img_logits_s = apply_margin(image_logits, labels, margin, scale) # TODO 用原始的text feat和image feat拼接，然后分别计算它们各自的margin和scale?
    if text_logits is not None:
        txt_logits_s = apply_margin(text_logits, labels, margin, scale)
        return ( criterion(img_logits_s, labels) + criterion(txt_logits_s, labels) ) /2
    else:

        return criterion(img_logits_s, labels)
    


def compute_TAL_per(image_features, text_features, pid, tau=0.02, margin=0.2):

    # # normalized features
    image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=-1, keepdim=True)
    scores = text_norm @ image_norm.t()


    batch_size = scores.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float().cuda()
    mask = 1 - labels

    alpha_i2t =((scores/tau).exp()* labels / ((scores/tau).exp()* labels).sum(dim=1, keepdim=True)).detach()
    alpha_t2i = ((scores.t()/tau).exp()* labels / ((scores.t()/tau).exp()* labels).sum(dim=1, keepdim=True)).detach()

    loss = (-  (alpha_i2t*scores).sum(1) + tau * ((scores / tau).exp() * mask).sum(1).clamp(max=10e35).log() + margin).clamp(min=0)  \
        +  (-  (alpha_t2i*scores.t()).sum(1) + tau * ((scores.t() / tau).exp() * mask).sum(1).clamp(max=10e35).log() + margin).clamp(min=0)
    
    return loss 

def CoRefineLoss(output1, output2):

    # Target is ignored at training time. Loss is defined as KL divergence
    # between the model output and the refined labels.
    if output2.requires_grad:
        raise ValueError("Refined labels should not require gradients.")

    output1_log_prob = F.log_softmax(output1, dim=1)
    output2_prob = F.softmax(output2, dim=1)

    _, pred_label = output2_prob.max(1)

    # Loss is normal cross entropy loss
    # base_loss = F.cross_entropy(output1, pred_label)

    # Loss is -dot(model_output_log_prob, refined_labels). Prepare tensors
    # for batch matrix multiplicatio

    model_output1_log_prob = output1_log_prob.unsqueeze(2)
    model_output2_prob = output2_prob.unsqueeze(1)

    # Compute the loss, and average/sum for the batch.
    kl_loss = -torch.bmm(model_output2_prob, model_output1_log_prob)

    return kl_loss.mean()
        
def compute_id(classifier, image_embeddings, text_embeddings, labels, verbose=False):
    # print(labels.shape, labels.dtype)
    # print(image_embeddings.dtype)
    # classifier.weight.to(image_embeddings.dtype)
    labels_re = labels.reshape(-1, 1).to(image_embeddings.device)
    labels_one_hot = torch.zeros(image_embeddings.shape[0], classifier.weight.shape[0]).to(image_embeddings.device)
    labels_one_hot = labels_one_hot.scatter_(1, labels_re, 1).to(image_embeddings.device).float()

    image_logits = classifier(image_embeddings)
    # text_logits = classifier(text_embeddings)

    criterion = nn.CrossEntropyLoss(reduction="mean")

    if text_embeddings is None:
        loss = criterion(image_logits, labels_one_hot)
    else:
        text_logits = classifier(text_embeddings)
        loss = criterion(image_logits, labels_one_hot) + criterion(text_logits, labels_one_hot)
    # classification accuracy for observation
    if verbose:
        image_pred = torch.argmax(image_logits, dim=1)
        image_precision = torch.mean((image_pred == labels).float())

        if text_embeddings is None:
            return loss, image_precision, None
        
        text_pred = torch.argmax(text_logits, dim=1)
        text_precision = torch.mean((text_pred == labels).float())

        return loss, image_precision, text_precision
    
    return loss


def compute_cmpm(image_embeddings, text_embeddings, labels, epsilon=1e-8):
    """
    Cross-Modal Projection Matching Loss(CMPM)
    :param image_embeddings: Tensor with dtype torch.float32
    :param text_embeddings: Tensor with dtype torch.float32
    :param labels: Tensor with dtype torch.int32
    :return:
        i2t_loss: cmpm loss for image projected to text
        t2i_loss: cmpm loss for text projected to image
        pos_avg_sim: average cosine-similarity for positive pairs
        neg_avg_sim: averate cosine-similarity for negative pairs
    """

    batch_size = image_embeddings.shape[0]
    labels_reshape = torch.reshape(labels, (batch_size, 1))
    labels_dist = labels_reshape - labels_reshape.t()
    labels_mask = (labels_dist == 0).float()

    image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
    text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
    image_proj_text = torch.matmul(image_embeddings, text_norm.t())
    text_proj_image = torch.matmul(text_embeddings, image_norm.t())

    # normalize the true matching distribution
    labels_mask_norm = labels_mask / labels_mask.norm(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_mask_norm + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_mask_norm + epsilon))

    cmpm_loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return cmpm_loss


def compute_mcq(a, b, temperature=0.05, eps=1e-8):
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    x = torch.mm(a_norm, b_norm.transpose(0, 1))

    i_logsm = F.log_softmax(x/temperature, dim=1)
    j_logsm = F.log_softmax(x.t()/temperature, dim=1)

    # sum over positives
    idiag = torch.diag(i_logsm)
    loss_i = idiag.sum() / len(idiag)

    jdiag = torch.diag(j_logsm)
    loss_j = jdiag.sum() / len(jdiag)

    return - loss_i - loss_j


def CrossModalSupConLoss(image_fetures, text_fetures, labels, temperature):
    """
    Args:
        features: hidden vector of shape [bsz, n_views, ...].
        labels: ground truth of shape [bsz].
        mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
            has the same class as sample i. Can be asymmetric.
    Returns:
        A loss scalar.
    """
    device = (torch.device('cuda') if image_fetures.is_cuda else torch.device('cpu'))


    batch_size = image_fetures.shape[0]

    labels = labels.contiguous().view(-1, 1)
    if labels.shape[0] != batch_size:
        raise ValueError('Num of labels does not match num of features')
    mask = torch.eq(labels, labels.T).float().to(device)


    contrast_count = 2
    contrast_feature = torch.cat([image_fetures, text_fetures], dim=0)

    anchor_feature = contrast_feature
    anchor_count = contrast_count

    # compute logits
    anchor_dot_contrast = torch.matmul(anchor_feature, contrast_feature.T) * temperature
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # tile mask
    mask = mask.repeat(anchor_count, contrast_count)
    # mask-out self-contrast cases
    # logits_mask = torch.scatter(
    #     torch.ones_like(mask),
    #     1,
    #     torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
    #     0
    # )
    # mask = mask * logits_mask

    # compute log_prob
    # exp_logits = torch.exp(logits) * logits_mask
    exp_logits = torch.exp(logits)
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

    # loss
    loss = -mean_log_prob_pos
    loss = loss.view(anchor_count, batch_size).mean()

    return loss 


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, device="cuda"):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.device = device

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(
            1, targets.unsqueeze(1).data.cpu(), 1
        )
        if self.use_gpu:
            targets = targets.to(self.device)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss



# class SupConLoss(nn.Module):
#     """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
#     It also supports the unsupervised contrastive loss in SimCLR"""
#     def __init__(self, temperature=0.07, contrast_mode='all',
#                  base_temperature=0.07):
#         super(SupConLoss, self).__init__()
#         self.temperature = temperature
#         self.contrast_mode = contrast_mode
#         self.base_temperature = base_temperature

def SupConLoss(features, labels=None, mask=None, temperature=2.0, contrast_mode='all',
                 base_temperature=0.07):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (temperature / base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


