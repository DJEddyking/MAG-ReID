from model import objectives
from .clip_model_original import Transformer, QuickGELU, LayerNorm, \
    build_CLIP_from_openai_pretrained, convert_weights, TransformerOrignal, TransformerWithAdapter, InputImageType
# from .clip_model_original_only_LN import QuickGELU, LayerNorm, \
#     build_CLIP_from_openai_pretrained, convert_weights, TransformerOrignal, InputImageType

# from .clip_model_long_clip import Transformer, QuickGELU, LayerNorm, convert_weights, TransformerOrignal, TransformerWithAdapter, InputImageType

# from .clip_model_long_clip import build_CLIP_from_longclip_pretrained
# from .clip_model_original import build_CLIP_from_openai_pretrained

# from .denoising_diffusion_pytorch_1d import GaussianDiffusion1D_norm, Unet1D
# from .dit_model_from_unet1d import DiT1D
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import itertools
# from .CrossEmbeddingLayer_tse import VisualEmbeddingLayer, TexualEmbeddingLayer
from .CrossEmbeddingLayer_tse_lora import VisualEmbeddingLayer, TexualEmbeddingLayer
from .objectives import CrossEntropyLabelSmooth, MultiModalIDHead, CosFaceLoss, ArcFaceLoss
from loss import TripletLoss, SupConLoss, CrossModalTripletLoss, SumConstraintCrossModalTripletLoss
from utils.simple_tokenizer import SimpleTokenizer
from datasets.bases import tokenize

class FM_cat(nn.Module):
    def __init__(self,in_channels):
        super(FM_cat, self).__init__()

        self.W = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels,
                      kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(in_channels)
        )
        nn.init.normal_(self.W[1].weight.data, 1.0, 0.01)
        nn.init.zeros_(self.W[1].bias.data)


        # self.bottleneck = nn.BatchNorm1d(in_channels)
        # self.bottleneck.bias.requires_grad_(False)  # no shift

        # nn.init.normal_(self.bottleneck.weight.data, 1.0, 0.01)
        # nn.init.zeros_(self.bottleneck.bias.data)

    def forward(self,f):

        f = f.view(f.size(0),f.size(1),1,1)
        f = self.W(f)
        f = f.view(f.size(0),-1)
        # f = self.bottleneck(f+feat)

        return f


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

class TransformerConcat(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.ln = LayerNorm(input_dim)
        self.attention = nn.MultiheadAttention(input_dim, 8, batch_first=True) ### head数目暂定为固定(self.embed_dim * 2) // 64
        self.tf = TransformerOrignal(width=input_dim, layers=1, heads=8)## (self.embed_dim * 2) // 64
        self.fc1 = nn.Linear(input_dim, input_dim)  ## mlp也暂定固定
        self.relu = QuickGELU()
        self.fc2 = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        x = self.ln(x)
        x, _ = self.attention(x, x , x, need_weights=False)
        x = self.tf(x)
        x = self.fc2(self.relu(self.fc1(x)))
        x = x.mean(dim=1)
        return x



class Mlp(nn.Module):
    """用于将文本的eos token转为pseudo image token"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# 新增深度可分离卷积, 将image patch token转为 pseudo text token
class InverseImage(nn.Module):
    def __init__(self, input_channel, hidden_channel, out_channel, drop=0.):
        super().__init__()
        self.point_depth_wise_conv = nn.Sequential(
                nn.Conv2d(input_channel, hidden_channel, 1, 1, 0),
                nn.GELU(),
                nn.Conv2d(hidden_channel, hidden_channel, 2, 2, 0, groups=hidden_channel),
            )
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(hidden_channel, out_channel)  ### 768 -> 512 等价于CLIP里面的 proj 层
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.reshape(x.shape[0], 24, 8, x.shape[-1]).permute(0, 3, 1, 2)  # bs, c, h, w
        x = self.point_depth_wise_conv(x)
        x = self.avg_pool(x).view(x.size(0), -1)
        x = self.fc(x)
        x = self.drop(x)
        return x


class CLIP2ReID(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        # self.test_setting = args.test_setting

        self._set_task()

        ### 增加一个 args 入参
        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args, args.pretrain_choice, args.img_size, args.stride_size)
        # self.base_model, base_cfg = build_CLIP_from_longclip_pretrained(args, args.pretrain_choice, args.img_size, args.stride_size)
        
        self.embed_dim = base_cfg['embed_dim']

        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.02))  # 0.02 ==> for rgb
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / args.temperature))  # 0.07 ==> for sk
        # self.logit_scale_cp = nn.Parameter(torch.ones([]) * np.log(1 / args.temperature))  # 0.07 ==> for cp
        self.logit_scale_fuse = nn.Parameter(torch.ones([]) * np.log(1 / 0.02))  # 0.02 ==> for nir, 估计这个比较难, 需要hard一点的temp
        self.logit_scale_txt = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))  # 0.07 ==> for text
        self.logit_scale_sup_cons = nn.Parameter(torch.ones([]) * np.log(1 / 0.02))  # 0.07 ==> for text

        # self.tau_scale = nn.Parameter(torch.ones([]) * np.log(1/ 0.1))

        # 每个类都给一个可学习的Margin
        # self.raw_margin = nn.Parameter(torch.full((num_classes,), 0.35))
        # 每个模态都有一个可学习的scale
        # self.raw_scale = nn.Parameter(torch.log(torch.exp(torch.ones(5) * 64)))
        

        self.kl_div = nn.KLDivLoss(reduction='batchmean')

        self.original_logits_dict = {'rgb_logits': {}, 'sk_logits': {}, 'nir_logits': {}, 'cp_logits': {}, 'txt_logits': {}, 'labels': {}}
        self.after_logits_dict = {'rgb_logits': {}, 'sk_logits': {}, 'nir_logits': {}, 'cp_logits': {}, 'txt_logits': {}, 'labels': {}}
        self._feat_dict = {'rgb_features': {}, 'sk_features': {}, 'nir_features': {}, 'cp_features': {}, 'txt_features': {}, 'labels': {}}

        ## 增加一个adapter: 每个模态输入到这个adapter, 然后与rgb计算相似度loss
        # self.adapter_shared = nn.Sequential(
        #     nn.LayerNorm(512),
        #     nn.Linear(512, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 512)
        # )
        # self.register_buffer('mod_prototypes', torch.zeros(5, num_classes, 512))
        # # cross-modal prototypes: [num_ids, dim]
        # self.register_buffer('cross_prototypes', torch.zeros(num_classes, 512))
        # # init small random to avoid zeros
        # nn.init.normal_(self.mod_prototypes, mean=0.0, std=0.01)
        # nn.init.normal_(self.cross_prototypes, mean=0.0, std=0.01)

        self.teacher_base_model = copy.deepcopy(self.base_model)
        for param in self.teacher_base_model.parameters():
            param.requires_grad = False


        self.in_planes = 768
        self.in_planes_proj = 512
        # self.xent = CrossEntropyLabelSmooth(self.num_classes, epsilon=0.1, use_gpu=True, device="cuda:"+args.gpu_id)  ### TODO 不用smooth ？？？
        self.triplet = TripletLoss(margin=0.3)
        self.triplet_cross_modal = SumConstraintCrossModalTripletLoss(normalize_feature=True)#CrossModalTripletLoss(margin=0.3)
        # self.triplet_cross_modal_sum = SumConstraintCrossModalTripletLoss()
        # self.supcontrast_proj = SupConLoss(device="cuda:"+args.gpu_id, temperature=args.temperature)

        self.cls_head = ArcFaceLoss(self.embed_dim, num_classes, s=64, m=0.35)# CosFaceLoss(self.embed_dim, num_classes, s=30, m=0.1)#MultiModalIDHead(self.embed_dim, num_classes)  # 包含: 先对feat进行BN -> 分类头没有bias -> 加上logits margin/scale

        self.prompt_num = 2
        # self.inverseNet = Mlp(in_features=512, hidden_features=512 * 4, out_features=768, drop=0.1)
        # self.tokenizer = SimpleTokenizer()

        # self.rgb_caption_prefix = "Visible image of a" + " X X " + "person with natural colors: " # 4 个, 不加sos token; rgb不加多的提示词? TODO 
        # self.nir_caption_prefix = "Near-infrared image of a" + " X X " + "person with high reflectance contrast: " # 6 个
        # self.cp_caption_prefix = "Color-pencil drawing of a" + " X X " + "person with vivid colors: " # 6个
        # self.sk_caption_prefix = "Sketch image of a" + " X X " + "person with clean line contours: "  ## 4个
        self.text_length = self.args.text_length
        # 这个用于text 
        # self._transformer = Transformer(width=self.embed_dim, layers=args.cmt_depth, heads=self.embed_dim // 64)
        # self.cross_attn = nn.MultiheadAttention(self.embed_dim, self.embed_dim // 64, batch_first=True)
        # self.ln_pre_t, self.ln_pre_i,  self.ln_post = LayerNorm(self.embed_dim), LayerNorm(self.embed_dim), LayerNorm(self.embed_dim)
        # self._init_cross_attn_tf(self._transformer, self.cross_attn)
        # self._transformer_img_shared = TransformerOrignal(width=self.embed_dim, layers=args.cmt_depth, heads=self.embed_dim // 64)

        # self.mask_weight_sk, self.mask_weight_cp, self.mask_weight_nir = nn.Linear(75, 75), nn.Linear(75, 75), nn.Linear(75, 75)

        # 如果把image也inverse, 会陷入循环, 要么 ema, 要么冻结encoder后取pseudo token, 然后分别塞进text和image token, TODO
        # self.conv_inverse_net = InverseImage(input_channel=512, hidden_channel=512, out_channel=512, drop=0.1)

        # self.sk_learned_feat = nn.Parameter(torch.randn(77, 512).half().to(device = 'cuda:' + self.args.gpu_id)) # 可学习特征
        # self.cp_learned_feat = nn.Parameter(torch.randn(77, 512).half().to(device = 'cuda:' + self.args.gpu_id)) # 可学习特征
        # self.nir_learned_feat = nn.Parameter(torch.randn(77, 512).half().to(device = 'cuda:' + self.args.gpu_id)) # 可学习特征
        self.rgb_state, self.text_state, self.sk_state, self.cp_state, self.nir_state = None, None, None, None, None

        # self.token_counts = torch.zeros(77)

        # self._transformer_shared = TransformerOrignal(width=self.embed_dim,
        #                                                         layers=args.cmt_depth,
        #                                                         heads=self.embed_dim // 64)
        # self.rgb_emb_layer = VisualEmbeddingLayer(ratio=args.select_ratio).half()
        # self.texual_emb_layer = TexualEmbeddingLayer(ratio=args.select_ratio).half()
        # self.sk_emb_layer = VisualEmbeddingLayer(ratio=args.select_ratio).half()
        # self.cp_emb_layer = VisualEmbeddingLayer(ratio=args.select_ratio).half()
        # self.nir_emb_layer = VisualEmbeddingLayer(ratio=args.select_ratio).half()

        # 融合模态: 在length维度concat, 然后经过ln, msa, tf, mlp
        # self._ln_fuse = LayerNorm(self.embed_dim * 2)  # 1024: concat cls + patch
        # self._msa_fuse = nn.MultiheadAttention(self.embed_dim * 2, num_heads=8, batch_first=True)
        
        # self.visul_emb_layer = VisualEmbeddingLayer(ratio=args.select_ratio)
        # self.texual_emb_layer = TexualEmbeddingLayer(ratio=args.select_ratio)

        if 'id' in args.loss_names:
            self.classifier_proj = nn.Linear(self.embed_dim, self.num_classes)   ##### 分类 loss 配合bn neck
            nn.init.normal_(self.classifier_proj.weight.data, std=0.001)
            nn.init.constant_(self.classifier_proj.bias.data, val=0.0)

            # bias设置为5个模态独有的, 但是weight共享

            if self.args.use_multi_classifier:
                self.classifier_proj_sk = nn.Linear(self.embed_dim, self.num_classes) 
                nn.init.normal_(self.classifier_proj_sk.weight.data, std=0.001)
                # self.classifier_proj_sk.weight = self.classifier_proj.weight
                nn.init.constant_(self.classifier_proj_sk.bias.data, val=0.0)

                self.classifier_proj_cp = nn.Linear(self.embed_dim, self.num_classes)   ##### 分类 loss 配合bn neck
                nn.init.normal_(self.classifier_proj_cp.weight.data, std=0.001)
                # self.classifier_proj_cp.weight = self.classifier_proj.weight
                nn.init.constant_(self.classifier_proj_cp.bias.data, val=0.0)

                self.classifier_proj_nir = nn.Linear(self.embed_dim, self.num_classes)   ##### 分类 loss 配合bn neck
                nn.init.normal_(self.classifier_proj_nir.weight.data, std=0.001)
                # self.classifier_proj_nir.weight = self.classifier_proj.weight
                nn.init.constant_(self.classifier_proj_nir.bias.data, val=0.0)

                self.classifier_proj_text = nn.Linear(self.embed_dim, self.num_classes)   ##### 分类 loss 配合bn neck
                nn.init.normal_(self.classifier_proj_text.weight.data, std=0.001)
                # self.classifier_proj_text.weight = self.classifier_proj.weight
                nn.init.constant_(self.classifier_proj_text.bias.data, val=0.0)

                # self.classifier_text = nn.Linear(self.in_planes, self.num_classes)  ### dj: 由于分布差异比较大, 所有模态共享, bias应该不能设置为False !!! 
                # self.classifier.apply(weights_init_classifier)
                # self.classifier_proj = nn.Linear(self.in_planes_proj, self.num_classes)
                # self.classifier_proj.apply(weights_init_classifier)

            # self.bottleneck = nn.BatchNorm1d(self.embed_dim)
            # self.bottleneck.bias.requires_grad_(False)
            # self.bottleneck.apply(weights_init_kaiming)
            # self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
            # self.bottleneck_proj.bias.requires_grad_(False)
            # self.bottleneck_proj.apply(weights_init_kaiming)

        if 'mlm' in args.loss_names:
            fc_std = (2 * self._transformer.width)**-0.5
            scale = self._transformer.width**-0.5
            proj_std = scale * ((2 * self._transformer.layers)**-0.5)
            self.mlm_head = nn.Sequential(
                OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                            ('gelu', QuickGELU()),
                            ('ln', LayerNorm(self.embed_dim)),
                            ('fc', nn.Linear(self.embed_dim, args.vocab_size))]))
            # self.mlm_head = nn.Linear(self.embed_dim, args.num_colors)
            # init mlm head
            nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
            nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)

    def cross_former(self, q, k, v, ln_pre_i, ln_pre_t, ln_post, tf, attn):
        x = attn(
                ln_pre_t(q),
                ln_pre_i(k),
                ln_pre_i(v),
                need_weights=False)[0]
        x = q + x # residual connection (invalid for mcq and mcqmlm, valid for mlm)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # x2 = x2.permute(1, 0, 2)
        x = tf(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = ln_post(x)
        return x  # 平均池化

    def _init_cross_attn_tf(self, tf, attn):
        scale = tf.width**-0.5
        # self.pos_embedding = nn.Parameter(scale * torch.randn(self.embed_dim))

        proj_std = scale * ((2 * tf.layers)**-0.5)
        attn_std = scale
        fc_std = (2 * tf.width)**-0.5
        for block in tf.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        # init cross attn
        nn.init.normal_(attn.in_proj_weight, std=attn_std)
        nn.init.normal_(attn.out_proj.weight, std=proj_std)


    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')
    
    # def encode_image(self, image, input_img_type=None):
    #     x = self.base_model.encode_image(image, input_img_type)  ### 还有一个attention map
    #     return x

    def encode_image(self, image, input_img_type=None, text_inverse=None):
        if self.args.use_EMA_model:
            self.teacher_base_model.eval()
            _, _, image_features_proj = self.teacher_base_model.encode_image(image, input_img_type, text_inverse=text_inverse)
        else:
            _, _, image_features_proj = self.base_model.encode_image(image, input_img_type, text_inverse=text_inverse)  ### x11(分类), x12(分类), 经过映射后的特征(用于做对比学习)
        img_feature_proj = image_features_proj[:,0,:]
        return img_feature_proj.float(), image_features_proj[:, -1, :].float() # 返回图像的cls token, inverse之后的文本token(推理过程作为辅助)
        

    def encode_text(self, text, modality=None, use_prompt=False):
        if self.args.use_EMA_model:
            self.teacher_base_model.eval()
            x = self.teacher_base_model.encode_text(text, modality=modality, use_prompt=use_prompt)
        else:
            x = self.base_model.encode_text(text, modality=modality, use_prompt=use_prompt)
        return x.float() #[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()

    
    ################ 局部特征 #############################
    # def encode_image_tse(self, image, input_img_type=None):
    #     x, atten_i = self.base_model.encode_image(image, input_img_type)
    #     if input_img_type == InputImageType.rgb:
    #         i_tse_f = self.rgb_emb_layer(x, atten_i)
    #     elif input_img_type == InputImageType.sketch:
    #         i_tse_f = self.sk_emb_layer(x, atten_i)
    #     elif input_img_type == InputImageType.color_pencil:
    #         i_tse_f = self.cp_emb_layer(x, atten_i)
    #     elif input_img_type == InputImageType.nir:
    #         i_tse_f = self.nir_emb_layer(x, atten_i)
    #     return i_tse_f.float()
 
    # def encode_text_tse(self, text):
    #     x, atten_t = self.base_model.encode_text(text)
    #     t_tse_f = self.texual_emb_layer(x, text, atten_t)
    #     return t_tse_f.float()
    #########################################################

    def _use_inverse(self, caption_ids, imgs, input_img_type, use_prompt=True, weight_m=None, origin_token_masked_embed=None, mask_pos_list=None):
        # text_features = self.base_model.encode_text(caption_ids, modality=input_img_type, use_prompt=use_prompt)
        # t_feats_ = text_features[torch.arange(text_features.shape[0]), caption_ids.argmax(dim=-1)].float()
        # t_feats_inverse = self.inverseNet(t_feats_)

        # 找到masked的位置, 获得对应位置的原始text embedding
        # 做加权
        text_features = self.base_model.encode_text(
            caption_ids, modality=input_img_type, use_prompt=use_prompt, weight_m=weight_m, origin_token_masked_embed=origin_token_masked_embed, mask_pos_list=mask_pos_list
        )
        # 全局 token（通常对应 caption 的整体语义）
        t_global = text_features[torch.arange(text_features.shape[0]), caption_ids.argmax(dim=-1)]

        # 默认：只使用全局 token 作为 inverseNet 的输入，更稳健，且训练 / 测试的文本分布更一致。
        # 若需要复现旧实现，可在命令行加 --inverse_use_prompt_tokens True，
        # 此时会额外平均若干固定位置的 prompt token。
        # if getattr(self.args, "inverse_use_prompt_tokens", False):
        #     if input_img_type == InputImageType.sketch:
        #         t_feats_ = torch.mean(
        #             torch.cat([t_global.unsqueeze(1), text_features[:, 5: 5 + 2]], dim=1),
        #             dim=1,
        #         ).float()
        #     elif input_img_type == InputImageType.color_pencil:
        #         t_feats_ = torch.mean(
        #             torch.cat([t_global.unsqueeze(1), text_features[:, 7: 7 + 2]], dim=1),
        #             dim=1,
        #         ).float()
        #     elif input_img_type == InputImageType.nir:
        #         t_feats_ = torch.mean(
        #             torch.cat([t_global.unsqueeze(1), text_features[:, 7: 7 + 2]], dim=1),
        #             dim=1,
        #         ).float()
        #     else:
        #         t_feats_ = t_global.float()
        # else:
        #     t_feats_ = t_global.float()

        ## 进阶: 选择整个text token, 77个token一起inverse??
        t_feats_inverse = self.inverseNet(t_global.float())  ###
        # 传入image encoder
        return *self.base_model.encode_image(imgs, input_img_type, text_inverse=t_feats_inverse.half()), t_global.float(), text_features

    def _get_new_attn_w(self, _attn_w, fc):
        _fc_w = fc(_attn_w.float()).softmax(dim=1)
        _each_w = torch.cat([torch.ones(_fc_w.shape[0], 1).to(_fc_w.device), _fc_w, torch.ones(_fc_w.shape[0], 1).to(_fc_w.device)],dim=1) # eos, sos token位置占位1
        return _each_w.half()

    def _get_mask_position_lst(self, captions, mask_token):
        masked_position_idx = []
        for x in range(captions.shape[0]):
            _idx = torch.where(captions[x] == mask_token)[0]
            masked_position_idx.append(_idx)
        return masked_position_idx

    def _get_new_token(self, token_list):
        res_t = []
        for each_t in token_list:
            res_t.append(tokenize(self.tokenizer.decode(each_t),tokenizer=self.tokenizer, text_length=self.text_length, truncate=True))
        # imgs[0] = torch.stack(res_t, dim=0).to(imgs[0].device)
        return res_t
    

    def _mask_sim_token(self, i_feats_proj, origin_text_features, rgb_caption_ids, mask_token):
        """
        计算rgb cls token与没有mask的eos token之间的相似度, 找到每个样本最相近的前15个token, 然后mask掉, 作为其他图像模态的text token, 即与其他图像模态的cls token要对齐
        """
        _t_mask = (rgb_caption_ids.sum(dim=1) !=0)
        # 其他图像模态的token会先根据该图像模态特征与原文本所有token特征,计算一个相似度, 动态mask掉,再重组为新的增加prompt的
        with torch.no_grad():
            _attn_w = torch.softmax(
                (i_feats_proj.unsqueeze(1) @ origin_text_features.transpose(1, 2)[:, :, 1:-1]).squeeze() / math.sqrt(origin_text_features.shape[-1]), 
                    dim=1
            )  # bs, 77
            # 把text与rgb最相关的前77*0.2个token,对应的其他图像模态直接mask; eos 和 sos token保留, 这两个是最相关的
            _mask = torch.zeros_like(rgb_caption_ids[:, 1:-1])
            _top_num = int(origin_text_features.shape[1] * 0.2)
            _topk_token_idx = torch.sort(_attn_w, dim=1, descending=True)[1][:, :_top_num]
            _mask.scatter_(1, _topk_token_idx, 1)
            # 找到最相关的位置, 然后替换为随机可学习向量
            rgb_caption_ids_masked = torch.where(_mask.bool(), torch.full_like(rgb_caption_ids[:, 1:-1], mask_token), rgb_caption_ids[:, 1:-1])

        dynamic_mask_caption = torch.stack(self._get_new_token(rgb_caption_ids_masked.cpu().tolist()), dim=0).to(i_feats_proj.device) # tokenizer会自动加上前后两个token
        dynamic_mask_caption = torch.where(_t_mask.unsqueeze(1), dynamic_mask_caption, torch.zeros_like(dynamic_mask_caption))
        return dynamic_mask_caption


    def forward(self, batch, epoch=None):
        ret = dict()
        images = batch['rgbs']
        # label = batch['pids']
        combs_select = batch['combs_select']
        # combs_select = torch.tensor([True] * 4).to(images.device)
        caption_ids, caption_ids_rgb, caption_ids_ni, caption_ids_cp = None, None, None, None

        if self.args.use_inverse:
            rgb_caption_ids = batch['rgb_caption_ids'] if combs_select[3] else None
            masked_rgb_caption_ids = batch['masked_rgb_caption_ids'] if combs_select[3] else None
            # nir_caption_ids = batch['nir_caption_ids'] if combs_select[2] else None
            # cp_caption_ids = batch['cp_caption_ids'] if combs_select[1] else None
            # sk_caption_ids = batch['sk_caption_ids'] if combs_select[0] else None
        else:
            if combs_select[3]:
                if 'rgb_text' in batch:
                    caption_ids_rgb = batch['rgb_text']
                    caption_ids_ni = batch['nirs_text']
                    caption_ids_cp = batch['cps_text']
                else:
                    caption_ids = batch['caption_ids']

            # rgb_caption_ids = batch['rgb_caption_ids'] if combs_select[3] else None
            # masked_rgb_caption_ids = batch['masked_rgb_caption_ids'] if combs_select[3] else None

        ## 0730: 全部换成 mask之后的text feature
        # caption_ids = batch.get('mlm_ids')

        simages =  batch['sks'] if combs_select[0] else None
        cimages = batch['cps'] if combs_select[1] else None
        nimages = batch['nirs'] if combs_select[2] else None
        cat_images = [images]
        cat_images += [img for img in [simages, cimages, nimages]]

        # 全部共用一个 CLIP image encoder ==> 改为: 多个adapter mlp
        # image_feats, text_feats = self.base_model(torch.concat(cat_images, dim=0), caption_ids)  ### 不concat images

        ############## 随机选择一个batch样本
        # rgbimage_feats, simage_feats, cimage_feats, nimage_feats, text_feats = self.base_model(
        #     cat_images, caption_ids, combs_select=combs_select)

        # x11和x12都用于smooth 分类, 区别在于: 用的不同的分类层;
        i_feats, i_feats_last, i_feats_proj, si_feats, si_feats_last, si_feats_proj, ci_feats, ci_feats_last, \
            ci_feats_proj, ni_feats, ni_feats_last, ni_feats_proj, t_feats, t_feats_cp, t_feats_nir = None, None, None, None, None, None, None, None, None, None, None, None, None, None, None
        sk_txt_g, cp_txt_g, nir_txt_g = None, None, None
        si_feats_inverse_txt, ci_feats_inverse_txt, ni_feats_inverse_txt = None, None, None

        masked_feats, masked_feats_sk, masked_feats_cp, masked_feats_nir = None, None, None, None
        i_feats_proj_t, si_feats_proj_t, ci_feats_proj_t, ni_feats_proj_t = None, None, None, None

        sk_text_mean_feat, cp_text_mean_feat, nir_text_mean_feat = None, None, None
        # si_local_feat, ci_local_feat, ni_local_feat, i_local_feat = None, None, None, None
        si_shared_feat, ci_shared_feat, ni_shared_feat, i_shared_feat = None, None, None, None
        si_feats_before_proj, ci_feats_before_proj, ni_feats_before_proj = None, None, None

        fusion_img_feat = None

        if self.args.use_inverse:
            mask_token = self.tokenizer.encoder["<|mask|>"]
            # rgb
            # image_features_rgb_last, image_features_rgb, image_features_rgb_proj = self._use_inverse(rgb_caption_ids, images, InputImageType.rgb, use_prompt=False)
            _, _, image_features_rgb_proj = self.base_model.encode_image(images, InputImageType.rgb)
            _, _, image_features_sk_proj = self.base_model.encode_image(simages, InputImageType.sketch)
            _, _, image_features_cp_proj = self.base_model.encode_image(cimages, InputImageType.color_pencil)
            _, _, image_features_nir_proj = self.base_model.encode_image(nimages, InputImageType.nir)
            i_feats_proj = image_features_rgb_proj[:, 0, :].float() ## proj
            i_local_feat = image_features_rgb_proj[:, 1:, :].mean(dim=1).float()
            i_merge_feat = torch.cat([i_feats_proj, i_local_feat], dim=1)

            # 提取图像的局部特征
            # if self.args.use_local_feat:
            #     i_feats_proj_local = self.conv_inverse_net(image_features_rgb_proj[:, 1:, :].float())

            # dynamic_mask_caption_sk, dynamic_mask_caption_cp, dynamic_mask_caption_nir = None, None, None
            dynamic_mask_caption = None
            if rgb_caption_ids is not None and masked_rgb_caption_ids is not None:
                _t_mask = (rgb_caption_ids.sum(dim=1) !=0)

                # 没有mask的rgb的text token特征的 eos token, 用于计算其他图像模态mask attn
                origin_text_features = self.encode_text(rgb_caption_ids)
                # 计算loss采用mask之后的原text
                origin_text_features_masked = self.encode_text(masked_rgb_caption_ids)
                t_feats = origin_text_features_masked[torch.arange(origin_text_features_masked.shape[0]), masked_rgb_caption_ids.argmax(dim=-1)].float()

                _attn_w = torch.softmax(
                        (i_feats_proj.unsqueeze(1) @ origin_text_features.transpose(1, 2)[:, :, 1:-1]).squeeze() / math.sqrt(origin_text_features.shape[-1]), 
                            dim=1
                    )  # bs, 77
                
                # 其他图像模态的token会先根据该图像模态特征与原文本所有token特征,计算一个相似度, 动态mask掉,再重组为新的增加prompt的
                with torch.no_grad():
                    # _attn_w = torch.softmax(
                    #     (i_feats_proj.unsqueeze(1) @ origin_text_features.transpose(1, 2)[:, :, 1:-1]).squeeze() / math.sqrt(origin_text_features.shape[-1]), 
                    #         dim=1
                    # )  # bs, 77
                    # 把text与rgb最相关的前77*0.2个token,对应的其他图像模态直接mask; eos 和 sos token保留, 这两个是最相关的
                    _mask = torch.zeros_like(rgb_caption_ids[:, 1:-1])
                    _top_num = int(origin_text_features.shape[1] * 0.2)
                    _topk_token_idx = torch.sort(_attn_w, dim=1, descending=True)[1][:, :_top_num]
                    _mask.scatter_(1, _topk_token_idx, 1)
                    # 找到最相关的位置, 然后替换为随机可学习向量
                    rgb_caption_ids_masked = torch.where(_mask.bool(), torch.full_like(rgb_caption_ids[:, 1:-1], mask_token), rgb_caption_ids[:, 1:-1])
                    # rgb_caption_ids_masked = torch.cat([rgb_caption_ids[:, 0].unsqueeze(1), rgb_caption_ids_masked, rgb_caption_ids[:, -1].unsqueeze(1)], dim=1) # 修复bug: 不需要人为加前后的token
                    # 把rgb caption与rgb图像模态最相关的token mask之后, 作为其他图像模态的pseudo mask caption

                dynamic_mask_caption = torch.stack(self._get_new_token(rgb_caption_ids_masked.cpu().tolist()), dim=0).to(images.device) # tokenizer会自动加上前后两个token
                dynamic_mask_caption = torch.where(_t_mask.unsqueeze(1), dynamic_mask_caption, torch.zeros_like(dynamic_mask_caption))
                # dynamic_mask_caption_cp = copy.deepcopy(dynamic_mask_caption_sk)
                # dynamic_mask_caption_nir = copy.deepcopy(dynamic_mask_caption_sk)
                # 进阶: 其他图像模态也和rgb caption计算相似度, 如果最相关的正好也是rgb与rgb caption最相关的, 保留, 否则mask TODO
                # 结合门控机制: 注意, mask的位置本质上就是会用一个可学习的token embedding更新 !!! 
                masked_position_idx = self._get_mask_position_lst(
                    torch.cat([rgb_caption_ids[:, 0].unsqueeze(1), rgb_caption_ids_masked, rgb_caption_ids[:, -1].unsqueeze(1)], dim=1), 
                    mask_token)
                
                i_local_feat_txt = origin_text_features_masked[torch.arange(origin_text_features_masked.shape[0]).unsqueeze(1), torch.stack(masked_position_idx, dim=0)].mean(dim=1).float()
                i_merge_feat_txt = torch.cat([t_feats, i_local_feat_txt], dim=1)  # bs, 1024

                # 统计被mask位置的个数(当前的)
                # all_vals = torch.cat(masked_position_idx)  # (N*15,)
                # counts = torch.bincount(all_vals, minlength=76)  # (76,)
                # _top_token_counts, _top_token_indices = torch.topk(counts, _top_num)
                # self.token_counts[_top_token_indices.detach().cpu()] += 1
                # sk_text_mean_feat = self.sk_learned_feat[_top_token_indices].mean(dim=0)
                # masked_position_idx_rgb = self._get_mask_position_lst(masked_rgb_caption_ids, mask_token)
                
                # if self.args.use_local_feat:
                #     i_local_feat = origin_text_features[torch.arange(origin_text_features.shape[0]).unsqueeze(1), torch.stack(masked_position_idx, dim=0)].mean(dim=1).float()
                #     masked_feats = []
                #     for xx in range(len(masked_position_idx_rgb)):
                #         # 把最相似的若干个局部text特征取平均(用masked还是没masked的???): 用于对齐image的局部特征; 
                #         masked_feats.append(origin_text_features[xx][masked_position_idx_rgb[xx], :].mean(dim=0).float())
                #     masked_feats = torch.stack(masked_feats, dim=0)


                # self.base_model.token_embedding(torch.where(rgb_caption_ids_masked == mask_token)).type(self.base_model.dtype)
                # sk_caption_ids = torch.stack(self._get_new_token(rgb_caption_ids_masked.cpu().tolist(), ''), dim=0).to(images.device)
                # cp_caption_ids = torch.stack(self._get_new_token(rgb_caption_ids_masked.cpu().tolist(), ''), dim=0).to(images.device)
                # nir_caption_ids = torch.stack(self._get_new_token(rgb_caption_ids_masked.cpu().tolist(), ''), dim=0).to(images.device)  ## 不加prompt

                # 只有text的情况, 也能用不同的mask, 获得全局token
                # if simages is None and cimages is None and nimages is None:
                text_features_sk = self.base_model.encode_text(
                    # 用不同的caption
                    dynamic_mask_caption, 
                    weight_m=True, #self._get_new_attn_w(_attn_w, self.mask_weight_sk) if self.args.mask_weight else None, 
                    origin_token_masked_embed=self.sk_learned_feat, # if self.args.mask_weight else None, # 进阶: TODO: 把选择最多的前15个可以记录并辅助推理; 也可以对只有图像的时候辅助训练
                    mask_pos_list=masked_position_idx #if self.args.mask_weight else None
                )
                si_feats_inverse_txt = text_features_sk[torch.arange(text_features_sk.shape[0]), dynamic_mask_caption.argmax(dim=-1)].float()
                # 可学习的局部特征提取出来
                si_local_feat_txt = text_features_sk[torch.arange(text_features_sk.shape[0]).unsqueeze(1), torch.stack(masked_position_idx, dim=0)].mean(dim=1).float()
                si_merge_feat_txt = torch.cat([si_feats_inverse_txt, si_local_feat_txt], dim=1)

                text_features_cp = self.base_model.encode_text(
                    dynamic_mask_caption, 
                    weight_m=True, #self._get_new_attn_w(_attn_w, self.mask_weight_cp) if self.args.mask_weight else None, 
                    origin_token_masked_embed=self.cp_learned_feat, # if self.args.mask_weight else None, 
                    mask_pos_list=masked_position_idx # if self.args.mask_weight else None
                )
                ci_feats_inverse_txt = text_features_cp[torch.arange(text_features_cp.shape[0]), dynamic_mask_caption.argmax(dim=-1)].float()
                ci_local_feat_txt = text_features_cp[torch.arange(text_features_cp.shape[0]).unsqueeze(1), torch.stack(masked_position_idx, dim=0)].mean(dim=1).float()
                ci_merge_feat_txt = torch.cat([ci_feats_inverse_txt, ci_local_feat_txt], dim=1)


                text_features_nir = self.base_model.encode_text(
                    dynamic_mask_caption, 
                    weight_m=True, #self._get_new_attn_w(_attn_w, self.mask_weight_nir) if self.args.mask_weight else None, 
                    origin_token_masked_embed=self.nir_learned_feat, # if self.args.mask_weight else None, 
                    mask_pos_list=masked_position_idx # if self.args.mask_weight else None
                )
                ni_feats_inverse_txt = text_features_nir[torch.arange(text_features_nir.shape[0]), dynamic_mask_caption.argmax(dim=-1)].float()
                ni_local_feat_txt = text_features_nir[torch.arange(text_features_nir.shape[0]).unsqueeze(1), torch.stack(masked_position_idx, dim=0)].mean(dim=1).float()
                ni_merge_feat_txt = torch.cat([ni_feats_inverse_txt, ni_local_feat_txt], dim=1)


            # 新增: rgb也用inverse net; 覆盖i_feats_proj !!!
            # if self.args.use_rgb_inverse and masked_rgb_caption_ids is not None:
            #     _, _, masked_rgb_feat_proj, t_feats, _ = self._use_inverse(masked_rgb_caption_ids, images, InputImageType.rgb, use_prompt=False)
            #     masked_rgb_i_feats = masked_rgb_feat_proj[:, 0, :].float() ## proj
            #     i_feats_proj = masked_rgb_i_feats
            #     t_feats = self._transformer(t_feats + masked_rgb_feat_proj[:, -1, :].float())

            ### 增加没有 text 的情况: 对于只有图像的情况, 不经过inverse, 直接拿image encoder的结果
            ## sk
            # if simages is not None:
            #     if dynamic_mask_caption is not None:
            #         image_features_sk_last, image_features_sk, image_features_sk_proj, sk_txt_g, sk_text_features_masked = self._use_inverse(
            #             dynamic_mask_caption, simages, InputImageType.sketch, use_prompt=False, 
            #             weight_m=self._get_new_attn_w(_attn_w, self.mask_weight_sk) if self.args.mask_weight else None, 
            #             origin_token_masked_embed=self.sk_learned_feat if self.args.mask_weight else None, 
            #             mask_pos_list=masked_position_idx if self.args.mask_weight else None)
            #         # si_local_feat = sk_text_features_masked[torch.arange(sk_text_features_masked.shape[0]).unsqueeze(1), torch.stack(masked_position_idx, dim=0)].mean(dim=1).float()
            #         # 对于有图像和文本的, 就提取对应的局部特征
            #         # if self.args.use_local_feat:
            #             # masked_feats_sk = []
            #             # for xx in range(len(masked_position_idx)):
            #             #     # 把最相似的若干个局部text特征取平均(用masked还是没masked的???): 用于对齐image的局部特征; ==> TODO 注意: 每个token embedding都是一样的, 所以优化会变差
            #             #     masked_feats_sk.append(sk_text_features_masked[xx][masked_position_idx[xx], :].mean(dim=0).float())
            #             # masked_feats_sk = torch.stack(masked_feats_sk, dim=0)
            #         sk_img_patchs = image_features_sk_proj[:, 1:-1, :].float()
            #     else:
            #         image_features_sk_last, image_features_sk, image_features_sk_proj = self.base_model.encode_image(simages, InputImageType.sketch)
            #         # 对于只有图像的, 只提取图像的局部特征
            #         sk_img_patchs = image_features_sk_proj[:, 1:, :].float()
            #     if self.args.use_local_feat:
            #         si_feats_proj_local = self.conv_inverse_net(sk_img_patchs)
            #     # if fusion_img_feat is None:
            #     #     fusion_img_feat = image_features_sk_proj[:, 0, :].float()
            #     # else:
            #     #     fusion_img_feat += image_features_sk_proj[:, 0, :].float()
                
            # ## cp
            # if cimages is not None:
            #     if dynamic_mask_caption is not None:
            #         image_features_cp_last, image_features_cp, image_features_cp_proj, cp_txt_g, cp_text_features_masked = self._use_inverse(
            #             dynamic_mask_caption, cimages, InputImageType.color_pencil, use_prompt=False,
            #             weight_m=self._get_new_attn_w(_attn_w, self.mask_weight_cp) if self.args.mask_weight else None,
            #              origin_token_masked_embed=self.cp_learned_feat if self.args.mask_weight else None, 
            #              mask_pos_list=masked_position_idx if self.args.mask_weight else None)
            #         # ci_local_feat = cp_text_features_masked[torch.arange(cp_text_features_masked.shape[0]).unsqueeze(1), torch.stack(masked_position_idx, dim=0)].mean(dim=1).float()
            #         # if self.args.use_local_feat:
            #         #     masked_feats_cp = []
            #         #     for xx in range(len(masked_position_idx)):
            #         #         masked_feats_cp.append(cp_text_features_masked[xx][masked_position_idx[xx], :].mean(dim=0).float())
            #         #     masked_feats_cp = torch.stack(masked_feats_cp, dim=0)
            #         cp_img_patchs = image_features_cp_proj[:, 1:-1, :].float()
            #     else:
            #         image_features_cp_last, image_features_cp, image_features_cp_proj = self.base_model.encode_image(cimages, InputImageType.color_pencil)
            #         cp_img_patchs = image_features_cp_proj[:, 1:, :].float()
            #     if self.args.use_local_feat:
            #         ci_feats_proj_local = self.conv_inverse_net(cp_img_patchs)
            #     # if fusion_img_feat is None:
            #     #     fusion_img_feat = image_features_cp_proj[:, 0, :].float()
            #     # else:
            #     #     fusion_img_feat += image_features_cp_proj[:, 0, :].float()
            # ## nir
            # if nimages is not None:
            #     if dynamic_mask_caption is not None:
            #         image_features_nir_last, image_features_nir, image_features_nir_proj, nir_txt_g, nir_text_features_masked = self._use_inverse(
            #             dynamic_mask_caption, nimages, InputImageType.nir, use_prompt=False,
            #             weight_m=self._get_new_attn_w(_attn_w, self.mask_weight_nir) if self.args.mask_weight else None, 
            #             origin_token_masked_embed=self.nir_learned_feat if self.args.mask_weight else None,
            #             mask_pos_list=masked_position_idx if self.args.mask_weight else None)
            #         # ni_local_feat = nir_text_features_masked[torch.arange(nir_text_features_masked.shape[0]).unsqueeze(1), torch.stack(masked_position_idx, dim=0)].mean(dim=1).float()
            #         # if self.args.use_local_feat:
            #         #     masked_feats_nir = []
            #         #     for xx in range(len(masked_position_idx)):
            #         #         masked_feats_nir.append(nir_text_features_masked[xx][masked_position_idx[xx], :].mean(dim=0).float())
            #         #     masked_feats_nir = torch.stack(masked_feats_nir, dim=0)
            #         nir_img_patchs = image_features_nir_proj[:, 1:-1, :].float()
            #     else:
            #         image_features_nir_last, image_features_nir, image_features_nir_proj = self.base_model.encode_image(nimages, InputImageType.nir)
            #         nir_img_patchs = image_features_nir_proj[:, 1:, :].float()
            #     if self.args.use_local_feat:
            #         ni_feats_proj_local = self.conv_inverse_net(nir_img_patchs)
            #     # if fusion_img_feat is None:
            #     #     fusion_img_feat = image_features_nir_proj[:, 0, :].float()
            #     # else:
            #     #     fusion_img_feat += image_features_nir_proj[:, 0, :].float()
        else:
            # x11, x12, xproj
            image_features_rgb_last, image_features_rgb, image_features_rgb_proj, \
            image_features_sk_last, image_features_sk, image_features_sk_proj, \
            image_features_cp_last, image_features_cp, image_features_cp_proj,\
            image_features_nir_last, image_features_nir, image_features_nir_proj, text_features, text_features_sk, text_features_cp, text_features_nir = self.base_model(
                    cat_images, caption_ids, combs_select=combs_select, rgb_caption_ids=caption_ids_rgb, nir_caption_ids=caption_ids_ni, cp_caption_ids=caption_ids_cp)

            i_feats_proj = image_features_rgb_proj[:, 0, :].float() ## proj
            # 2026.0601: 已经把i_feats_proj换为nir了, 而不是rgb; 而原来的nir变为rgb了

            # i_shared_feat = image_features_rgb_proj[:, -1, :].float()
            # i_merge_feat = torch.cat([i_feats_proj, i_local_feat], dim=1)
            # i_feats_before_proj = image_features_rgb[:, 0, :].float()

            if text_features is not None:
                if caption_ids_rgb is not None:
                    t_feats = text_features[torch.arange(text_features.shape[0]), caption_ids_rgb.argmax(dim=-1)].float()
                else:
                    t_feats = text_features[torch.arange(text_features.shape[0]), caption_ids.argmax(dim=-1)].float()
            
            if text_features_cp is not None:
                t_feats_cp = text_features_cp[torch.arange(text_features_cp.shape[0]), caption_ids_cp.argmax(dim=-1)].float()
            
            if text_features_nir is not None:
                t_feats_nir = text_features_nir[torch.arange(text_features_nir.shape[0]), caption_ids_ni.argmax(dim=-1)].float()
                
            # if self.args.use_EMA_model:
            #     with torch.no_grad():
            #         # 如果用了ema model
            #         _, _, image_features_rgb_proj_t, \
            #         _, _, image_features_sk_proj_t, \
            #         _, _, image_features_cp_proj_t,\
            #         _, _, image_features_nir_proj_t, text_features_t, text_features_sk_t, text_features_cp_t, text_features_nir_t = self.teacher_base_model(
            #             cat_images, caption_ids, combs_select=combs_select, rgb_caption_ids=caption_ids_rgb, nir_caption_ids=caption_ids_ni, cp_caption_ids=caption_ids_cp
            #         )
                # i_feats_proj_t = image_features_rgb_proj_t[:, 0, :].float() ## proj teacher feat
                # if text_features_t is not None:
                #     t_feats_t = text_features_t[torch.arange(text_features_t.shape[0]), caption_ids.argmax(dim=-1)].float() ## teacher feat

        logit_scale = self.logit_scale.exp()
        logit_scale_txt = self.logit_scale_txt.exp()
        logit_scale_sup_cons = self.logit_scale_sup_cons.exp()
        logit_scale_fuse = self.logit_scale_fuse.exp()
        # tau_scale = self.tau_scale.exp()

        ret.update({'temperature': 1 / logit_scale})

        # mask_token = self.tokenizer.encoder["<|mask|>"]
        # dynamic_token_ids = self._mask_sim_token(i_feats_proj, self.encode_text(rgb_caption_ids), rgb_caption_ids, mask_token)
        # dynamic_mask_text_feats = self.encode_text(dynamic_token_ids)
        # dynamic_mask_t_feats = dynamic_mask_text_feats[torch.arange(dynamic_mask_text_feats.shape[0]), dynamic_token_ids.argmax(dim=-1)].float()

        # logit_scale_sk = self.logit_scale_sk.exp()
        # ret.update({'temperature_sk': 1 / logit_scale_sk})

        # logit_scale_cp = self.logit_scale_cp.exp()
        # ret.update({'temperature_cp': 1 / logit_scale_cp})

        # logit_scale_nir = self.logit_scale_nir.exp()
        # ret.update({'temperature_nir': 1 / logit_scale_nir})

        # 新增inverse之后的text token
        # i_feats_inverse_txt = image_features_rgb_proj[:, -1, :].float()  ## 用于id loss, sdm loss
        
        if simages is not None:
            # assert image_features_sk_proj.shape[1] == 193
            # si_feats = image_features_sk[:, 0, :].float()
            # si_feats_last = image_features_sk_last[:, 0, :].float()
            si_feats_proj = image_features_sk_proj[:, 0, :].float()
            # if self.args.use_EMA_model:
            #     si_feats_proj_t = image_features_sk_proj_t[:, 0, :].float()
            # si_feats_before_proj = image_features_sk[:, 0, :].float()
            # si_shared_feat = image_features_sk_proj[:, -1, :].float()
            # si_local_feat = image_features_sk_proj[:, 1:, :].mean(dim=1).float()
            # si_merge_feat = torch.cat([si_feats_proj, si_local_feat], dim=1)
            # if rgb_caption_ids is not None:
            #     si_feats_inverse_txt = image_features_sk_proj[:, -1, :].float()
            #     # 原始的text eos token 和 经过image encoder之后的 eos token相加, 再经过transformer融合
                
            #     si_feats_inverse_txt = self._transformer(sk_txt_g + si_feats_inverse_txt)# 进阶: 再加一个残差 TODO

        if cimages is not None:
            # assert image_features_cp_proj.shape[1] == 193
            # ci_feats = image_features_cp[:, 0, :].float()
            # ci_feats_last = image_features_cp_last[:, 0, :].float()
            ci_feats_proj = image_features_cp_proj[:, 0, :].float()
            # if self.args.use_EMA_model:
            #     ci_feats_proj_t = image_features_cp_proj_t[:, 0, :].float()
            # ci_feats_before_proj = image_features_cp[:, 0, :].float()
            # ci_shared_feat = image_features_cp_proj[:, -1, :].float()
            # ci_local_feat = image_features_cp_proj[:, 1:, :].mean(dim=1).float()
            # ci_merge_feat = torch.cat([ci_feats_proj, ci_local_feat], dim=1)
            # if rgb_caption_ids is not None:
            #     ci_feats_inverse_txt = image_features_cp_proj[:, -1, :].float()
            #     ci_feats_inverse_txt = self._transformer(cp_txt_g + ci_feats_inverse_txt)
        if nimages is not None:
            # assert image_features_nir_proj.shape[1] == 193
            # ni_feats = image_features_nir[:, 0, :].float()
            # ni_feats_last = image_features_nir_last[:, 0, :].float()
            ni_feats_proj = image_features_nir_proj[:, 0, :].float()  ### 2026.0601: 这个已经是rgb了
            # if self.args.use_EMA_model:
            #     ni_feats_proj_t = image_features_nir_proj_t[:, 0, :].float()
            # ni_feats_before_proj = image_features_nir[:, 0, :].float()
            # ni_shared_feat = image_features_nir_proj[:, -1, :].float()
            # ni_local_feat = image_features_nir_proj[:, 1:, :].mean(dim=1).float()
            # ni_merge_feat = torch.cat([ni_feats_proj, ni_local_feat], dim=1)
            # if rgb_caption_ids is not None:
            #     ni_feats_inverse_txt = image_features_nir_proj[:, -1, :].float()
            #     ni_feats_inverse_txt = self._transformer(nir_txt_g + ni_feats_inverse_txt)

        # return i_feats_proj,  si_feats_proj, ci_feats_proj, ni_feats_proj, t_feats

        # 增加一个supervised constrast loss: 每个样本会和所有其他模态相同id样本计算对比loss,分母是和这个batch的所有模态样本除了它本身
        if 'sup_cons' in self.current_task:
            all_batch_f = i_feats_proj
            all_batch_label = batch['pids']
            for ff in [t_feats, si_feats_proj, ci_feats_proj, ni_feats_proj]:
                # 2026.0601: 换为nir的text features
            # for ff in [t_feats_nir, si_feats_proj, ci_feats_proj, ni_feats_proj]:
                if ff is not None:
                    all_batch_f = torch.cat([all_batch_f, ff], dim=0)
                    all_batch_label = torch.cat([all_batch_label, batch['pids']], dim=0)
            ret['sup_cons_loss'] = objectives.contrastive_loss_batch(all_batch_f, all_batch_label, tau=0.1)  ### TODO : temp是否要变为可学习logits? 换成可学习0.02效果下降

        # 增加余弦相似度loss:
        if 'cos_sim' in self.current_task:
            adapter_i = self.adapter_shared(i_feats_proj)
            feat_list = []
            for _img_feat in [t_feats, si_feats_proj, ci_feats_proj, ni_feats_proj]:
                if _img_feat is not None:
                    feat_list.append(self.adapter_shared(_img_feat))
            #         sdm_loss = objectives.compute_sdm(i_feats_proj, shared_img_feat, batch['pids'], logit_scale=0.02)
            #         if 'sdm_loss' in ret:
            #             ret['sdm_loss'] += sdm_loss
            #         else:
            #             ret['sdm_loss'] = sdm_loss
            feat_list = [F.normalize(i, p=2, dim=-1) for i in feat_list]
            for i in range(len(feat_list)):
                cos_sim = F.cosine_similarity(feat_list[i], adapter_i, dim=-1)
                if 'cos_loss' in ret:
                    ret['cos_loss'] += (1 - cos_sim).mean()
                else:
                    ret['cos_loss'] = (1 - cos_sim).mean()


        # 修改itc loss为sum positive constrastive loss
        if 'itc' in self.current_task:
            itc_scale_lst = [logit_scale_txt, logit_scale, logit_scale, logit_scale]
            for idx, feat in enumerate([t_feats, si_feats_proj, ci_feats_proj, ni_feats_proj]):
            # for idx, feat in enumerate([i_merge_feat_txt, si_merge_feat, ci_merge_feat, ni_merge_feat]):
                if feat is not None:
                    itc_total = objectives.multi_positive_itc(i_feats_proj, feat, batch['pids'], itc_scale_lst[idx])
                    # itc_total = itc_total + (self.supcontrast_proj(i_feats_proj, feat, batch['pids'], batch['pids']) + \
                    #         self.supcontrast_proj(feat, i_feats_proj, batch['pids'], batch['pids']) / 2)  ###双向 itc loss, 
                    if 'itc_loss' in ret:
                        ret['itc_loss'] += itc_total
                    else:
                        ret['itc_loss'] = itc_total
            
            # 其他图像模态都要和text对齐, 加上0.5的权重
            if t_feats is not None:
                for img_feat in [si_feats_proj, ci_feats_proj, ni_feats_proj]:
                    if img_feat is not None:
                        itc_total = 0.5 * objectives.multi_positive_itc(img_feat, t_feats, batch['pids'], logit_scale_txt)
                        if 'itc_loss' in ret:
                            ret['itc_loss'] += itc_total
                        else:
                            ret['itc_loss'] = itc_total


        if 'kl' in self.current_task:
            # rgb与text的对比分布作为gt, 其他图像与text的对比分布作为input, 计算KL 散度
            if t_feats is not None:
                logits_per_image, logits_per_text = objectives.multi_positive_itc_get_logits(i_feats_proj, t_feats, logit_scale_txt)
                logits_per_image, logits_per_text = F.softmax(logits_per_image, dim=-1), F.softmax(logits_per_text, dim=-1)
                for img_feat in [si_feats_proj, ci_feats_proj, ni_feats_proj]:
                    if img_feat is not None:
                        logits_per_image_each, logits_per_text_each = objectives.multi_positive_itc_get_logits(img_feat, t_feats, logit_scale_txt)
                        logits_per_image_each, logits_per_text_each = F.log_softmax(logits_per_image_each, dim=-1), F.log_softmax(logits_per_text_each, dim=-1)
                        kl_loss = self.kl_div(logits_per_image_each, logits_per_image) + self.kl_div(logits_per_text_each, logits_per_text)
                        if 'kl_loss' in ret:
                            ret['kl_loss'] += kl_loss
                        else:
                            ret['kl_loss'] = kl_loss

            # 局部特征 TODO


        if 'fuse' in self.current_task:
            # 思路一: 融合特征加上rgb, 且只用id loss, 推理不加融合特征 => 去掉text
            # 思路二: 融合特征不加入rgb, 推理加上融合特征 TODO 
            # base_feat = i_feats_proj.unsqueeze(1)
            base_feat = None
            for ff in [t_feats, si_feats_proj, ci_feats_proj, ni_feats_proj]:
                if ff is not None:
                    if base_feat is None:
                        base_feat = ff.unsqueeze(1)
                    else:
                        base_feat = torch.cat([base_feat, ff.unsqueeze(1)], dim=1)
            base_feat = base_feat.permute(1, 0, 2)  ## N, batchsize, 512
            fuse_feat = self._transformer_shared(base_feat)
            fuse_feat = fuse_feat.permute(1, 0, 2).mean(dim=1)  # batchsize, 512
            # ret['sdm_loss_fuse'] = objectives.compute_sdm(i_feats_proj, fuse_feat, batch['pids'], logit_scale_fuse)
            # 用不同的分类层
            fuse_logits_proj = self.classifier_proj_fuse(fuse_feat)
            ret.update({'fuse_loss': objectives.compute_id_new( fuse_logits_proj, batch['pids'])}) 
            self._compute_id_loss_and_acc(None, fuse_logits_proj, batch['pids'], 'fuse_acc', ret)


        if 'sdm' in self.current_task:
            # 对于rgbnt201数据集, 由于评测不再只是rgb作为gallery, 因此sdm用每个图像模态与其对应的文本模态计算sdm loss, 
            # 而不是把 rgb 作为anchor; 所有图像模态特征是否要和rgb对齐 ??? TODO
            # if self.args.dataset_name in ['RGBNT201_Text']:
            #     if t_feats is not None:
            #         # 只有text的时候才会有sdm loss
            #         sdm_loss = objectives.compute_sdm(i_feats_proj, t_feats, batch['pids'], logit_scale_txt)
            #         if 'sdm_loss' in ret:
            #             ret['sdm_loss'] += sdm_loss
            #         else:
            #             ret['sdm_loss'] = sdm_loss
            #     if ci_feats_proj is not None and t_feats_cp is not None:
            #         sdm_loss = objectives.compute_sdm(ci_feats_proj, t_feats_cp, batch['pids'], logit_scale_txt)
            #         if 'sdm_loss' in ret:
            #             ret['sdm_loss'] += sdm_loss
            #         else:
            #             ret['sdm_loss'] = sdm_loss
            #     if ni_feats_proj is not None and t_feats_nir is not None:
            #         sdm_loss = objectives.compute_sdm(ni_feats_proj, t_feats_nir, batch['pids'], logit_scale_txt)
            #         if 'sdm_loss' in ret:
            #             ret['sdm_loss'] += sdm_loss
            #         else:
            #             ret['sdm_loss'] = sdm_loss
            # else:
            sdm_scale_lst = [logit_scale_txt, logit_scale, logit_scale, logit_scale, logit_scale]
            for idx, f in enumerate([t_feats, si_feats_proj, ci_feats_proj, ni_feats_proj]):  # 所有图像和text去对齐 ?
                # 2026.0601: 换成nir text
            # for idx, f in enumerate([t_feats_nir, si_feats_proj, ci_feats_proj, ni_feats_proj]):
            # for idx, f in enumerate([i_merge_feat_txt, si_merge_feat, ci_merge_feat, ni_merge_feat]):
                if f is not None:
                    # if self.args.use_EMA_model:
                    #     with torch.no_grad():
                    #         teacher_i2t, teacher_t2i = objectives.compute_sdm_teacher(i_feats_proj, f, sdm_scale_lst[idx])
                    #     sdm_loss = objectives.compute_sdm(i_feats_proj, f, batch['pids'], sdm_scale_lst[idx], teacher_i2t=teacher_i2t, teacher_t2i=teacher_t2i) 
                    # else:
                    sdm_loss = objectives.compute_sdm(i_feats_proj, f, batch['pids'], sdm_scale_lst[idx])  ### 所有的都只和rgb对齐, 因为rgb是检索目标
                    if 'sdm_loss' in ret:
                        ret['sdm_loss'] += sdm_loss
                    else:
                        ret['sdm_loss'] = sdm_loss
                

        if 'triplet' in self.current_task:
            # 只对image 
            # all_batch_f_before_proj = i_feats_before_proj
            # all_batch_label_before_proj = batch['pids']
            # all_batch_modality = torch.zeros_like(batch['pids'], device=batch['pids'].device)
            # num = [1,2,3]
            # for idx, ff in enumerate([si_feats_before_proj, ci_feats_before_proj, ni_feats_before_proj]):
            #     if ff is not None:
            #         all_batch_f_before_proj = torch.cat([all_batch_f_before_proj, ff], dim=0)
            #         all_batch_label_before_proj = torch.cat([all_batch_label_before_proj, batch['pids']], dim=0)
            #         all_batch_modality = torch.cat([all_batch_modality, torch.zeros_like(batch['pids'], device=batch['pids'].device) + num[idx]], dim=0)
            # triplet_loss = self.triplet_cross_modal(all_batch_f_before_proj, all_batch_label_before_proj, all_batch_modality)[0]
            # if 'triplet_loss' in ret:
            #     ret['triplet_loss'] += triplet_loss
            # else:
            #     ret['triplet_loss'] = triplet_loss
            for ff in [i_feats_proj, si_feats_proj, ci_feats_proj, ni_feats_proj]: #, t_feats, t_feats_cp, t_feats_nir]:  ## text也加入进去 ? TODO
                if ff is not None:
                    triplet_loss = self.triplet(ff, batch['pids'])[0]
                    if 'triplet_loss' in ret:
                        ret['triplet_loss'] += triplet_loss
                    else:
                        ret['triplet_loss'] = triplet_loss

        
        if 'id' in self.current_task:
            """
            id loss: 包括每个图像模态的cls token + inverse pseudo image token
            """
            # rgb_logits = self.classifier(i_feats)
            if self.args.use_new_id_loss:
                rgb_id_loss, rgb_logits_proj = self.cls_head(i_feats_proj, batch['pids'] )
                ret.update({'rgb_loss': rgb_id_loss})
            else:
                rgb_logits_proj = self.classifier_proj(i_feats_proj)
                if epoch in [1, 16]:
                    if epoch not in self.original_logits_dict['rgb_logits']:
                        self.original_logits_dict['rgb_logits'][epoch] = [rgb_logits_proj.detach().cpu().numpy()]
                        self.original_logits_dict['labels'][epoch] = [batch['pids'].detach().cpu().numpy()]
                        # 把原始的特征装载
                        self._feat_dict['nir_features'][epoch] = [i_feats_proj.detach().cpu().numpy()]
                        self._feat_dict['labels'][epoch] = [batch['pids'].detach().cpu().numpy()]
                    else:
                        self.original_logits_dict['rgb_logits'][epoch].append(rgb_logits_proj.detach().cpu().numpy())
                        self.original_logits_dict['labels'][epoch].append(batch['pids'].detach().cpu().numpy())
                        # 把原始的特征装载
                        self._feat_dict['nir_features'][epoch].append(i_feats_proj.detach().cpu().numpy())
                        self._feat_dict['labels'][epoch].append(batch['pids'].detach().cpu().numpy())
                
                rgb_loss, rgb_logits_proj_after = objectives.compute_id_new( rgb_logits_proj, batch['pids'])
                ret.update({'rgb_loss': rgb_loss}) 

                if epoch in [1, 16]:
                    if epoch not in self.after_logits_dict['rgb_logits']:
                        self.after_logits_dict['rgb_logits'][epoch] = [rgb_logits_proj_after.detach().cpu().numpy()]
                        self.after_logits_dict['labels'][epoch] = [batch['pids'].detach().cpu().numpy()]
                    else:
                        self.after_logits_dict['rgb_logits'][epoch].append(rgb_logits_proj_after.detach().cpu().numpy())
                        self.after_logits_dict['labels'][epoch].append(batch['pids'].detach().cpu().numpy())
                # if self.args.use_EMA_model:
                #     with torch.no_grad():
                #         rgb_logits_proj_t = self.classifier_proj(i_feats_proj_t)
                #     kl_loss = self.kl_div(F.log_softmax(rgb_logits_proj / 0.2, dim=1), F.softmax(rgb_logits_proj_t / 0.2, dim=1)) * (0.2 * 0.2)
                #     if 'kl_loss' in ret:
                #         ret['kl_loss'] += kl_loss
                #     else:
                #         ret['kl_loss'] = kl_loss

            # rgb_logits_proj, self.rgb_state = objectives.adaptive_margin_scale(rgb_logits_proj, batch['pids'], state=self.rgb_state)
            # ret.update({'rgb_loss': objectives.compute_id_new( rgb_logits_proj, batch['pids'])})  # 换成普通id loss??? no scale
            self._compute_id_loss_and_acc(None, rgb_logits_proj, batch['pids'], 'rgb_acc', ret)

            # if self.args.use_local_feat:
            #     rgb_logits_proj_local = self.classifier_proj(i_feats_proj_local)
            #     ret.update({'rgb_loss_local': objectives.compute_id_new(rgb_logits_proj_local, batch['pids'])})
            #     self._compute_id_loss_and_acc(None, rgb_logits_proj_local, batch, 'rgb_acc_local', ret)

            if self.args.use_inverse:
                if i_merge_feat_txt is not None:

                    ######### 去掉为0的
                    _mask = (i_merge_feat_txt.sum(dim=1) != 0)
                    i_merge_feat_txt = i_merge_feat_txt[_mask, ...]
                    batch_pid = batch['pids'][_mask, ...]
                    #########

                    acc_num = 1
                    txt_logits = self.classifier_proj(i_merge_feat_txt)
                    ret.update({'txt_loss': objectives.compute_id_new( txt_logits, batch_pid)})
                    txt_pred = torch.argmax(txt_logits, dim=1)
                    txt_precision = (txt_pred == batch_pid).float().mean()
                    ret.update({'txt_acc': txt_precision})

                    # if self.args.use_local_feat:
                    #     txt_logits_local = self.classifier_proj(masked_feats)
                    #     ret.update({'txt_loss_local': objectives.compute_id_new( txt_logits_local, batch['pids'])})
                    #     txt_pred_local = torch.argmax(txt_logits_local, dim=1)
                    #     txt_precision_local = (txt_pred_local == batch['pids']).float().mean()
                    #     ret.update({'txt_acc_local': txt_precision_local})

                    if si_merge_feat_txt is not None:
                        ######### 去掉为0的
                        _mask = (si_merge_feat_txt.sum(dim=1) != 0)
                        si_merge_feat_txt = si_merge_feat_txt[_mask, ...]
                        batch_pid = batch['pids'][_mask, ...]
                        #########
                        # txt_logits_sk = self.classifier_proj(si_feats_inverse_txt)
                        txt_logits_sk_g = self.classifier_proj(si_merge_feat_txt)
                        ret['txt_loss'] += objectives.compute_id_new( txt_logits_sk_g, batch_pid)
                        # txt_precision = (torch.argmax(txt_logits_sk, dim=1) == batch['pids']).float().mean()
                        txt_precision_2 = (torch.argmax(txt_logits_sk_g, dim=1) == batch_pid).float().mean()
                        ret['txt_acc'] += txt_precision_2#(txt_precision + txt_precision_2) / 2
                        acc_num += 1

                    if ni_merge_feat_txt is not None:
                        ######### 去掉为0的
                        _mask = (ni_merge_feat_txt.sum(dim=1) != 0)
                        ni_merge_feat_txt = ni_merge_feat_txt[_mask, ...]
                        batch_pid = batch['pids'][_mask, ...]
                        #########
                        # txt_logits_nir = self.classifier_proj(ni_feats_inverse_txt)
                        txt_logits_nir_g = self.classifier_proj(ni_merge_feat_txt)
                        ret['txt_loss'] += objectives.compute_id_new( txt_logits_nir_g, batch_pid)
                        # txt_precision = (torch.argmax(txt_logits_nir, dim=1) == batch['pids']).float().mean()
                        txt_precision_2 = (torch.argmax(txt_logits_nir_g, dim=1) == batch_pid).float().mean()
                        ret['txt_acc'] += txt_precision_2
                        acc_num += 1

                    if ci_merge_feat_txt is not None:
                        ######### 去掉为0的
                        _mask = (ci_merge_feat_txt.sum(dim=1) != 0)
                        ci_merge_feat_txt = ci_merge_feat_txt[_mask, ...]
                        batch_pid = batch['pids'][_mask, ...]
                        #########
                        # txt_logits_cp = self.classifier_proj(ci_feats_inverse_txt)
                        txt_logits_cp_g = self.classifier_proj(ci_merge_feat_txt)
                        ret['txt_loss'] += objectives.compute_id_new( txt_logits_cp_g, batch_pid)
                        # txt_precision = (torch.argmax(txt_logits_cp, dim=1) == batch['pids']).float().mean()
                        txt_precision_2 = (torch.argmax(txt_logits_cp_g, dim=1) == batch_pid).float().mean()
                        ret['txt_acc'] += txt_precision_2#(txt_precision + txt_precision_2) / 2
                        acc_num += 1

                    ret['txt_acc'] /= acc_num
                    ret['txt_loss'] /= acc_num
            else:
                if t_feats is not None:
                    ######### 去掉为0的
                    _mask = (t_feats.sum(dim=1) != 0)
                    t_feats = t_feats[_mask, ...]
                    batch_pid = batch['pids'][_mask, ...]
                    #########
                    if self.args.use_new_id_loss:
                        txt_id_loss, txt_logits = self.cls_head(t_feats, batch_pid)#, feat_type='txt')
                        ret.update({'txt_loss': txt_id_loss}) 
                    else:
                        if self.args.use_multi_classifier:
                            txt_logits = self.classifier_proj_text(t_feats)
                        else:
                            txt_logits = self.classifier_proj(t_feats)  ### text 要不要加上bn neck ？？？效果不好啊.... TODO: 单独给text设置分类器? 

                        if epoch in [1, 16]:

                            if epoch not in self.original_logits_dict['txt_logits']:
                                self.original_logits_dict['txt_logits'][epoch] = [txt_logits.detach().cpu().numpy()]
                                # 把原始的特征装载
                                self._feat_dict['txt_features'][epoch] = [t_feats.detach().cpu().numpy()]
                            else:
                                self.original_logits_dict['txt_logits'][epoch].append(txt_logits.detach().cpu().numpy())
                                self._feat_dict['txt_features'][epoch].append(t_feats.detach().cpu().numpy())

                        txt_loss, txt_logits_after = objectives.compute_id_new( txt_logits, batch_pid)
                        ret.update({'txt_loss': txt_loss}) 

                        if epoch in [1, 16]:

                            if epoch not in self.after_logits_dict['txt_logits']:
                                self.after_logits_dict['txt_logits'][epoch] = [txt_logits_after.detach().cpu().numpy()]
                            else:
                                self.after_logits_dict['txt_logits'][epoch].append(txt_logits_after.detach().cpu().numpy())
                    
                    txt_pred = torch.argmax(txt_logits, dim=1)
                    txt_precision = (txt_pred == batch_pid).float().mean()
                    ret.update({'txt_acc': txt_precision})


                    ######### 去掉为0的
                    # _mask_dy = (dynamic_mask_t_feats.sum(dim=1) != 0)
                    # dynamic_mask_t_feats = dynamic_mask_t_feats[_mask_dy, ...]
                    # batch_pid_dy = batch['pids'][_mask_dy, ...]

                    # txt_logits_dy = self.classifier_proj(dynamic_mask_t_feats)  ### text 要不要加上bn neck ？？？效果不好啊.... TODO: 单独给text设置分类器? 
                    # ret['txt_loss'] += objectives.compute_id_new( txt_logits_dy, batch_pid_dy)
                    # txt_pred_dy = torch.argmax(txt_logits_dy, dim=1)
                    # txt_precision_dy = (txt_pred_dy == batch_pid_dy).float().mean()
                    # ret['txt_acc'] += txt_precision_dy
                    # ret['txt_acc'] /= 2
                    # ret['txt_loss'] /= 2

            if si_feats_proj is not None:
                # si_logits = self.classifier(si_feats)
                ######### 去掉为0的
                _mask = (si_feats_proj.sum(dim=1) != 0)
                si_feats_proj = si_feats_proj[_mask, ...]
                # si_shared_feat = si_shared_feat[_mask, ...]
                batch_pid = batch['pids'][_mask, ...]
                #########
                if self.args.use_new_id_loss:
                    si_id_loss, si_logits_proj = self.cls_head(si_feats_proj, batch_pid)
                    ret.update({'si_loss': si_id_loss})
                else:
                    if self.args.use_multi_classifier:
                        si_logits_proj = self.classifier_proj_sk(si_feats_proj)
                    else:
                        si_logits_proj = self.classifier_proj(si_feats_proj)
                    if epoch in [1, 16]:
                        if epoch not in self.original_logits_dict['sk_logits']:
                            self.original_logits_dict['sk_logits'][epoch] = [si_logits_proj.detach().cpu().numpy()]
                            self._feat_dict['sk_features'][epoch] = [si_feats_proj.detach().cpu().numpy()]
                        else:
                            self.original_logits_dict['sk_logits'][epoch].append(si_logits_proj.detach().cpu().numpy())
                            self._feat_dict['sk_features'][epoch].append(si_feats_proj.detach().cpu().numpy())

                    si_loss, si_logits_proj_after = objectives.compute_id_new(si_logits_proj, batch_pid)
                    ret.update({'si_loss': si_loss})

                    if epoch in [1, 16]:

                        if epoch not in self.after_logits_dict['sk_logits']:
                            self.after_logits_dict['sk_logits'][epoch] = [si_logits_proj_after.detach().cpu().numpy()]
                        else:
                            self.after_logits_dict['sk_logits'][epoch].append(si_logits_proj_after.detach().cpu().numpy())
                    # if self.args.use_EMA_model:
                    #     with torch.no_grad():
                    #         si_logits_proj_t = self.classifier_proj(si_feats_proj_t)
                    #     kl_loss = self.kl_div(F.log_softmax(si_logits_proj / 0.2, dim=1), F.softmax(si_logits_proj_t / 0.2, dim=1)) * (0.2 * 0.2)
                    #     if 'kl_loss' in ret:
                    #         ret['kl_loss'] += kl_loss
                    #     else:
                    #         ret['kl_loss'] = kl_loss
                # ret.update({'si_loss': objectives.compute_id_new(si_logits_proj, batch_pid) + 0.2 * objectives.compute_id_new(si_logits_proj_shared, batch_pid)})
                self._compute_id_loss_and_acc(None, si_logits_proj, batch_pid, 'si_acc', ret)
                # if self.args.use_local_feat:
                #     if si_feats_proj_local is not None:
                #         si_logits_proj_local = self.classifier_proj(si_feats_proj_local)
                #         ret.update({'si_loss_local': objectives.compute_id_new(si_logits_proj_local, batch['pids'])})
                #         self._compute_id_loss_and_acc(None, si_logits_proj_local, batch, 'si_acc_local', ret)

            if ci_feats_proj is not None:
                ######### 去掉为0的
                _mask = (ci_feats_proj.sum(dim=1) != 0)
                ci_feats_proj = ci_feats_proj[_mask, ...]
                # ci_shared_feat = ci_shared_feat[_mask, ...]
                batch_pid = batch['pids'][_mask, ...]
                #########
                if self.args.use_new_id_loss:
                    ci_id_loss, ci_logits_proj = self.cls_head(ci_feats_proj, batch_pid)
                    ret.update({'ci_loss': ci_id_loss})
                else:
                    if self.args.use_multi_classifier:
                        ci_logits_proj = self.classifier_proj_cp(ci_feats_proj)
                    else:
                        ci_logits_proj = self.classifier_proj(ci_feats_proj)

                    if epoch in [1, 16]:
                        if epoch not in self.original_logits_dict['cp_logits']:
                            self.original_logits_dict['cp_logits'][epoch] = [ci_logits_proj.detach().cpu().numpy()]
                            self._feat_dict['cp_features'][epoch] = [ci_feats_proj.detach().cpu().numpy()]
                        else:
                            self.original_logits_dict['cp_logits'][epoch].append(ci_logits_proj.detach().cpu().numpy())
                            self._feat_dict['cp_features'][epoch].append(ci_feats_proj.detach().cpu().numpy())

                    ci_loss, ci_logits_proj_after = objectives.compute_id_new(ci_logits_proj, batch_pid)
                    ret.update({'ci_loss': ci_loss})

                    if epoch in [1, 16]:
                        if epoch not in self.after_logits_dict['cp_logits']:
                            self.after_logits_dict['cp_logits'][epoch] = [ci_logits_proj_after.detach().cpu().numpy()]
                        else:
                            self.after_logits_dict['cp_logits'][epoch].append(ci_logits_proj_after.detach().cpu().numpy())
                    # if self.args.use_EMA_model:
                    #     with torch.no_grad():
                    #         ci_logits_proj_t = self.classifier_proj(ci_feats_proj_t)
                    #     kl_loss = self.kl_div(F.log_softmax(ci_logits_proj / 0.2, dim=1), F.softmax(ci_logits_proj_t / 0.2, dim=1)) * (0.2 * 0.2)
                    #     if 'kl_loss' in ret:
                    #         ret['kl_loss'] += kl_loss
                    #     else:
                    #         ret['kl_loss'] = kl_loss
                # ci_logits_proj_shared = self.classifier_proj(ci_shared_feat)
                # ci_logits_proj, self.cp_state = objectives.adaptive_margin_scale(ci_logits_proj, batch_pid, state=self.cp_state)
                self._compute_id_loss_and_acc(None, ci_logits_proj, batch_pid, 'ci_acc', ret)
                # if self.args.use_local_feat:
                #     if ci_feats_proj_local is not None:
                #         ci_logits_proj_local = self.classifier_proj(ci_feats_proj_local)
                #         ret.update({'ci_loss_local': objectives.compute_id_new(ci_logits_proj_local, batch['pids'])})
                #         self._compute_id_loss_and_acc(None, ci_logits_proj_local, batch, 'ci_acc_local', ret)

            if ni_feats_proj is not None:
                ######### 去掉为0的
                _mask = (ni_feats_proj.sum(dim=1) != 0)
                ni_feats_proj = ni_feats_proj[_mask, ...]
                # ni_shared_feat = ni_shared_feat[_mask, ...]
                batch_pid = batch['pids'][_mask, ...]
                #########
                if self.args.use_new_id_loss:
                    ni_id_loss, ni_logits_proj = self.cls_head(ni_feats_proj, batch_pid)
                    ret.update({'ni_loss': ni_id_loss})
                else:
                    if self.args.use_multi_classifier:
                        ni_logits_proj = self.classifier_proj_nir(ni_feats_proj)
                    else:
                        ni_logits_proj = self.classifier_proj(ni_feats_proj)

                    if epoch in [1, 16]:
                        if epoch not in self.original_logits_dict['nir_logits']:
                            self.original_logits_dict['nir_logits'][epoch] = [ni_logits_proj.detach().cpu().numpy()]
                        else:
                            self.original_logits_dict['nir_logits'][epoch].append(ni_logits_proj.detach().cpu().numpy())

                    ni_loss, ni_logits_proj_after = objectives.compute_id_new(ni_logits_proj, batch_pid)
                    ret.update({'ni_loss': ni_loss})

                    if epoch in [1, 16]:
                        if epoch not in self.after_logits_dict['nir_logits']:
                            self.after_logits_dict['nir_logits'][epoch] = [ni_logits_proj_after.detach().cpu().numpy()]
                        else:
                            self.after_logits_dict['nir_logits'][epoch].append(ni_logits_proj_after.detach().cpu().numpy())
                    # if self.args.use_EMA_model:
                    #     with torch.no_grad():
                    #         ni_logits_proj_t = self.classifier_proj(ni_feats_proj_t)
                    #     kl_loss = self.kl_div(F.log_softmax(ni_logits_proj / 0.2, dim=1), F.softmax(ni_logits_proj_t / 0.2, dim=1)) * (0.2 * 0.2)
                    #     if 'kl_loss' in ret:
                    #         ret['kl_loss'] += kl_loss
                    #     else:
                    #         ret['kl_loss'] = kl_loss
                # ni_logits_proj_shared = self.classifier_proj(ni_shared_feat)
                # ni_logits_proj, self.nir_state = objectives.adaptive_margin_scale(ni_logits_proj, batch_pid, state=self.nir_state)
                self._compute_id_loss_and_acc(None, ni_logits_proj, batch_pid, 'ni_acc', ret)
                # if self.args.use_local_feat:
                #     if ni_feats_proj_local is not None:
                #         ni_logits_proj_local = self.classifier_proj(ni_feats_proj_local)
                #         ret.update({'ni_loss_local': objectives.compute_id_new(ni_logits_proj_local, batch['pids'])})
                #         self._compute_id_loss_and_acc(None, ni_logits_proj_local, batch, 'ni_acc_local', ret)

        if 'mlm' in self.current_task:
            num_mlm = 0
            for idx_2 , img_f in enumerate([image_features_rgb_proj, image_features_sk_proj, image_features_cp_proj, image_features_nir_proj]):  ### 0907: rgb只和 mask 掉 color 的 text 计算; 其余的
                if img_f is not None:
                    ret = self._compute_multi_mlm_loss(text_features, img_f, batch, ret, self.cross_former, self.mlm_head, 'mlm_labels')  ### 除了color 没有被mask, 其余的token 被随机 masked
                    num_mlm += 1

            ret['mlm_acc'] /= num_mlm

        return ret

    def _compute_id_loss_and_acc(self, logits, logits_proj, batch_pids, name, ret):
        # rgb_pred = torch.argmax(logits, dim=1)
        rgb_pred_proj = torch.argmax(logits_proj, dim=1)
        # rgb_precision = (rgb_pred == batch['pids']).float().mean()
        rgb_precision_proj = (rgb_pred_proj == batch_pids).float().mean()
        ret.update({name: rgb_precision_proj})

    
    def _compute_multi_mlm_loss(self, mlm_feats, image_feats, batch, ret, cross_former, mlm_head, label_str_type, param_weight=1.0):
        # 
        """注意这里用的是全量feature, 不再只有cls features; 原来所有 Image feature 都共享cross former, mlm head"""
        # 
        nonzero_mask_A = (mlm_feats.abs().sum(dim=(1,2)) > 1e-12)
        nonzero_mask_B = (image_feats.abs().sum(dim=(1,2)) > 1e-12)

        # 2. 同时非零的行
        _mask = nonzero_mask_A & nonzero_mask_B
        if _mask.sum().item() == 0:
            return ret
        mlm_feats = mlm_feats[_mask]
        image_feats = image_feats[_mask]

        x = cross_former(mlm_feats, image_feats, image_feats, self.ln_pre_i, self.ln_pre_t, self.ln_post, self._transformer, self.cross_attn)  # 一个cross attention + transformer block + 残差

        x = mlm_head(x)  # [batch_size, text_len, num_colors]  # 两层全连接层

        scores = x.float().reshape(-1, self.args.vocab_size)  # x resize 为 词表大小
        mlm_labels = batch[label_str_type][_mask].reshape(-1)  # mask的token,没有mask的token是0 ,[token1, token2, 0, 0, .., tokeni, ...], label
        
        # 计算一个交叉熵损失: label中只有mask掉的token才有loss
        if 'mlm_loss' in ret:
            ret['mlm_loss'] +=  objectives.compute_mcm_or_mlm(scores, mlm_labels) * param_weight ###*self.args.mlm_loss_weight
        else:
            ret.update({'mlm_loss': objectives.compute_mcm_or_mlm(scores, mlm_labels) * param_weight })

        pred = scores.max(1)[1]
        mlm_label_idx = torch.nonzero(mlm_labels)
        acc = (pred[mlm_label_idx] == mlm_labels[mlm_label_idx]).float().mean()

        if 'mlm_acc' in ret:
            ret['mlm_acc'] += acc
        else:
            ret.update({'mlm_acc': acc})
        return ret

def build_model(args, num_classes=1103):
    model = CLIP2ReID(args, num_classes)
    # covert model to fp16
    convert_weights(model.base_model)   ### 11.06: 只有CLIP是float 16
    # if getattr(args, "use_EMA_model", False):
    #     convert_weights(model.teacher_base_model)
    # convert_weights(model._transformer)
    # convert_weights(model.cross_attn)
    # convert_weights(model.mlm_head)
    return model
