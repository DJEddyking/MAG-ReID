import argparse


def get_args():
    parser = argparse.ArgumentParser(description="TransTextReID")
    ######################## general settings ########################
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--gpu_id", default='2', type=str)
    parser.add_argument("--name", default="no_change_with_margin_scale_with_single_lora_trainable_only_0_6_with_tokenizer_no_EMA_exp_tau_5_lora_slover_", help="experiment name to save")
    # parser.add_argument("--log_dir", default="logs")
    parser.add_argument("--select_ratio", default=0.3, type=float)
    parser.add_argument("--output_dir", default="logs")
    # parser.add_argument("--output_dir", default="/root/autodl-tmp/original_code备份_last/outputs")
    # parser.add_argument("--trained_model_path", type=str, default="/root/autodl-tmp/original_code备份_last/longclip-L.pt")
    # parser.add_argument("--gpu_id", default="0", help="select gpu to run")
    parser.add_argument("--log_period", default=100, type=int)  ### 100, RGBNT201: 50; PKU-Sketch:1
    parser.add_argument("--eval_period", default=30, type=int) ##### 1
    parser.add_argument("--val_dataset", default="test") # use val set when evaluate, if test use test set  #### test

    parser.add_argument("--resume", default=False, action='store_true')
    parser.add_argument("--resume_ckpt_file", default="", help='resume from ...')

    parser.add_argument("--rerank", default=False, type=bool, help='whether use reranking')

    # new eval
    parser.add_argument("--new_eval", default=False, type=bool, help="") 
    # mask weight
    parser.add_argument("--mask_weight", default=False, type=bool, help="") 
    # 是否用局部特征
    parser.add_argument("--use_local_feat", default=False, type=bool, help="") 

    parser.add_argument("--use_prompt", default=False, type=bool, help="whether to use prompt")  # 是否使用prompt, 已作废, 暂时不加
    parser.add_argument("--use_inverse", default=False, type=bool, help="whether to use inverseNet") # 是否使用inversenet
    parser.add_argument("--data_use_inverse", default=False, type=bool, help="")
    parser.add_argument("--use_conv_inverse", default=False, type=bool, help="") # 是否使用inversenet => 针对图像的
    parser.add_argument("--use_IN", default=False, type=bool, help="whether to use inverseNet")  # 是否在image encoder的tokenizer中conv2d后面加入IN层
    parser.add_argument("--use_rgb_inverse", default=False, type=bool, help="whether to use rgb feat and text to inversenet")  # 原始text是否和rgb特征拼接送入encoder获得pseudo image token, 再和原text eos token融合
    # 是否使用带有lora和adapter的transformer作为 image encoder
    parser.add_argument("--img_lora_adapter", default=False, type=bool, help="whether to use lora_adapter")  #####
    # parser.add_argument("--shared_B", default=True, type=bool, help="")  #####

    parser.add_argument("--txt_lora_adapter", default=False, type=bool, help="")  ###
    parser.add_argument("--mod_prompts", default=False, type=bool, help="")
    parser.add_argument("--multi_lora", default=False, type=bool, help="")
    parser.add_argument("--idx_list", default=[0, 6], type=list, help="")  # [0, 6]
    parser.add_argument("--add_shared_token", default=False, type=bool, help="")

    parser.add_argument("--use_EMA_model", default=False, type=bool, help="") # 由于训练是随机混合, 因此尝试ema model  => 进阶: 用teacher model的结果约束student model
    parser.add_argument("--use_new_id_loss", default=False, type=bool, help="")

    parser.add_argument("--use_mix_lora_adapter", default=False, type=bool, help="")  ### qkv采用并联adapter; ffn采用并联lora

    parser.add_argument("--use_adapter", default=False, type=bool, help="whether to use real Adapter instead of image lora_adapter")
    parser.add_argument("--adapter_type", default="parallel", choices=["parallel", "sequential"], help="parallel or sequential Adapter in ViT FFN")
    parser.add_argument("--adapter_layers", default=[0, 6], type=int, nargs='+', help="ViT adapter layers, [start end) by default")
    parser.add_argument("--freeze_vit_single_adapter_base", default=False, type=lambda x: str(x).lower() in ("true", "1", "yes"), help="use one shared ViT Adapter and freeze the base FFN when the Adapter is parallel")
    
    parser.add_argument("--use_vit_single_lora", default=True, type=lambda x: str(x).lower() in ("true", "1", "yes"), help="add one shared parallel LoRA to each ViT Linear layer")
    parser.add_argument("--vit_single_lora_layers", default=6, type=int, nargs='+', help="ViT single LoRA layers: N means [0, N), start end means [start, end); default is all layers")
    parser.add_argument("--freeze_vit_single_lora_base", default=False, type=lambda x: str(x).lower() in ("true", "1", "yes"), help="freeze original ViT Linear weights when use_vit_single_lora is enabled")
    
    parser.add_argument("--use_vit_moe", default=False, nargs='?', const=True, type=lambda x: str(x).lower() in ("true", "1", "yes"), help="replace Linear layers in the first N ViT blocks with MoE Linear layers")
    parser.add_argument("--freeze_vit_for_moe", default=False, type=lambda x: str(x).lower() in ("true", "1", "yes"), help="freeze ViT base parameters and train only MoE expert/gate parameters when use_vit_moe is enabled")
    parser.add_argument("--vit_moe_num_experts", default=6, type=int, help="number of ViT MoE experts")
    parser.add_argument("--vit_moe_top_k", default=2, type=int, help="top-K experts selected by the ViT MoE gate")
    parser.add_argument("--vit_moe_layers", default=6, type=int, help="replace Linear layers in the first N ViT transformer blocks")

    parser.add_argument("--use_multi_classifier", default=False, type=bool, help="")

    # parser.add_argument("--rnt_eval_mode", default='concat', type=str, help="")  ### 评测指标选择方式

    # 是否使用带有lora和adapter的transformer作为 text encoder => 由于随机mask了text以此获得不同图像模态的pseudo image token, 因此text encoder也可以用expert

    # parser.add_argument(
    #     "--inverse_use_prompt_tokens",
    #     default=False,
    #     type=bool,
    #     help="whether inverseNet uses fixed prompt token positions in addition to the global token",
    # )

    ######################## model general settings ########################
    parser.add_argument("--pretrain_choice", default='ViT-B/16') # whether use pretrained model
    parser.add_argument("--temperature", type=float, default=0.02, help="initial temperature value, if 0, don't use temperature")## 0.07


    parser.add_argument("--img_aug", default=True, action='store_true')  ####### 加上 image transform, 包括: pad,水平翻转, 随机擦除,随机裁剪
    # parser.add_argument("--nlp_aug", default=False, action='store_true')  ######## 这个是和数据集有关, 加载nlp_aug.json进行操作的
    # parser.add_argument("--embed_dim", type=int, default=512, help="the final visual and textual feature dim")
    # parser.add_argument("--sampling_timesteps", type=int, default=10, help="ddim steps for training-time diffusion sampling") # 一大步的time
    # parser.add_argument("--train_timesteps", type=int, default=100, help="training timesteps for DDPM, default 1000")
    # parser.add_argument("--inference_step", type=int, default=100, help="inference steps for deterministic refinement or q-sample bridge")

    # parser.add_argument("--denoise_eval", type=bool, default=False, help="whether to apply denoising at evaluation for both gallery and queries")
    # parser.add_argument("--eval_mix", type=float, default=0.2, help="blending weight for evaluation denoised features")
    # parser.add_argument("--adaptive_mix", type=bool, default=True, help="use cosine-adaptive blending between denoised and original features")
    # parser.add_argument("--adaptive_mix_beta", type=float, default=1.0, help="exponent for cosine in adaptive blending")

    # cond_layers 仅对前N个transformer block的LN层进行操作
    # parser.add_argument("--cond_layers", type=int, default=2, help="condition layers for non-rgb modalities") ### 默认是 2 
    # parser.add_argument("--shared_layers", type=int, default=[1, 12], help="shared LN start and end transformer block") ### 默认是 6 

    # parser.add_argument("--freeze_vision_encoder", default=False, type=bool)
    # parser.add_argument("--add_lora", default=True, type=bool)
    # parser.add_argument("--lora_r", type=int,default=4)
    # parser.add_argument("--num_loras", type=int, default=4)   ### 这里默认是 4 ,但是后面是 16 ??
    # parser.add_argument("--lora_layers", type=int, default=2)
    # parser.add_argument("--lora_mode", type=str, default='all')  ### 是替换掉 vit 的全部全连接层还是只有 ffn的
    parser.add_argument('--lora_rank', default=4, type=int, help='')  ##### 4
    parser.add_argument('--lora_alpha', default=4, type=int, help='')

    ## cross transfomer setting
    # parser.add_argument("--num_colors", type=int, default=60, help="num colors of Mask Color Modeling labels")
    parser.add_argument("--cmt_depth", type=int, default=4, help="cross modal transformer self attn layers")
    # parser.add_argument("--masked_token_rate", type=float, default=0.8, help="masked token rate for mcm task, 1.0 indicates mask every color in a caption")
    # parser.add_argument("--masked_token_unchanged_rate", type=float, default=0.1, help="masked token unchanged rate")
    parser.add_argument("--lr_factor", type=float, default=5.0, help="lr factor for random init self implement module")
    # parser.add_argument("--use_imageid", default=False, action='store_true', help="whether to use image_id info to build soft label.")
    # parser.add_argument("--MCQ", default=False, action='store_true', help="whether to use Multiple Choice Questions dataset")
    # parser.add_argument("--MCM", default=False, action='store_true', help="whether to use Mask Color Modeling dataset")

    #### mlm mask text进行训练
    # parser.add_argument("--MLM", default=False, action='store_true', help="whether to use Mask Language Modeling dataset")

    # parser.add_argument("--MSM", default=False, action='store_true', help="whether to use Mask Subsequence Matching dataset")
    # parser.add_argument("--MCQMLM", default=False, action='store_true', help="whether to use MCQMLM dataset")
    # parser.add_argument("--MSMMLM", default=False, action='store_true', help="whether to use MSMMLM dataset")
    ##### 每个图像模态采用单独的adapter MLP: from adapterFormer
    # parser.add_argument('--ffn_adapt', default=True, action='store_true', help='whether activate AdaptFormer')
    # parser.add_argument('--ffn_num', default=128, type=int, help='bottleneck middle dimension')  #### 64
    # parser.add_argument('--ffn_option', default='parallel', type=str, help='parallel or sequential to add adapterMLP')  ### 暂时没用
    # parser.add_argument('--ffn_adapter_layernorm_option', default='in', type=str, help='layer normalization position, in or out')
    # parser.add_argument('--ffn_adapter_init_option', default='lora', type=str, help='adapter mlp initilizaition, lora or bert')
    # parser.add_argument('--ffn_adapter_scalar', default='0.1', type=str, 
    #                     help='final output of one transformer block = ffn_adapter_scalar * adaptermlp_output + original_mlp_ouput,' \
    #                     'it can also be set to: learnable_scalar')  # '0.1'

    # parser.add_argument("--mse_target", type=float, default=0.1, help="target mse loss weight after warmup") ## 更保守，避免抹平
    # parser.add_argument("--mix_target", type=float, default=0.15, help="target denoise mix after warmup") # 更小的去噪混合比例
    # parser.add_argument("--mse_warmup_epochs", type=int, default=20, help="epochs to warmup mse loss weight from 0 to target")
    # parser.add_argument("--mix_warmup_epochs", type=int, default=20, help="epochs to warmup denoise mix from 0 to target")


    ######################## loss settings ########################
    parser.add_argument("--loss_names", default='id+sdm+sup_cons', help="which loss to use ['mlm', 'sup_cons', 'id', 'itc', 'sdm', 'triplet']")  # itc默认就有
    # itc -> id+sdm+mlm
    # focal_three_fusion_loss3为true了，默认有itc loss

    parser.add_argument("--cmm_loss_weight", type=float, default=1.0, help="cross modal matching loss (tcmpm, cmpm, infonce...) weight")
    parser.add_argument("--mcm_loss_weight", type=float, default=1.0, help="mcm loss weight")
    parser.add_argument("--mlm_loss_weight", type=float, default=1.0, help="mlm loss weight")
    parser.add_argument("--mcq_loss_weight", type=float, default=1.0, help="mcq loss weight")
    parser.add_argument("--id_loss_weight", type=float, default=1.0, help="id loss weight")
    
    ######################## vison trainsformer settings ########################
    parser.add_argument("--img_size", type=tuple, default=(384, 128))  ############ (384, 128)
    parser.add_argument("--stride_size", type=int, default=16)

    ######################## text transformer settings ########################


    parser.add_argument("--text_length", type=int, default=77)  ##################  77
    parser.add_argument("--vocab_size", type=int, default=49408)

    ######################## solver ########################
    parser.add_argument("--learnable_loss_weight", default=False)
    parser.add_argument("--label_mix", default=False, action='store_true', help="whether mix pid and imagid label")
    parser.add_argument("--optimizer", type=str, default="Adam", help="[SGD, Adam, AdamW]")
    parser.add_argument("--lr", type=float, default=1e-5)                             ############## 1e-5
    parser.add_argument("--bias_lr_factor", type=float, default=2.)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=4e-5)
    parser.add_argument("--weight_decay_bias", type=float, default=0.)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--beta", type=float, default=0.999)
    
    ######################## scheduler ########################
    parser.add_argument("--num_epoch", type=int, default=60)  ##########  60
    parser.add_argument("--milestones", type=int, nargs='+', default=(20, 50))  #### 20, 50
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--warmup_factor", type=float, default=0.1)
    parser.add_argument("--warmup_epochs", type=int, default=5)     ##### 5
    parser.add_argument("--warmup_method", type=str, default="linear")
    parser.add_argument("--lrscheduler", type=str, default="step")
    parser.add_argument("--target_lr", type=float, default=1e-8)   ###### 1e-8
    parser.add_argument("--power", type=float, default=0.9)
    # parser.add_argument("--cond_ema_decay", type=float, default=0.999, help="EMA decay for condition embedding inside Unet1D")

    ######################## dataset ########################
    parser.add_argument("--dataset_name", default="ORBench", help="[ORBench, CUHK-PEDES, ICFG-PEDES, RSTPReid, RGBNT201, RGBNT201_Text, PKU-Sketch]")  #### ORBench
    parser.add_argument("--modality_num", default=4, type=int, help="modality number of each datasets")  #### 4


    parser.add_argument("--sampler", default="random", help="[identity, random]")  ##### 如果要triplet loss, 就必须要用PK采样
    parser.add_argument("--num_instance", type=int, default=4)
    
    parser.add_argument("--root_dir", type=str, default="/data/dj_2025")
    parser.add_argument("--batch_size", type=int, default=32)   ##### 64
    parser.add_argument("--test_batch_size", type=int, default=128)  #### 512
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--test", dest='training', default=True, action='store_false') # whether in training mode
    # parser.add_argument("--test_setting", type=int, default=0)

    ######################## multi-modality model settings ########################
    # parser.add_argument("--fusion_way", default='dit', help="[add, weight add, cross attention]") # whether use text and sketch fusion method
    
    
    # parser.add_argument("--only_sketch", default=False, action='store_true', help="whether training with only sketch")
    # parser.add_argument("--only_text", default=False, action='store_true', help="whether training with only text")
    parser.add_argument("--pa", type=float, default=0.1, help="parameter add for fusion")
    # parser.add_argument("--only_fusion_loss", default=False, action='store_true', help="whether training with only text")
    parser.add_argument("--al", type=float, default=1.0, help="parameter add for fusion")
    parser.add_argument("--ga", type=float, default=2.0, help="parameter add for fusion")
    parser.add_argument("--klp", type=float, default=1.0, help="parameter add for fusion")

    args = parser.parse_args()
    # if args.dataset_name == "ORBench":
    #     args.test_batch_size = 512

    return args
