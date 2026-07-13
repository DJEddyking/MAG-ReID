import os
import os.path as op
import torch
import numpy as np
import random
import time

from datasets import build_dataloader, build_dataloader_Tri_reid, build_dataloader_rnt
from processor.processor import do_train
from utils.checkpoint import Checkpointer
from utils.iotools import save_train_configs
from utils.logger import setup_logger
from solver import build_optimizer, build_lr_scheduler
from model import build_model, EMA


# from model.build_reid5o import build_model


from utils.metrics import Evaluator

# from utils.metrics_reid5o import Evaluator
# from utils.metrics_prcv import Evaluator
from utils.options import get_args
from utils.comm import get_rank, synchronize


# def get_test_data(args):
#     test_img_loader, test_text_loader, test_sketch_loader, test_color_pencil_loader, test_nir_loader,\
#     test_text_sk_loader, test_text_cp_loader, test_text_nir_loader, test_sk_cp_loader, test_sk_nir_loader, test_cp_nir_loader,\
#     test_text_cp_sk_loader, test_text_cp_nir_loader, test_text_sk_nir_loader, test_cp_sk_nir_loader,\
#     test_text_cp_sk_nir_loader = build_dataloader(args, training=False)
#     return test_img_loader, test_text_loader, test_sketch_loader, test_color_pencil_loader, test_nir_loader,\
#             test_text_sk_loader, test_text_cp_loader, test_text_nir_loader, test_sk_cp_loader, test_sk_nir_loader, test_cp_nir_loader,\
#             test_text_cp_sk_loader, test_text_cp_nir_loader, test_text_sk_nir_loader, test_cp_sk_nir_loader,\
#             test_text_cp_sk_nir_loader

import torch
from collections import defaultdict

def format_num(x, unit):
    if unit == 'M':
        return f"{x/1e6:.3f}M"
    if unit == 'G':
        return f"{x/1e9:.3f}G"
    return str(x)

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

# thop 示例（若使用 thop）
def profile_with_thop(model, input_size, device='cuda'):
    try:
        from thop import profile
    except ImportError:
        raise ImportError("请先安装 thop: pip install thop")
    model.eval()
    dummy = torch.randn(*input_size).to(device)
    with torch.no_grad():
        macs, _ = profile(model, inputs=(dummy,), verbose=False)
    # thop 返回 MACs（multiply-accumulate），若需 FLOPs 通常乘以 2
    flops = macs * 2
    return macs, flops

# ptflops 示例（若使用 ptflops）
def profile_with_ptflops(model, input_res):
    try:
        from ptflops import get_model_complexity_info
    except ImportError:
        raise ImportError("请先安装 ptflops: pip install ptflops")
    model.eval()
    macs, params = get_model_complexity_info(model, input_res, as_strings=False, print_per_layer_stat=False)
    flops = macs * 2
    return macs, flops

# 综合封装
# 注意: 这个会改变随机种子，所以正常训练，要把profile model注释掉 !!!!
def profile_model(model, logger, input_size=(1,3,224,224), device='cuda', flops_tool='thop'):
    model = model.to(device)
    total, trainable = count_params(model)
    was_training = model.training
    original_hooks = {
        module: (
            module._forward_hooks.copy(),
            module._forward_pre_hooks.copy(),
        )
        for module in model.modules()
    }

    def clean_thop_state(profiled_model):
        for module in model.modules():
            forward_hooks, forward_pre_hooks = original_hooks[module]
            module._forward_hooks = forward_hooks
            module._forward_pre_hooks = forward_pre_hooks
            if hasattr(module, 'total_ops'):
                delattr(module, 'total_ops')
            if hasattr(module, 'total_params'):
                delattr(module, 'total_params')

    try:
        profiled_model = model
        if hasattr(model, 'base_model'):
            class BaseModelWrapper(torch.nn.Module):
                def __init__(self, base_model, text_length):
                    super().__init__()
                    self.base_model = base_model
                    self.text_length = text_length

                def forward(self, x):
                    batch_size = x.shape[0]
                    caption_ids = torch.zeros(
                        batch_size, self.text_length, dtype=torch.long, device=x.device
                    )
                    caption_ids[:, -1] = 1
                    return self.base_model(
                        [x, x, x, x],
                        caption_ids,
                        combs_select=[True, True, True, True],
                    )

            profiled_model = BaseModelWrapper(
                model.base_model, getattr(model, 'text_length', 77)
            ).to(device)

        if flops_tool == 'thop':
            macs, flops = profile_with_thop(profiled_model, input_size, device=device)
        else:
            # ptflops expects (C,H,W)
            macs, flops = profile_with_ptflops(profiled_model, input_size[1:])
    finally:
        clean_thop_state(model)
        if was_training:
            model.train()

    # 输出
    logger.info("=== Model Profile ===")
    logger.info(f"Params total: {format_num(total, 'M')}, trainable: {format_num(trainable, 'M')}")
    logger.info(f"MACs: {format_num(macs, 'G')} (MACs), FLOPs: {format_num(flops, 'G')} (FLOPs)")
    # 按模块分组统计（可选）
    # group = defaultdict(int)
    # for name, p in model.named_parameters():
    #     key = 'lora' if 'lora' in name.lower() else ('parallel' if 'parallel' in name.lower() else 'backbone')
    #     group[key] += p.numel()
    # for k, v in group.items():
    #     logger.info(f"{k} params: {format_num(v, 'M')}")
    return {
        'total_params': total,
        'trainable_params': trainable,
        'macs': macs,
        'flops': flops,
        # 'group': dict(group)
    }



def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    args = get_args()
    set_seed(1+get_rank())
    name = args.name

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()
    
    device = "cuda:"+args.gpu_id
    cur_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    args.output_dir = op.join(args.output_dir, args.dataset_name, f'{cur_time}_{name}')
    # args.log_dir = op.join(args.output_dir, args.log_dir)
    logger = setup_logger('ORBench', save_dir=args.output_dir, if_train=args.training, distributed_rank=get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(str(args).replace(',', '\n'))
    # logger.info("Training only sketch {}".format(args.only_sketch))
    # logger.info("Using only text {}".format(args.only_text))
    # logger.info("Using only fusion {}".format(args.only_fusion_loss))
    # logger.info("Using {} fusion method".format(args.fusion_way))

    save_train_configs(args.output_dir, args)
    num_query = None
    if args.dataset_name == 'ORBench':
        train_loader, \
        test_gallery_loader, \
        nir_query_loader, \
        cp_query_loader, \
        sk_query_loader, \
        text_query_loader, \
        cp_nir_query_loader, \
        sk_nir_query_loader, \
        text_nir_query_loader, \
        sk_cp_query_loader, \
        text_cp_query_loader, \
        text_sk_query_loader, \
        cp_sk_nir_query_loader, \
        text_cp_nir_query_loader, \
        text_sk_nir_query_loader, \
        text_cp_sk_query_loader, \
        text_cp_sk_nir_query_loader, num_classes = build_dataloader(args)
    elif args.dataset_name in ['CUHK-PEDES', 'ICFG-PEDES', 'RSTPReid', 'PKU-Sketch']:
        train_loader, test_gallery_loader, text_query_loader, sk_query_loader, text_sk_query_loader, num_classes = build_dataloader_Tri_reid(args)
        nir_query_loader, cp_query_loader, cp_nir_query_loader, sk_nir_query_loader, text_nir_query_loader, sk_cp_query_loader, text_cp_query_loader, \
            cp_sk_nir_query_loader, text_cp_nir_query_loader, text_sk_nir_query_loader, text_cp_sk_query_loader, text_cp_sk_nir_query_loader = None, None, None, None, None, None, None, None, None, None, None, None
    elif args.dataset_name in ['RGBNT201', 'RGBNT201_Text']:
        # build_dataloader_rnt returns: train_loader, val_query_loader, val_gallery_loader, num_classes, num_query
        train_loader, val_query_loader, val_gallery_loader, num_classes, num_query = build_dataloader_rnt(args)
        test_gallery_loader = val_gallery_loader
        nir_query_loader = val_query_loader
        sk_query_loader, text_sk_query_loader = None, None
        text_query_loader, cp_query_loader, cp_nir_query_loader, sk_nir_query_loader, text_nir_query_loader, sk_cp_query_loader, text_cp_query_loader, \
            cp_sk_nir_query_loader, text_cp_nir_query_loader, text_sk_nir_query_loader, text_cp_sk_query_loader, text_cp_sk_nir_query_loader = None, None, None, None, None, None, None, None, None, None, None, None

    print('num class', num_classes)
    model = build_model(args, num_classes)
    model.to(device)

    # if args.distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(
    #         model,
    #         device_ids=[args.local_rank],
    #         output_device=args.local_rank,
    #         # this should be removed if we update BatchNorm stats
    #         broadcast_buffers=False,
    #     )
    optimizer = build_optimizer(args, model)
    scheduler = build_lr_scheduler(args, optimizer)

    is_master = get_rank() == 0
    checkpointer = Checkpointer(model, optimizer, scheduler, args.output_dir, is_master)
    # evaluator = Evaluator(args, val_img_loader, val_txt_loader, val_sketch_loader)
    # evaluator = Evaluator(args, 
    #                       val_img_loader, val_text_loader, val_sketch_loader, val_color_pencil_loader, val_nir_loader,\
    #                         val_text_sk_loader, val_text_cp_loader, val_text_nir_loader, val_sk_cp_loader, val_sk_nir_loader, val_cp_nir_loader,\
    #                         val_text_cp_sk_loader, val_text_cp_nir_loader, val_text_sk_nir_loader, val_cp_sk_nir_loader,\
    #                         val_text_cp_sk_nir_loader)

    # 统计模型参数
    profile_model(model, logger, input_size=(1,3,384,128), device=device, flops_tool='thop')

    # reid5o的评测方式
    evaluator = Evaluator(
        args,
        gallery_loader=test_gallery_loader,
        get_mAP=True,  ## 这个计算会比较耗时, reid5o方法把它置为了False
        num_query=num_query,  ### RGBNT201数据集存在, 默认是None
        # 单模态
        nir_query_loader=nir_query_loader,
        cp_query_loader=cp_query_loader,
        sk_query_loader=sk_query_loader,
        text_query_loader=text_query_loader,
        # 双模态
        text_sk_query_loader=text_sk_query_loader,
        text_cp_query_loader=text_cp_query_loader,
        text_nir_query_loader=text_nir_query_loader,
        sk_cp_query_loader=sk_cp_query_loader,
        sk_nir_query_loader=sk_nir_query_loader,
        cp_nir_query_loader=cp_nir_query_loader,
        # 三模态
        text_cp_sk_query_loader=text_cp_sk_query_loader,
        text_cp_nir_query_loader=text_cp_nir_query_loader,
        text_sk_nir_query_loader=text_sk_nir_query_loader,
        cp_sk_nir_query_loader=cp_sk_nir_query_loader,
        # 四模态
        text_cp_sk_nir_query_loader=text_cp_sk_nir_query_loader,
    )

    start_epoch = 1

    if args.resume:
        checkpoint = checkpointer.resume(args.resume_ckpt_file) #args.resume_ckpt_file
        start_epoch = checkpoint['epoch']
    #     # onemodal_top1, twomodal_top1, threemodal_top1, fourmodal_top1 = evaluator.eval(model.eval())
    #     # 加载最优权重模型做tsne可视化
    #     ckpt = torch.load('/home/dj_2025/competitions/original_code/outputs/ORBench_PRCV/20250909_221756_5_itc_5_sdm_4_mlm_5_id_random_train_img_aug_moe_shared_tf_rank_4_alpha_4_ffn_128_retain_mlm_text_LongCLIP_bs_16/average_best.pth')
    #     model.load_state_dict(ckpt['model'])
    #     __train_loader = build_dataloader(args, combs_select=[1,1,1,1], train_only=True)

    do_train(start_epoch, args, model, train_loader, evaluator, optimizer, scheduler, checkpointer)
