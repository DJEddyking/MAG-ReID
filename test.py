from prettytable import PrettyTable
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
import torch.nn.parallel
import numpy as np
import pandas as pd
import time
import os.path as op

import warnings
warnings.filterwarnings("ignore")

# import clip
from datasets import build_dataloader, build_dataloader_rnt, build_dataloader_Tri_reid
from processor.processor import do_inference, do_tsne_visualization
from utils.checkpoint import Checkpointer
from utils.logger import setup_logger
from model import build_model
from utils.metrics import Evaluator
import argparse
from utils.iotools import load_train_configs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LMR-OM-ReID testing")
    parser.add_argument("--configs_path",
                         default='/home/dj_2025/competitions/original_code备份_last_H20/logs/ORBench/20260108_235508_no_change_with_margin_scale_with_lora_with_tokenizer_with_EMA_exp_tau_5_lora_slover_random_choose——best')
    parser.add_argument("--test_gpu_id", default=2, type=int, help='only support single gpu for testing')
    parser.add_argument("--test_batch_size", default=512, type=int)

    test_args = parser.parse_args()
    args = load_train_configs(os.path.join(test_args.configs_path, 'configs.yaml'))
 
    args.test_batch_size = test_args.test_batch_size
    # args.training = False
    #############################
    # baseline方法要去掉几个创新点
    # args.img_lora_adapter = False
    # args.use_EMA_model = False
    # args.use_multi_classifier = False
    # args.loss_names = 'id'
    # args.output_dir = '/home/dj_2025/TIPR/4卡备份_1216_baseline方法/logs/ORBench/20251216_110919_sdm_id_loss_baseline_temp_0_02_data_aug_only_txt_0_07'
    ###############################

    logger = setup_logger('ORBench', save_dir=args.output_dir, if_train=args.training)
    logger.info(test_args)
    device = torch.device(f"cuda:{test_args.test_gpu_id}")
    
    # build loader与train保持一致
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
    # elif args.dataset_name in ['CUHK-PEDES', 'ICFG-PEDES', 'RSTPReid']:
    #     train_loader, test_gallery_loader, text_query_loader, sk_query_loader, text_sk_query_loader, num_classes = build_dataloader_Tri_reid(args)
    #     nir_query_loader, cp_query_loader, cp_nir_query_loader, sk_nir_query_loader, text_nir_query_loader, sk_cp_query_loader, text_cp_query_loader, \
    #         cp_sk_nir_query_loader, text_cp_nir_query_loader, text_sk_nir_query_loader, text_cp_sk_query_loader, text_cp_sk_nir_query_loader = None, None, None, None, None, None, None, None, None, None, None, None
    # elif args.dataset_name in ['RGBNT201', 'RGBNT201_Text']:
    #     # build_dataloader_rnt returns: train_loader, val_query_loader, val_gallery_loader, num_classes, num_query
    #     train_loader, val_query_loader, val_gallery_loader, num_classes, num_query = build_dataloader_rnt(args)
    #     test_gallery_loader = val_gallery_loader
    #     nir_query_loader = val_query_loader
    #     sk_query_loader, text_sk_query_loader = None, None
    #     text_query_loader, cp_query_loader, cp_nir_query_loader, sk_nir_query_loader, text_nir_query_loader, sk_cp_query_loader, text_cp_query_loader, \
    #         cp_sk_nir_query_loader, text_cp_nir_query_loader, text_sk_nir_query_loader, text_cp_sk_query_loader, text_cp_sk_nir_query_loader = None, None, None, None, None, None, None, None, None, None, None, None

    print('num class', num_classes)
    model = build_model(args, num_classes)
    # model.to(device)
    
    # test_img_loader, test_text_loader, test_sketch_loader, test_color_pencil_loader, test_nir_loader,\
    # test_text_sk_loader, test_text_cp_loader, test_text_nir_loader, test_sk_cp_loader, test_sk_nir_loader, test_cp_nir_loader,\
    # test_text_cp_sk_loader, test_text_cp_nir_loader, test_text_sk_nir_loader, test_cp_sk_nir_loader,\
    # test_text_cp_sk_nir_loader = build_dataloader(args, training=False)

    # model = build_model(args, num_classes=600)  ## 固定为训练集
    checkpointer = Checkpointer(model, logger=logger)
    checkpointer.load(f=op.join(test_args.configs_path, 'average_best.pth'))
    model.to(device)

    do_inference(
        args, 
        model, 
        num_query,

        test_gallery_loader,
        nir_query_loader,
        cp_query_loader,
        sk_query_loader,
        text_query_loader,
        cp_nir_query_loader,
        sk_nir_query_loader,
        text_nir_query_loader,
        sk_cp_query_loader,
        text_cp_query_loader,
        text_sk_query_loader,
        cp_sk_nir_query_loader,
        text_cp_nir_query_loader,
        text_sk_nir_query_loader,
        text_cp_sk_query_loader,
        text_cp_sk_nir_query_loader
    )
    
    # tsne可视化
    # do_tsne_visualization(device, model, train_loader)
