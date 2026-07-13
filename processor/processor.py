import logging
import time
import torch
import math
from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.runtime_metrics import log_training_time_per_epoch
# from utils.metrics_autodl import Evaluator
from utils.comm import get_rank, synchronize
from torch.utils.tensorboard import SummaryWriter
from prettytable import PrettyTable
from datasets import build_dataloader
import pandas as pd
import numpy as np
import itertools, random, copy
from collections import defaultdict
from model.utils import InputImageType
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os



NUM_MODALITIES = 4  # 模态数量（示例为4）
GROUP_PROBS = {
    1: 0.6,  # 只有一个1的组合总体概率
    2: 0.2,  # 两个1的组合总体概率
    3: 0.1,  # 三个1的组合总体概率
    4: 0.1   # 四个1（全1）组合总体概率
}
# ---------- 生成所有非全零组合并按 popcount 分组 ----------
def generate_combinations(num_modalities):
    combos = []
    for bits in itertools.product([0, 1], repeat=num_modalities):
        if sum(bits) == 0:
            continue  # 排除全0
        combos.append(tuple(bits))
    return combos

def group_combinations_by_count(combos):
    groups = defaultdict(list)
    for c in combos:
        groups[sum(c)].append(c)
    return groups

all_combos = generate_combinations(NUM_MODALITIES)  # 长度 2^4 - 1 = 15
groups = group_combinations_by_count(all_combos)    # groups[1], groups[2], groups[3], groups[4]

# ---------- 采样函数：先按组采样，再组内均匀采样 ----------
def sample_combo_per_batch(group_probs=GROUP_PROBS, groups=groups, seed=None):
    """
    返回一个长度 NUM_MODALITIES 的 0/1 tuple，表示本 batch 使用的模态组合。
    """
    if seed is not None:
        random.seed(seed)
    # 构造组级别的概率向量（确保顺序为 1,2,3,4）
    counts = sorted(group_probs.keys())  # [1,2,3,4]
    probs = [group_probs[c] for c in counts]
    # 选择组（按给定概率）
    chosen_count = random.choices(counts, weights=probs, k=1)[0]
    # 在该组内均匀选择一个组合
    chosen_combo = random.choice(groups[chosen_count])
    return tuple(chosen_combo)



def sample_modal_mask(bs, prob_lst = [0.6, 0.2, 0.1, 0.1]):
    # 概率分布
    probs = torch.tensor(prob_lst)
    modal_nums = torch.multinomial(probs, bs, replacement=True) + 1  # (bs,)

    masks = [None] * bs

    # 定义组合
    single = [[0],[1],[2],[3]]
    double = list(itertools.combinations(range(4), 2))  # 6 种
    triple = list(itertools.combinations(range(4), 3))  # 4 种

    def assign_even(indices, combos):
        n = len(indices)
        if n == 0: return []
        combos_tensor = []
        reps = (n // len(combos)) + 1
        for _ in range(reps):
            for c in combos:
                m = torch.zeros(4)
                m[list(c)] = 1
                combos_tensor.append(m)
        combos_tensor = torch.stack(combos_tensor)[:n]
        combos_tensor = combos_tensor[torch.randperm(n)]
        return list(zip(indices, combos_tensor))

    # 单模态
    single_indices = (modal_nums == 1).nonzero(as_tuple=True)[0]
    single_assign = assign_even(single_indices, single)
    for idx, m in single_assign:
        masks[idx] = m

    # 双模态
    double_indices = (modal_nums == 2).nonzero(as_tuple=True)[0]
    double_assign = assign_even(double_indices, double)
    for idx, m in double_assign:
        masks[idx] = m

    # 三模态
    triple_indices = (modal_nums == 3).nonzero(as_tuple=True)[0]
    triple_assign = assign_even(triple_indices, triple)
    for idx, m in triple_assign:
        masks[idx] = m

    # 四模态
    four_indices = (modal_nums == 4).nonzero(as_tuple=True)[0]
    for idx in four_indices:
        masks[idx] = torch.ones(4)

    return torch.stack(masks)  # (bs, 4)


def get_combination_chosen(modality_num):
    combs = []
    for num_none in range(1, modality_num+1):############# 15 种组合状态
        for none_positions in itertools.combinations(range(modality_num), num_none):
            # 创建一个长度为 n 的列表，初始值为 1
            temp = [1] * modality_num
            for i in range(modality_num):
                if i not in none_positions:
                    temp[i] = 0
            combs.append(temp)
    # 为了保证每个epoch都能被选择到
    epoch_combos = copy.deepcopy(combs)#.copy()
    # print(epoch_combos)
    random.shuffle(epoch_combos)   # 重新打乱
    # logger.info(f'current epoch combos:{epoch_combos}')
    return epoch_combos


def do_train(start_epoch, args, model, train_loader, evaluator, optimizer,
             scheduler, checkpointer):

    log_period = args.log_period
    eval_period = args.eval_period
    device = "cuda:"+args.gpu_id
    num_epoch = args.num_epoch

    arguments4 = {}
    arguments4["num_epoch"] = num_epoch
    arguments4["iteration"] = 0

    logger = logging.getLogger("ORBench.train")
    logger.info('start training')

    meters = {
        "loss": AverageMeter(),
        # sdm loss
        "sdm_loss": AverageMeter(),
        "sdm_loss_fuse": AverageMeter(),
        "sdm_loss_local": AverageMeter(),
        # itc loss
        "itc_loss": AverageMeter(),
        "itc_loss_fuse": AverageMeter(),
        "itc_loss_local": AverageMeter(),
        # id loss
        "rgb_loss": AverageMeter(),
        "ni_loss": AverageMeter(),
        "si_loss": AverageMeter(),
        "ci_loss": AverageMeter(),
        "txt_loss": AverageMeter(),
        "fuse_loss": AverageMeter(),

        "kl_loss": AverageMeter(),
        "cos_loss": AverageMeter(),
        "sup_cons_loss": AverageMeter(),

        "rgb_loss_local": AverageMeter(),
        "txt_loss_local": AverageMeter(),
        "si_loss_local": AverageMeter(),
        "ci_loss_local": AverageMeter(),
        "ni_loss_local": AverageMeter(),

        "mlm_loss": AverageMeter(),
        "triplet_loss": AverageMeter(),

        # adaLN 对齐正则
        "adaln_match_loss": AverageMeter(),

        "rgb_acc": AverageMeter(),
        "rgb_acc_local": AverageMeter(),
        "si_acc": AverageMeter(),
        "ni_acc": AverageMeter(),
        "ci_acc": AverageMeter(),
        "txt_acc": AverageMeter(),
        "fuse_acc": AverageMeter(),
        "txt_acc_local": AverageMeter(),
        "si_acc_local": AverageMeter(),
        "ci_acc_local": AverageMeter(),
        "ni_acc_local": AverageMeter(),
        "mlm_acc": AverageMeter(),

        'mse_t_loss': AverageMeter(),
        'mse_si_loss': AverageMeter(),
        'mse_ci_loss': AverageMeter(),
        'mse_ni_loss': AverageMeter(),
    }

    tb_writer = SummaryWriter(log_dir=args.output_dir)


    best_ftop1 = 0.0


    chosen_statisc = [0] * args.modality_num
    combo_ptr = 0
    ########### 每个epoch 选 #############
    epoch_combos = get_combination_chosen(args.modality_num)
    #####################################


    for epoch in range(start_epoch, num_epoch + 1):
        start_time = time.time()
        for meter in meters.values():
            meter.reset()
        model.train()

        ########### 每个epoch选 ###################
        # combs_select = epoch_combos[(epoch - 1) % len(epoch_combos)]

        ### 2026.1.8: 尝试改为轮询一轮之后, 重新打乱combos
        if epoch > len(epoch_combos):
        # if (epoch - 1) % len(epoch_combos) == 0:
            random.shuffle(epoch_combos)   # 
        combs_select = epoch_combos[(epoch - 1) % len(epoch_combos)]

        if args.modality_num == 1:
            combs_select = [1]

        combs_select = [1,1,1,1]
        # 全部是1, 没有gate selection!!! 用来作消融

        chosen_statisc = [chosen_statisc[x] + combs_select[x] for x in range(args.modality_num)]

        if args.dataset_name == 'ORBench':
            logger.info(f'Chosen statisc: sk, cp, nir, text ===> {chosen_statisc}')
        elif args.dataset_name in ['CUHK-PEDES', 'ICFG-PEDES', 'RSTPReid', 'PKU-Sketch']:
            if args.modality_num == 1:
                # 只有text
                combs_select = [0, 0, 0] + [combs_select[0]]
                logger.info(f'Chosen statisc: text ===> {chosen_statisc}')
            else:
                combs_select = [combs_select[0]] + [0, 0] + [combs_select[1]]  ## 0,0是占位符
                logger.info(f'Chosen statisc: sk, text ===> {chosen_statisc}')
        elif args.dataset_name in ['RGBNT201']:
            # cp的分支作为热成像分支
            combs_select = [0] + [combs_select[0]] + [combs_select[1]] + [0]  ## 0,0是占位符, cp位置是tir, 
            logger.info(f'Chosen statisc: tir, nir ===> {chosen_statisc}')

        elif args.dataset_name in ['RGBNT201_Text']:
            # cp的分支作为热成像分支
            combs_select = [0] + combs_select  ## 0,0是占位符, cp位置是tir, 
            logger.info(f'Chosen statisc: tir, rgb, text ===> {chosen_statisc}')  ### nir 改为 rgb, 替换anchor

        ##########################################

        ############
        # 可视化所有batch样本
        # print('vis L2 norm before and after ......')
        # with torch.no_grad():
        #     for n_iter, batch in enumerate(train_loader):
        #         batch = {k: v.to(device) for k, v in batch.items()}
        #         combs_select = torch.tensor([1,1,1,1], dtype=torch.bool).to(device)
        #         batch['combs_select'] = combs_select
        #         ret = model(batch, epoch=epoch)
                # break
        # # 拿到一个epoch的所有样本, 可视化logits和L2特征前后的KDE图; 注意: 不进行反向传播 !!!!
        # from vis.visualization import plot_modalities_feature_norms_overlay
        # r = plot_modalities_feature_norms_overlay(model._feat_dict, epoch=epoch, save_path='/home/dj_2025/competitions/original_code备份_last_H20/vis')
        # print('finish vis L2 norm !')
        # from vis.visualization import plot_modalities_target_non_target_overlay
        # _dict_copy = copy.deepcopy(model.original_logits_dict)
        # for key in ['rgb_logits','sk_logits','nir_logits','cp_logits','txt_logits']:
        #     for each_logit in _dict_copy[key][1]:
        #         each_logit -= model.classifier_proj.bias.detach().cpu().numpy()
        # plot_modalities_target_non_target_overlay(model.original_logits_dict, _dict_copy, epoch=1, save_path='/home/dj_2025/competitions/original_code备份_last_H20/vis', save_name='logits_bias_and_no_bias_all')
        #############
        all_grad_norm = []
        all_feat_norm = []
        for n_iter, batch in enumerate(train_loader): # dj: 用的是包含所有模态的 train loader,  
            ########### 每个batch 选 #############
            # 轮转选择当前step要启用的模态组合
            # if combo_ptr >= len(epoch_combos):
            #     epoch_combos = get_combination_chosen(args.modality_num)
            #     combo_ptr = 0
            # combs_select = epoch_combos[combo_ptr]
            # combo_ptr += 1
            # chosen_statisc = [chosen_statisc[x] + combs_select[x] for x in range(4)]
            # if (n_iter + 1) % log_period == 0:
            #     logger.info(f'Chosen statisc (accumulated): sk, cp, nir, text ===> {chosen_statisc}')
            #####################################

            if args.dataset_name in ['RGBNT201']:
                batch = {
                    'rgbs': batch[0]['RGB'].to(device),
                    'nirs': batch[0]['NI'].to(device),
                    'cps': batch[0]['TI'].to(device),  ### cp作为tir的占位符
                    'pids': batch[1].to(device)
                }
            elif args.dataset_name in ['RGBNT201_Text']:
                batch = {
                    'rgbs': batch[0]['RGB'].to(device),
                    'nirs': batch[0]['NI'].to(device),
                    'cps': batch[0]['TI'].to(device),  ### cp作为tir的占位符
                    'pids': batch[1].to(device),
                    'rgb_text': batch[-1]['rgb_text'].to(device),
                    'nirs_text': batch[-1]['ni_text'].to(device),
                    'cps_text': batch[-1]['ti_text'].to(device),

                }
            else:
                batch = {k: v.to(device) for k, v in batch.items()}

            ################ 每个样本dropout #########################
            # _mask = sample_modal_mask(batch['rgbs'].shape[0]).to(device)  # bs, 4
            # batch['sks'] = batch['sks'] * _mask[:, 0].view(batch['rgbs'].shape[0], 1, 1, 1)
            # batch['cps'] = batch['cps'] * _mask[:, 1].view(batch['rgbs'].shape[0], 1, 1, 1)
            # batch['nirs'] = batch['nirs'] * _mask[:, 2].view(batch['rgbs'].shape[0], 1, 1, 1)
            # if args.data_use_inverse:
            #     batch['rgb_caption_ids'] = batch['rgb_caption_ids'] * _mask[:, 3].view(batch['rgbs'].shape[0], 1)
            #     batch['masked_rgb_caption_ids'] = batch['masked_rgb_caption_ids'] * _mask[:, 3].view(batch['rgbs'].shape[0], 1)
            # else:
            #     batch['caption_ids'] = batch['caption_ids'] * _mask[:, 3].view(batch['rgbs'].shape[0], 1)
            ########################################################

            ############ 每个batch or epoch 随机选 ##################
            combs_select = torch.tensor(combs_select, dtype=torch.bool).to(device)
            batch['combs_select'] = combs_select
            #########################################################

            ret = model(batch, epoch=epoch)

            ######## 分别把特征经过全连接层, 反向传播计算梯度
            # from model import objectives
            # # rgb
            # # rgb_logits_proj = model.classifier_proj(ret[0])
            # # loss = objectives.compute_id_new( rgb_logits_proj, batch['pids'])[0]

            # # 记录一个batch每个样本的梯度norm值,以及每个特征norm值, 然后可视化散点图

            # loss = model.cls_head(ret[0], batch['pids'])[0]
            # # loss.backward()
            # for i in range(loss.shape[0]): 
            #     grads_i = torch.autograd.grad(loss[i], model.cls_head.weight, retain_graph=True, create_graph=False) 
            #     all_grad_norm.append(grads_i[0].detach().cpu().norm(2).item())
            #     all_feat_norm.append(ret[0][i].detach().cpu().norm(2).item())
            # # rgb_g_norm = model.cls_head.weight.grad.norm(2).item()

            # model.zero_grad()
            # # si_feats_proj = model.classifier_proj(ret[1])
            # # loss = objectives.compute_id_new( si_feats_proj, batch['pids'])[0]
            # loss = model.cls_head(ret[1], batch['pids'])[0]
            # # loss.backward()
            # # si_g_norm = model.cls_head.weight.grad.norm(2).item()
            # for i in range(loss.shape[0]): 
            #     grads_i = torch.autograd.grad(loss[i], model.cls_head.weight, retain_graph=True, create_graph=False) 
            #     all_grad_norm.append(grads_i[0].detach().cpu().norm(2).item())
            #     all_feat_norm.append(ret[1][i].detach().cpu().norm(2).item())

            # model.zero_grad()
            # # ci_feats_proj = model.classifier_proj(ret[2])
            # # loss = objectives.compute_id_new( ci_feats_proj, batch['pids'])[0]
            # loss = model.cls_head(ret[2], batch['pids'])[0]
            # # loss.backward()
            # # ci_g_norm = model.cls_head.weight.grad.norm(2).item()
            # for i in range(loss.shape[0]): 
            #     grads_i = torch.autograd.grad(loss[i], model.cls_head.weight, retain_graph=True, create_graph=False) 
            #     all_grad_norm.append(grads_i[0].detach().cpu().norm(2).item())
            #     all_feat_norm.append(ret[2][i].detach().cpu().norm(2).item())

            # model.zero_grad()
            # # ni_feats_proj = model.classifier_proj(ret[3])
            # # loss = objectives.compute_id_new( ni_feats_proj, batch['pids'])[0]
            # loss = model.cls_head(ret[3], batch['pids'])[0]
            # # loss.backward()
            # # ni_g_norm = model.cls_head.weight.grad.norm(2).item()
            # for i in range(loss.shape[0]): 
            #     grads_i = torch.autograd.grad(loss[i], model.cls_head.weight, retain_graph=True, create_graph=False) 
            #     all_grad_norm.append(grads_i[0].detach().cpu().norm(2).item())
            #     all_feat_norm.append(ret[3][i].detach().cpu().norm(2).item())

            # model.zero_grad()
            # # text_feat = model.classifier_proj(ret[4])
            # # loss = objectives.compute_id_new( text_feat, batch['pids'])[0]
            # loss = model.cls_head(ret[4], batch['pids'])[0]
            # # loss.backward()
            # # txt_g_norm = model.cls_head.weight.grad.norm(2).item()
            # for i in range(loss.shape[0]): 
            #     grads_i = torch.autograd.grad(loss[i], model.cls_head.weight, retain_graph=True, create_graph=False) 
            #     all_grad_norm.append(grads_i[0].detach().cpu().norm(2).item())
            #     all_feat_norm.append(ret[4][i].detach().cpu().norm(2).item())

            # model.zero_grad()
            ################# 

            total_loss = sum([v for k, v in ret.items() if "loss" in k])

            bs = batch['rgbs'].shape[0]
            meters['loss'].update(total_loss.item(), bs)
            # acc_meter.update(ret.get('acc', 0), 1)
            # print(ret.get('img_acc'))
            # img_acc_meter.update(ret.get('img_acc', 0), 1)
            # text_acc_meter.update(ret.get('text_acc', 0), 1)
            meters['rgb_acc'].update(ret.get('rgb_acc', 0), 1)
            meters['txt_acc'].update(ret.get('txt_acc', 0), 1)
            meters['si_acc'].update(ret.get('si_acc', 0), 1)
            meters['ni_acc'].update(ret.get('ni_acc', 0), 1)
            meters['ci_acc'].update(ret.get('ci_acc', 0), 1)
            meters['fuse_acc'].update(ret.get('fuse_acc', 0), 1)
            meters['mlm_acc'].update(ret.get('mlm_acc', 0), 1)
            
            meters['mlm_loss'].update(ret.get('mlm_loss', 0), bs)
            meters['triplet_loss'].update(ret.get('triplet_loss', 0), bs)
            # id_loss_meter.update(ret.get('id_loss', 0), bs)
            meters['itc_loss'].update(ret.get('itc_loss', 0), bs)
            meters['sdm_loss'].update(ret.get('sdm_loss', 0), bs)
            meters['sdm_loss_local'].update(ret.get('sdm_loss_local', 0), bs)

            meters['kl_loss'].update(ret.get('kl_loss', 0), bs)
            meters['cos_loss'].update(ret.get('cos_loss', 0), bs)
            meters['sup_cons_loss'].update(ret.get('sup_cons_loss', 0), bs)

            meters['rgb_acc_local'].update(ret.get('rgb_acc_local', 0), 1)
            meters['txt_acc_local'].update(ret.get('txt_acc_local', 0), 1)
            meters['si_acc_local'].update(ret.get('si_acc_local', 0), 1)
            meters['ci_acc_local'].update(ret.get('ci_acc_local', 0), 1)
            meters['ni_acc_local'].update(ret.get('ni_acc_local', 0), 1)

            meters['rgb_loss_local'].update(ret.get('rgb_loss_local', 0), bs)
            meters['txt_loss_local'].update(ret.get('txt_loss_local', 0), bs)
            meters['si_loss_local'].update(ret.get('si_loss_local', 0), bs)
            meters['ci_loss_local'].update(ret.get('ci_loss_local', 0), bs)
            meters['ni_loss_local'].update(ret.get('ni_loss_local', 0), bs)

            meters['sdm_loss_fuse'].update(ret.get('sdm_loss_fuse', 0), bs)
            meters['itc_loss_fuse'].update(ret.get('itc_fuse_loss', 0), bs)

            meters['rgb_loss'].update(ret.get('rgb_loss', 0), bs)
            meters['si_loss'].update(ret.get('si_loss', 0), bs)
            meters['ni_loss'].update(ret.get('ni_loss', 0), bs)
            meters['ci_loss'].update(ret.get('ci_loss', 0), bs)
            meters['txt_loss'].update(ret.get('txt_loss', 0), bs)
            meters['fuse_loss'].update(ret.get('fuse_loss', 0), bs)

            # adaLN 对齐正则
            meters['adaln_match_loss'].update(ret.get('adaln_match_loss', 0), bs)

            meters['mse_t_loss'].update(ret.get('mse_loss_t', 0),bs)
            meters['mse_ni_loss'].update(ret.get('mse_loss_ni', 0), bs)
            meters['mse_ci_loss'].update(ret.get('mse_loss_ci', 0), bs)
            meters['mse_si_loss'].update(ret.get('mse_loss_si', 0), bs)



            # # 拿到一个batch的样本, 可视化logits和L2特征前后的KDE图
            # from vis.visualization import plot_modalities_feature_norms_overlay
            # r = plot_modalities_feature_norms_overlay(model._feat_dict, epoch=1, save_path='/home/dj_2025/competitions/original_code备份_last_H20/vis')
            # from vis.visualization import plot_modalities_target_non_target_overlay
            # _dict_copy = copy.deepcopy(model.original_logits_dict)
            # for key in ['rgb_logits','sk_logits','nir_logits','cp_logits','txt_logits']:
            #     for each_logit in _dict_copy[key][1]:
            #         each_logit -= model.classifier_proj.bias.detach().cpu().numpy()
            # plot_modalities_target_non_target_overlay(model.original_logits_dict, _dict_copy, epoch=1, save_path='/home/dj_2025/competitions/original_code备份_last_H20/vis', save_name='logits_bias_and_no_bias')

            optimizer.zero_grad()
            # with torch.autograd.set_detect_anomaly(True):
            total_loss.backward()
            optimizer.step()
            synchronize()

            if (n_iter + 1) % log_period == 0:
                # logger.info(
                #     f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}] Loss: {loss_meter.avg:.4f}, "
                #     f"rgb_id_loss: {rgb_loss_meter.avg:.4f}, text_id_loss: {txt_loss_meter.avg:.4f},si_id_loss: {si_loss_meter.avg:.4f},ci_id_loss: {ci_loss_meter.avg:.4f},ni_id_loss: {ni_loss_meter.avg:.4f},fu_id_loss: {fu_loss_meter.avg:.4f},"
                #     f"mlm_loss: {mlm_loss_meter.avg:.4f}, itc_loss: {itc_loss_meter.avg:.4f}, sdm_loss: {sdm_loss_meter.avg:.4f},"
                #     f"mlm_acc: {acc_meter.avg:.3f}, rgb_acc: {rgb_acc_meter.avg:.3f}, text_acc: {txt_acc_meter.avg:.3f},si_acc: {si_acc_meter.avg:.3f},ci_acc: {ci_acc_meter.avg:.3f},ni_acc: {ni_acc_meter.avg:.3f},fu_acc: {fu_acc_meter.avg:.3f}, "
                #     f"Base Lr: {scheduler.get_lr()[0]:.2e}" #, temp_txt: {ret['temp_txt']}, temp_sk: {ret['temp_sk']}, temp_nir: {ret['temp_nir']}, temp_fuse: {ret['temp_fuse']}"
                # )

                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]"
                # log loss and acc info
                for k, v in meters.items():
                    # 只打印大于0的loss
                    if v.avg > 0:
                        info_str += f", {k}: {v.avg:.4f}"
                info_str += f", Base Lr: {scheduler.get_lr()[0]:.2e}"
                logger.info(info_str)

        # logits的图在这里产生
        # from vis.visualization import plot_modalities_feature_norms_overlay
        # r = plot_modalities_feature_norms_overlay(model._feat_dict, epoch=1, save_path='/home/dj_2025/competitions/original_code备份_last_H20/vis')
        ## 可视化logits
        # from vis.visualization import plot_logit_distributions_from_nested_dicts, plot_modalities_target_non_target_overlay
        # 第1和第15个epoch, 此时5个模态都有, 可以全部可视化Logits
        # if epoch in [1, 16]:
            # plot_logit_distributions_from_nested_dicts(model.original_logits_dict, model.after_logits_dict, epoch=epoch, save_path=args.output_dir, logger=logger)
            # plot_modalities_target_non_target_overlay(model.original_logits_dict, model.after_logits_dict, epoch=epoch, save_path=args.output_dir)
        # 拿到一个epoch的所有样本, 可视化logits和L2特征前后的KDE图; 注意: 不进行反向传播 !!!!

        # from vis.visualization import plot_logit_distributions_from_nested_dicts
        # # r = plot_modalities_feature_norms_overlay(model._feat_dict, epoch=epoch, save_path='/home/dj_2025/competitions/original_code备份_last_H20/vis')
        # # logits图只可视化rgb logits
        # r = plot_logit_distributions_from_nested_dicts(model.original_logits_dict, model.after_logits_dict, epoch=1, modalities=('rgb_logits', ),
        #                                                 save_path='/home/dj_2025/competitions/original_code备份_last_H20/vis')
        # print('finish logits fig !')

        ### 每一个epoch更新完base model, 然后再更新teacher base model
        if args.use_EMA_model:
            # 线性增长
            # alpha = ((epoch - 1) / args.num_epoch) * 0.999
            # 指数增长
            alpha_end = 0.999
            tau = max(1.0, args.num_epoch /12.0)  # 5  ==> tau越大, 前期teacher占比就越小, tau越小, teacher很快就占主导地位而不是student
            alpha = alpha_end * (1.0 - math.exp(- (epoch - 1) / tau))
            logger.info(f'EMA alpha: {alpha}')

            # alpha = 1 - math.exp(-global_step / (len(train_loader) * args.num_epoch)) * 0.999 * (epoch-1)  ## 0, 1 - 0.999*exp(-s/ALL_STEP), s越大, exp越小,
            for ema_param, param in zip(model.teacher_base_model.parameters(), model.base_model.parameters()):
                ema_param.data.mul_(alpha).add_(1 - alpha, param.data)  # ema_param*alpha + (1-alpha)*param
                ema_param.requires_grad = False


        tb_writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
        tb_writer.add_scalar('temperature', ret['temperature'], epoch)
        for k, v in meters.items():
            if v.avg > 0:
                tb_writer.add_scalar(k, v.avg, epoch)
        # tb_writer.add_scalar('loss', loss_meter.avg, epoch)
        # # tb_writer.add_scalar('mcm_loss', mcm_loss_meter.avg, epoch)
        # tb_writer.add_scalar('id_loss', id_loss_meter.avg, epoch)
        # tb_writer.add_scalar('mlm_loss', mlm_loss_meter.avg, epoch)
        # # tb_writer.add_scalar('mcq_loss', mcq_loss_meter.avg, epoch)
        # tb_writer.add_scalar('acc', acc_meter.avg, epoch)
        # tb_writer.add_scalar('img_acc', img_acc_meter.avg, epoch)
        # tb_writer.add_scalar('text_acc', text_acc_meter.avg, epoch)

        scheduler.step()
        if get_rank() == 0:
            end_time = time.time()
            epoch_time = end_time - start_time
            # log_training_time_per_epoch(logger, epoch, epoch_time)
            time_per_batch = epoch_time / (n_iter + 1)
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                .format(epoch, time_per_batch,
                        train_loader.batch_size / time_per_batch))
        if epoch % eval_period == 0:
            if get_rank() == 0:
                logger.info("Validation Results - Epoch: {}".format(epoch))
                # if args.distributed:
                #     ttop1, stop1, itop1 = evaluator.eval(model.module.eval())

                # onemodal_top1, twomodal_top1, threemodal_top1, fourmodal_top1 = evaluator.eval(model.eval()) ## 这个是 map
                # ftop1 = (onemodal_top1 + twomodal_top1 + threemodal_top1 + fourmodal_top1) / 4.0
                ftop1 = evaluator.eval(model.eval())
                # top1_average = evaluator.eval(model.eval())
                # args.training = True
                # logger.info(f"rank_1: {np.mean(rank1)}, rank_5: {np.mean(rank5)}, rank_10: {np.mean(rank10)}, map: {np.mean(mAP)}")
                # ftop1 = np.mean(mAP)
                #######
                torch.cuda.empty_cache()

                if best_ftop1 < ftop1:
                    best_ftop1 = ftop1
                    arguments4["epoch"] = epoch
                    checkpointer.save("average_best", **arguments4)
                logger.info(f"average best mAP: {best_ftop1} at epoch {arguments4['epoch']}")


def do_inference(
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
    ):

    logger = logging.getLogger("ORBench.test")
    logger.info("Enter inferencing")

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
    ftop1 = evaluator.eval(model.eval())
    # top1 = evaluator.eval_by_proj(model.eval())

    # table = PrettyTable(["task", "R1", "R5", "R10", "mAP"])
    # table.float_format = '.4'
    # table.add_row(['t2i', cmc[0], cmc[4], cmc[9], mAP])
    # logger.info("Validation Results: ")
    # logger.info('\n' + str(table))


def get_query_feat(q_loader, img_type, device, model):
    qids, qfeats = [], []
    for pid, img in q_loader:
        img = img.to(device)
        with torch.no_grad():
            img_feat, _ = model.encode_image(img, img_type)
        qids.append(pid.view(-1).cpu())
        qfeats.append(img_feat.cpu())
    return qids, qfeats


import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import os
from matplotlib.lines import Line2D

def do_tsne_visualization(device, model, train_loader, 
                          save_path='/home/dj_2025/competitions/original_code备份_last_H20/vis/tsne'):
    """
    修改后的 t-SNE 可视化:
      - 不同模态用不同颜色 (指定 Hex)
      - 不同 ID 用不同形状
    """
    # os.makedirs(save_path, exist_ok=True)

    # ===========================
    # 1. 数据提取 (保持原有逻辑)
    # ===========================
    g_feats, nir_feats, cp_feats, sk_feats, text_feats = [], [], [], [], []
    ids_list = []
    
    print('Getting queries feats....')
    
    for _, batch in enumerate(train_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        combs_select = [1,1,1,1]
        combs_select = torch.tensor(combs_select, dtype=torch.bool).to(device)
        with torch.no_grad():
            _, _, image_features_rgb_proj_t, \
            _, _, image_features_sk_proj_t, \
            _, _, image_features_cp_proj_t,\
            _, _, image_features_nir_proj_t, text_features_t, text_features_sk_t, text_features_cp_t, text_features_nir_t = model.teacher_base_model(  ##### 如果是baseline, 没有EMA
                [batch['rgbs'], batch['sks'], batch['cps'], batch['nirs']], batch['caption_ids'], combs_select=combs_select
            )
            # 提取特征
            text_feat = text_features_t[torch.arange(text_features_t.shape[0]), batch['caption_ids'].argmax(dim=-1)].float()
            img_feat = image_features_cp_proj_t[:, 0, :].float()
            img2_feat = image_features_sk_proj_t[:, 0, :].float()
            img3_feat = image_features_nir_proj_t[:, 0, :].float()
            g_img_feat = image_features_rgb_proj_t[:, 0, :].float()

        g_feats.append(g_img_feat.cpu())
        ids_list.append(batch['pids'].cpu())
        cp_feats.append(img_feat.cpu())
        sk_feats.append(img2_feat.cpu())
        nir_feats.append(img3_feat.cpu())
        text_feats.append(text_feat.cpu())
        
    # 拼接数据
    ids_all = torch.cat(ids_list, 0).numpy()
    cp_feats = torch.cat(cp_feats, 0).numpy()
    sk_feats = torch.cat(sk_feats, 0).numpy()
    nir_feats = torch.cat(nir_feats, 0).numpy()
    text_feats = torch.cat(text_feats, 0).numpy()
    g_feats = torch.cat(g_feats, 0).numpy()

    # ===========================
    # 2. t-SNE 降维
    # ===========================
    # 筛选 ID (保持原逻辑: 前6个 + 后4个)
    all_ids_lst = list(set(list(ids_all)))
    # 确保 ID 数量足够，防止越界
    if len(all_ids_lst) > 10:
        select_ids = all_ids_lst[:6] + all_ids_lst[-5:-1]
    else:
        select_ids = all_ids_lst
        
    mask = np.isin(ids_all, np.array(select_ids))
    
    # 应用掩码
    g_feats = g_feats[mask]
    nir_feats = nir_feats[mask]
    cp_feats = cp_feats[mask]
    sk_feats = sk_feats[mask]
    text_feats = text_feats[mask]
    ids_filtered = ids_all[mask]

    # 构造 t-SNE 输入（按模态垂直堆叠）
    # 堆叠顺序：RGB -> NIR -> CP -> Sketch -> Text
    X = np.vstack([g_feats, nir_feats, cp_feats, sk_feats, text_feats])
    n_per_mod = g_feats.shape[0] # 每个模态的样本数

    print("Starting TSNE...")
    tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=42, n_jobs=-1)
    X_2d = tsne.fit_transform(X)
    print("TSNE finished.")

    # ===========================
    # 3. 绘图配置 (核心修改部分)
    # ===========================
    
    # 3.1 定义模态颜色 (Key必须与下面的 modality_names 对应)
    modality_names = ['RGB', 'NIR', 'ColorPencil', 'Sketch', 'Text']
    
    modality_colors = {
        'RGB': '#006EAF',         # 蓝色系
        'NIR': '#82B366',         # 绿色系
        'ColorPencil': '#B85450', # 红色系
        'Sketch': '#D6B656',      # 黄色系
        'Text': '#9673A6'         # 紫色系
    }

    # 3.2 定义 ID 形状 (Marker)
    # 准备一组形状池，分配给不同的 ID
    available_markers = ['o', 's', '^', 'D', 'P', 'X', '*', 'v', '<', 'p']
    unique_ids_viz = np.unique(ids_filtered)
    
    id_markers = {}
    for idx, pid in enumerate(unique_ids_viz):
        # 如果 ID 数量超过形状数量，循环使用
        marker = available_markers[idx % len(available_markers)]
        id_markers[int(pid)] = marker

    # 构造辅助变量：ids_repeated 用于索引每个点的 ID
    ids_repeated = np.concatenate([ids_filtered for _ in modality_names])

    # ===========================
    # 4. 开始绘制
    # ===========================
    sns.set_theme(style='white')
    plt.figure(figsize=(8, 10)) # 稍微加大一点画布方便放图例

    # 双重循环：外层模态(控制颜色)，内层ID(控制形状)
    for i, mod in enumerate(modality_names):
        start = i * n_per_mod
        end = start + n_per_mod
        
        # 当前模态的所有坐标和ID
        xs_mod = X_2d[start:end, 0]
        ys_mod = X_2d[start:end, 1]
        ids_mod = ids_repeated[start:end]
        
        # 获取当前模态的颜色
        c = modality_colors[mod]
        
        # 按 ID 分组绘制 (为了使用不同的 marker)
        for pid in np.unique(ids_mod):
            mask_id = (ids_mod == pid)
            m = id_markers[int(pid)]
            
            plt.scatter(
                xs_mod[mask_id], 
                ys_mod[mask_id],
                marker=m,              # 形状由 ID 决定
                color=c,               # 颜色由 模态 决定
                s=80,                  # 点的大小
                edgecolor='white',     # 描边增加对比度
                linewidth=0.5,
                alpha=0.85
            )

    # ===========================
    # 5. 构造图例 (Legends)   => 不要图例
    # ===========================
    
    # Legend 1: 模态 (颜色)
    # modality_handles = []
    # for mod in modality_names:
    #     # 使用方形 ('s') 代表颜色块
    #     h = Line2D([0], [0], marker='s', color='w', label=mod,
    #                markerfacecolor=modality_colors[mod], markersize=10)
    #     modality_handles.append(h)
        
    # leg1 = plt.legend(handles=modality_handles, title='Modality', loc='upper left', 
    #                   bbox_to_anchor=(1.01, 1), borderaxespad=0)
    # plt.gca().add_artist(leg1) # 手动添加第一个图例，防止被第二个覆盖

    # Legend 2: ID (形状)
    # id_handles = []
    # for pid in unique_ids_viz:
    #     m = id_markers[int(pid)]
    #     # 使用黑色 ('k') 显示形状，不干扰颜色逻辑
    #     h = Line2D([0], [0], marker=m, color='w', label=f'ID {pid}',
    #                markerfacecolor='k', markeredgecolor='k', markersize=8)
    #     id_handles.append(h)

    # plt.legend(handles=id_handles, title='Identity', loc='upper left', 
    #            bbox_to_anchor=(1.01, 0.6), borderaxespad=0)

    # 去除坐标轴刻度
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    
    out_file = os.path.join(save_path, 'tsne_modal_color_id_shape_CLIP_no_legend_1.png')
    plt.savefig(out_file, dpi=300, bbox_inches='tight') # bbox_inches='tight' 防止图例被切掉
    plt.close()
    print(f"Saved t-SNE figure to {out_file}")


import os
import numpy as np
import torch
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

def do_tsne_visualization_origin(device, model, train_loader, 
                          save_path='/home/dj_2025/competitions/original_code备份_last_H20/vis/tsne'):
    """
    改用训练集的tsne, 把训练集的特征提取出来
    绘图要求：
      - 不同 id 用不同颜色（柔和配色）
      - 5 种模态用 5 个不同形状表示
    """
    # os.makedirs(save_path, exist_ok=True)

    # ===========================
    # 1. 数据提取 (保持原有逻辑)
    # ===========================
    g_feats, nir_feats, cp_feats, sk_feats, text_feats = [], [], [], [], []
    ids_list = []
    
    print('Getting queries feats....')
    
    for _, batch in enumerate(train_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        combs_select = [1,1,1,1]
        combs_select = torch.tensor(combs_select, dtype=torch.bool).to(device)
        with torch.no_grad():
            _, _, image_features_rgb_proj_t, \
            _, _, image_features_sk_proj_t, \
            _, _, image_features_cp_proj_t,\
            _, _, image_features_nir_proj_t, text_features_t, text_features_sk_t, text_features_cp_t, text_features_nir_t = model.base_model(  ##### 如果是baseline, 没有EMA
                [batch['rgbs'], batch['sks'], batch['cps'], batch['nirs']], batch['caption_ids'], combs_select=combs_select
            )
            text_feat = text_features_t[torch.arange(text_features_t.shape[0]), batch['caption_ids'].argmax(dim=-1)].float()
            img_feat = image_features_cp_proj_t[:, 0, :].float()
            img2_feat = image_features_sk_proj_t[:, 0, :].float()
            img3_feat = image_features_nir_proj_t[:, 0, :].float()
            g_img_feat = image_features_rgb_proj_t[:, 0, :].float()

        g_feats.append(g_img_feat.cpu())
        ids_list.append(batch['pids'].cpu())
        cp_feats.append(img_feat.cpu())
        sk_feats.append(img2_feat.cpu())
        nir_feats.append(img3_feat.cpu())
        text_feats.append(text_feat.cpu())
        
    # 拼接 Query 数据
    ids_all = torch.cat(ids_list, 0).numpy()
    cp_feats = torch.cat(cp_feats, 0).numpy()
    sk_feats = torch.cat(sk_feats, 0).numpy()
    nir_feats = torch.cat(nir_feats, 0).numpy()
    text_feats = torch.cat(text_feats, 0).numpy()
        
    # 拼接 Gallery 数据
    g_feats = torch.cat(g_feats, 0).numpy()

    # ===========================
    # 2. t-SNE 降维
    # ===========================
    # 随机选若干个 id（保持原逻辑）
    all_ids_lst = list(set(list(ids_all)))
    select_ids = all_ids_lst[: 6] + all_ids_lst[-5: -1]
    mask = np.isin(ids_all, np.array(select_ids))
    g_feats, nir_feats, cp_feats, sk_feats, text_feats = g_feats[mask], nir_feats[mask], cp_feats[mask], sk_feats[mask], text_feats[mask]
    ids_filtered = ids_all[mask]


    # ### 把相同id的特征提取出来
    # _id_feat_dict = {}
    # for idx, _each_id in enumerate(ids_filtered):
    #     if _each_id in _id_feat_dict:
    #         _id_feat_dict[_each_id]['rgb_feat'].append(g_feats[idx])
    #         _id_feat_dict[_each_id]['nir_feat'].append(nir_feats[idx])
    #         _id_feat_dict[_each_id]['sk_feat'].append(sk_feats[idx])
    #         _id_feat_dict[_each_id]['cp_feat'].append(cp_feats[idx])
    #         _id_feat_dict[_each_id]['txt_feat'].append(text_feats[idx])
    #     else:
    #         _id_feat_dict[_each_id] = {}
    #         _id_feat_dict[_each_id].update( {'rgb_feat': [g_feats[idx]]})
    #         _id_feat_dict[_each_id].update({'nir_feat': [nir_feats[idx]]})
    #         _id_feat_dict[_each_id].update({'sk_feat' : [sk_feats[idx]]})
    #         _id_feat_dict[_each_id].update({'cp_feat': [cp_feats[idx]]})
    #         _id_feat_dict[_each_id].update({'txt_feat': [text_feats[idx]]})

    # 每个模态的样本数（应相同）
    n_per_mod = g_feats.shape[0]

    # 构造 t-SNE 输入（按模态垂直堆叠）
    X = np.vstack([g_feats, nir_feats, cp_feats, sk_feats, text_feats])

    modality_names = ['RGB', 'NIR', 'ColorPencil', 'Sketch', 'Text']
    modality_markers = {
        'RGB': 'o',
        'NIR': 's',
        'ColorPencil': '^',
        'Sketch': 'P',
        'Text': 'X'
    }

    # ids 对应的颜色（柔和 pastel）
    n_ids = len(select_ids)
    # 使用 seaborn pastel 调色板（长度为 n_ids）
    palette = sns.color_palette("pastel", n_colors=n_ids)
    id_to_color = {int(idv): palette[i] for i, idv in enumerate(select_ids)}

    # modality 对应的 id 列表（重复 ids_filtered）
    ids_repeated = np.concatenate([ids_filtered for _ in modality_names])

    print("Starting TSNE...")
    tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=42, n_jobs=-1)
    X_2d = tsne.fit_transform(X)
    print("TSNE finished.")

    # ===========================
    # 3. 绘图（按 id 着色，按模态用 marker）
    # ===========================
    sns.set_theme(style='white')
    plt.figure(figsize=(8, 8))

    # 为避免图例重复，逐模态绘制：在每个模态内按 id 分组绘制
    for i, mod in enumerate(modality_names):
        start = i * n_per_mod
        end = start + n_per_mod
        xs = X_2d[start:end, 0]
        ys = X_2d[start:end, 1]
        ids_this_mod = ids_repeated[start:end]

        # 对每个 id 单独绘制（保证颜色一致）
        for idv in np.unique(ids_this_mod):
            mask_id = ids_this_mod == idv
            plt.scatter(xs[mask_id], ys[mask_id],
                        marker=modality_markers[mod],
                        s=35,
                        color=id_to_color[int(idv)],
                        edgecolor='k',
                        linewidth=0.25,
                        alpha=0.75,
                        label=f'{mod}' if idv == np.unique(ids_this_mod)[0] else None)
            # 仅在该模态的第一个 id 上添加模态 label（避免重复 legend 条目）

    # 构造模态图例（marker）
    modality_handles = []
    for mod in modality_names:
        handle = Line2D([0], [0], marker=modality_markers[mod], color='w',
                        markerfacecolor='lightgray', markeredgecolor='k', markersize=8, linewidth=0)
        modality_handles.append(handle)

    # 构造 id 图例（颜色块）
    # id_handles = []
    # for idv in select_ids:
    #     patch = mpatches.Patch(color=id_to_color[int(idv)], label=str(idv), alpha=0.9)
    #     id_handles.append(patch)

    # 放置图例：左上为模态，右上为 id（可根据需要调整位置）
    legend1 = plt.legend(modality_handles, modality_names, title='Modality', loc='upper left', frameon=True)
    plt.gca().add_artist(legend1)
    # plt.legend(handles=id_handles, title='ID', loc='upper right', ncol=1, frameon=True, fontsize='small')

    # plt.title('t-SNE: IDs (colors) and Modalities (markers)', fontsize=12)
    # plt.xlabel('t-SNE 1')
    # plt.ylabel('t-SNE 2')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    out_file = os.path.join(save_path, '5_modality_tsne_10_ids_pastel_ours_student.png')
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved t-SNE figure to {out_file}")



def do_tsne_visualization_1(device, model, train_loader, 
                          save_path='/home/dj_2025/competitions/original_code备份_last_H20/vis'):
    """
    改用训练集的tsne, 把训练集的特征提取出来 TODO
    """
    # ===========================
    # 1. 数据提取 (保持原有逻辑)
    # ===========================
    g_feats, nir_feats, cp_feats, sk_feats, text_feats = [], [], [], [], []
    ids_list = []
    gids_list = []
    
    print('Getting queries feats....')
    
    for _, batch in enumerate(train_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        combs_select = [1,1,1,1]
        combs_select = torch.tensor(combs_select, dtype=torch.bool).to(device)
        # batch['combs_select'] = combs_select
        with torch.no_grad():
            _, _, image_features_rgb_proj_t, \
            _, _, image_features_sk_proj_t, \
            _, _, image_features_cp_proj_t,\
            _, _, image_features_nir_proj_t, text_features_t, text_features_sk_t, text_features_cp_t, text_features_nir_t = model.teacher_base_model(
                [batch['rgbs'], batch['sks'], batch['cps'], batch['nirs']], batch['caption_ids'], combs_select=combs_select
            )
            text_feat = text_features_t[torch.arange(text_features_t.shape[0]), batch['caption_ids'].argmax(dim=-1)].float()
            img_feat = image_features_cp_proj_t[:, 0, :].float()
            img2_feat = image_features_sk_proj_t[:, 0, :].float()
            img3_feat = image_features_nir_proj_t[:, 0, :].float()
            g_img_feat = image_features_rgb_proj_t[:, 0, :].float()

        g_feats.append(g_img_feat.cpu())
        ids_list.append(batch['pids'].cpu())
        cp_feats.append(img_feat.cpu())
        sk_feats.append(img2_feat.cpu())
        nir_feats.append(img3_feat.cpu())
        text_feats.append(text_feat.cpu())
        
    # 拼接 Query 数据
    ids_all = torch.cat(ids_list, 0).numpy()
    cp_feats = torch.cat(cp_feats, 0).numpy()
    sk_feats = torch.cat(sk_feats, 0).numpy()
    nir_feats = torch.cat(nir_feats, 0).numpy()
    text_feats = torch.cat(text_feats, 0).numpy()
        
    
    # 拼接 Gallery 数据
    # gids_all = torch.cat(gids_list, 0).numpy()
    g_feats = torch.cat(g_feats, 0).numpy()

    # ===========================
    # 2. t-SNE 降维
    # ===========================
    # 随机选若干个
    select_ids = list(set(list(ids_all)))[: 10]
    mask = np.isin(ids_all, np.array(select_ids))
    g_feats, nir_feats, cp_feats, sk_feats, text_feats = g_feats[mask], nir_feats[mask], cp_feats[mask], sk_feats[mask], text_feats[mask]

    X = np.vstack([g_feats, nir_feats, cp_feats, sk_feats, text_feats])
    # X = np.vstack([g_feats, nir_feats])
    
    # 定义模态顺序
    # 关键：我们将在绘图循环中控制层级，所以这里的顺序主要影响图例排序
    # modality_order = ['RGB', 'NIR']#, 'ColorPencil', 'Sketch', 'Text']
    
    modality = (['RGB']*len(g_feats) + ['NIR']*len(nir_feats)  +
                       ['ColorPencil']*len(cp_feats) + ['Sketch']*len(sk_feats) +
                       ['Text']*len(text_feats))

    print("Starting TSNE...")
    # perplexity=30 是标准值，n_jobs=-1 加速计算
    tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=42, n_jobs=-1)
    X_2d = tsne.fit_transform(X)
    print("TSNE finished.")

    sns.set_theme()
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=X_2d[:,0], y=X_2d[:,1], hue=modality,
                    s=35, palette='tab10')
    plt.title('5 one-Modality t-SNE')
    plt.legend(title='Modality')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, '5_modality_tsne_10_ids.png'), dpi=300)
    # plt.show()

    # ===========================
    # 3. 样式配置 (严格复刻原图风格)
    # ===========================
    
    # 3.1 颜色提取 (Hex Codes extracted from the sample image)
    # color_map = {
    #     'RGB':         '#C0504D',  # 对应图中的 GRID (红褐色)
    #     'NIR':         '#E4B07B',  # 对应图中的 PRID (黄褐色)
    #     'ColorPencil': '#55C1B8',  # 对应图中的 VIPeR (青绿色)
    #     'Sketch':      '#BA7CEE',  # 对应图中的 iLIDS (紫色)
    #     'Text':        '#6495ED'   # 新增：矢车菊蓝 (为了协调性)
    # }

    # # 3.2 Marker 形状映射
    # # 图中 Logic: Query 是六边形('h'), Gallery 是三叉形('1')
    # # 你的 Logic: RGB 是 Gallery, 其他是 Query
    # marker_map = {
    #     'RGB': '1',          # Tri_down (三叉形)
    #     'NIR': 'h',          # Hexagon (六边形)
    #     'ColorPencil': 'h',
    #     'Sketch': 'h',
    #     'Text': 'h'
    # }

    # # 3.3 视觉层级与透明度
    # # RGB 需要在最上面 (zorder大)，且稍微清晰一点
    # zorder_map = {'RGB': 10, 'NIR': 2, 'ColorPencil': 3, 'Sketch': 4, 'Text': 5}
    # alpha_map = {'RGB': 0.9, 'NIR': 0.5, 'ColorPencil': 0.5, 'Sketch': 0.5, 'Text': 0.5}
    # # 调整大小：六边形视觉上比三叉星大，所以三叉星(RGB)要设大一点数值来平衡
    # size_map = {'RGB': 45, 'NIR': 40, 'ColorPencil': 40, 'Sketch': 40, 'Text': 40}

    # # ===========================
    # # 4. 绘图执行
    # # ===========================
    # # 设置纯白风格
    # sns.set_style("white")
    # fig, ax = plt.subplots(figsize=(9, 9)) # 方形画布

    # for mod in modality_order:
    #     idx = [i for i, label in enumerate(modality_labels) if label == mod]
    #     points = X_2d[idx]
        
    #     # 核心绘图
    #     ax.scatter(points[:, 0], points[:, 1], 
    #                c=color_map[mod], 
    #                marker=marker_map[mod], 
    #                s=size_map[mod], 
    #                alpha=alpha_map[mod],
    #                zorder=zorder_map[mod], # 关键：RGB zorder=10 会盖在其他 zorder 上
    #                # RGB 稍微加粗一点线条增加辨识度，Query 去掉边框模拟原图的半透明重叠感
    #                linewidths=1.5 if mod == 'RGB' else 0,
    #                edgecolors=color_map[mod] if mod == 'RGB' else 'none', 
    #                label=mod)

    # # ===========================
    # # 5. 细节修饰 (复刻原图 UI)
    # # ===========================
    
    # # 坐标轴标签加粗
    # ax.set_xlabel('x', fontsize=12, fontweight='bold')
    # ax.set_ylabel('y', fontsize=12, fontweight='bold')

    # # 图例设置 (左上角，半透明背景，加粗字体)
    # legend = ax.legend(loc='upper left', prop={'weight': 'bold', 'size': 10}, frameon=True)
    # legend.get_frame().set_alpha(0.8) 
    # legend.get_frame().set_edgecolor('gray')

    # # 刻度加粗
    # for tick in ax.get_xticklabels() + ax.get_yticklabels():
    #     tick.set_fontweight('bold')

    # # 黑色边框 (Spines)
    # ax.set_facecolor('white')
    # for spine in ax.spines.values():
    #     spine.set_edgecolor('black')
    #     spine.set_linewidth(1.0)

    # # 保存与显示
    # plt.tight_layout()
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # save_file = os.path.join(save_path, 'tsne_style_replication_rgb_and_nir.png')
    # plt.savefig(save_file, dpi=300, bbox_inches='tight')
    # print(f"Visualization saved to: {save_file}")
    # plt.show()

    
    ###############################
    # # --- 2. 筛选 20 个共同 ID ---
    # common_ids = sorted(list(set(g_ids) & set(nir_ids) & set(cp_ids) & set(sk_ids) & set(txt_ids)))
    # if len(common_ids) < 10:
    #     selected_ids = common_ids
    # else:
    #     selected_ids = random.sample(common_ids, 10)
    
    # # 建立 ID 到颜色的映射 (使用 tab20 调色盘，确保 20 个 ID 颜色各不相同)
    # unique_colors = plt.cm.tab20(np.linspace(0, 1, len(selected_ids)))
    # id_to_color = {pid: unique_colors[i] for i, pid in enumerate(selected_ids)}

    # def filter_data(ids, feats, target_ids):
    #     mask = np.isin(ids, target_ids)
    #     return ids[mask], feats[mask]



    ############################################################
    # # # 提取过滤后的数据
    # # f_g_ids, f_g_feats = filter_data(g_ids, g_feats, selected_ids)
    # # f_nir_ids, f_nir_feats = filter_data(nir_ids, nir_feats, selected_ids)
    # # f_cp_ids, f_cp_feats = filter_data(cp_ids, cp_feats, selected_ids)
    # # f_sk_ids, f_sk_feats = filter_data(sk_ids, sk_feats, selected_ids)
    # # f_txt_ids, f_txt_feats = filter_data(txt_ids, txt_feats, selected_ids)

    # # # 合并用于 TSNE
    # X = np.vstack([g_feats, nir_feats, cp_feats, sk_feats, text_feats])
    # all_ids = np.concatenate([gids_all, ids_all, ids_all, ids_all, ids_all])
    # all_mods = (['RGB']*len(gids_all) + ['NIR']*len(ids_all) + ['ColorPencil']*len(ids_all) + ['Sketch']*len(ids_all) + ['Text']*len(ids_all))

    # # --- 3. 执行 TSNE ---
    # print(f"Running TSNE on {len(X)} samples...")
    # tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=42)
    # X_2d = tsne.fit_transform(X)

    # # --- 4. 绘图设置 ---
    # # 定义不同模态的形状 (根据之前的需求，RGB用三叉星，其他用不同形状)
    # marker_map = {
    #     'RGB': '1',           # 三叉星 (Gallery)
    #     'NIR': 'o',           # 圆形
    #     'ColorPencil': 's',    # 正方形
    #     'Sketch': '^',        # 三角形
    #     'Text': 'p'           # 五边形
    # }
    
    # sns.set_style("white")
    # fig, ax = plt.subplots()

    # # 遍历所有点进行绘制
    # for i in range(len(X_2d)):
    #     pid = all_ids[i]
    #     mod = all_mods[i]
    #     ax.scatter(X_2d[i, 0], X_2d[i, 1], 
    #                color=id_to_color[pid], 
    #                marker=marker_map[mod], 
    #                s=30, alpha=0.8, edgecolors='none')

    # # --- 5. 创建双图例 (关键步骤) ---
    # # 5.1 模态图例 (展示形状，颜色固定为灰色)
    # mod_handles = []
    # for mod in marker_map:
    #     handle = mlines.Line2D([], [], color='white', marker=marker_map[mod], linestyle='None',
    #                            markersize=10, label=mod)
    #     mod_handles.append(handle)
    # legend1 = ax.legend(handles=mod_handles, title="Modalities", loc='upper left', 
    #                     prop={'weight': 'bold', 'size': 9}, bbox_to_anchor=(1, 1))

    # # 5.2 ID 图例 (展示颜色，形状固定为圆形)
    # # id_handles = []
    # # # 仅展示前 10 个 ID 以防图例过长，或者全部展示
    # # for pid in selected_ids:
    # #     handle = mlines.Line2D([], [], color=id_to_color[pid], marker='o', linestyle='None',
    # #                            markersize=8, label=f'ID: {pid}')
    # #     id_handles.append(handle)
    # # legend2 = ax.legend(handles=id_handles, title="Identities", loc='upper left', 
    # #                     prop={'size': 8}, bbox_to_anchor=(1, 0.7), ncol=2)
    
    # # 重新添加第一个图例（因为第二个 legend 会覆盖第一个）
    # ax.add_artist(legend1)

    # # --- 6. 细节美化 ---
    # ax.set_xlabel('t-SNE dimension 1', fontsize=12, fontweight='bold')
    # ax.set_ylabel('t-SNE dimension 2', fontsize=12, fontweight='bold')
    # for tick in ax.get_xticklabels() + ax.get_yticklabels():
    #     tick.set_fontweight('bold')
    
    # ax.set_facecolor('white')
    # for spine in ax.spines.values():
    #     spine.set_edgecolor('black')
    #     spine.set_linewidth(1.5)

    # plt.tight_layout()
    # save_file = os.path.join(save_path, 'ID_Modality_Combined_tsne.png')
    # plt.savefig(save_file, dpi=300, bbox_inches='tight')
    # print(f"Saved to {save_file}")
    # # plt.show()
