import sys

import os
import random
import itertools
import functools
import numpy as np

import os.path as op
from typing import List

from utils.iotools import read_json
from .bases_prcv import BaseDataset


class ORBench_PRCV(BaseDataset):
    """
    ORBench_PRCV

    Reference:
    ReID5o: Achieving Omni Multi-modal Person Re-identification in a Single Model

    Dataset statistics:
    # identities: 1000
    # images: 45,113 (RGB) + 26,071 (IR) + 18,000 (color pencil) + 18,000 (sketch)
    # text: 45,113
    # cameras: no define
    
    note that, the gallery is all RGB, and the competition has restricted the query modalities, from 2 to 4.
    """
    dataset_dir = ''

    def __init__(self, root='', nlp_aug=False, verbose=True):
        super(ORBench_PRCV, self).__init__()
        self.dataset_dir = op.join(root, self.dataset_dir)
        # self.train_dir = op.join(self.dataset_dir, 'train')
        # self.cp_img_dir = op.join(self.dataset_dir, 'train/cp/')
        # # infrared
        # self.nir_img_dir = op.join(self.dataset_dir, 'train/nir/')
        # # sketch
        # self.sk_img_dir = op.join(self.dataset_dir, 'train/sk/')
        # # rgb
        # self.vis_img_dir = op.join(self.dataset_dir, 'train/vis/')
        self.test_dir = op.join(self.dataset_dir, 'test/')
        
        # # text
        # self.anno_path = op.join(self.dataset_dir, 'train/merge_text_annos.json')   ##### 
        self.test_anno_path = op.join(self.dataset_dir, 'test/test_queries.json')  ### 改为 val/val_queries.json ==> test/test_queries.json
        
        self._check_before_run()

        # self.train_annos, self.val_annos = self._split_train_anno(self.anno_path)
        self.test_annos = self._split_val_anno(self.test_anno_path)

        # self.train, self.val, self.train_id_container, self.val_id_container = self._process_anno(self.train_annos, training=True)
        self.test = self._process_anno_test(self.test_annos)
        self.test_id_container = set()

        if verbose:
            self.logger.info("=> ORBench_PRCV only test Images and Captions are loaded")
            # self.show_dataset_info()

        # TODO use prettytable print comand line table
        from prettytable import PrettyTable
        self.logger.info(f"{self.__class__.__name__} Dataset statistics:")
        table = PrettyTable(['subset', 'ids', 'images', 'captions'])
        # table.add_row(
            # ['train', num_train_pids, num_train_imgs, num_train_captions])
        table.add_row(
            ['test', 0, len(self.test_annos), 0])
        # table.add_row(['val', num_val_pids, num_val_imgs, num_val_captions])
        self.logger.info('\n' + str(table))
        # print('\n' + str(table))
        
    # def _split_train_anno(self, anno_path):
    #     """训练和测试用相同数据集"""
    #     train_annos, test_annos = [], []
    #     annos = read_json(anno_path)
    #     for anno in annos:
    #         if anno['split'] == 'train' or anno['split'] == 'val':
    #             train_annos.append(anno)
    #             test_annos.append(anno)
    #     return train_annos, test_annos
    
    def _split_val_anno(self, anno_path: str):
        val_annos = read_json(anno_path)
        return val_annos

  
    def _expand_dataset(self, img_lst, max_len):
        if max_len > len(img_lst):
            img_lst.extend(
                list(map(str, np.random.choice(img_lst, size=max_len - len(img_lst), replace=max_len > 2 *len(img_lst))))
            )
            
    def _process_anno_test(self, annos):
        """测试数据形式: 
        {
            'onemodal_NIR': [[1, 'nir/xxx.jpg'], [], ...],
            'twomodal_xx_xx': [[x, 'xxx', 'xxx'], [...], ...],
            ...
        }
        
        """
        dataset_test_dict = {}
        
        for anno_test in annos:
            
            query_type = anno_test['query_type']
            query_id = anno_test['query_idx']
            contents = anno_test['content']
            if query_type not in dataset_test_dict:
                dataset_test_dict[query_type] = []

            dataset_test_dict[query_type].append([query_id])
            dataset_test_dict[query_type][-1].extend(contents)
        
        res_dict = {}
        for modal_name, v_lst in dataset_test_dict.items():
            res_dict[modal_name] = {}
            id_lst = [d[0] for d in v_lst]
            res_dict[modal_name]['query_idxs'] = id_lst
            if 'onemodal' in modal_name:
                # 0903: text错误了, 增加判断
                data_lst = [d[1] for d in v_lst] if 'text' in modal_name.lower() else [op.join(self.test_dir, d[1]) for d in v_lst]
                res_dict[modal_name][modal_name.split('_')[-1].lower()] = data_lst
                
            elif 'twomodal' in modal_name:
                data_lst_1 = [d[1] for d in v_lst] if 'text' in modal_name.split('_')[-2].lower() else [op.join(self.test_dir, d[1]) for d in v_lst]
                data_lst_2 = [d[2] for d in v_lst] if 'text' in modal_name.split('_')[-1].lower() else [op.join(self.test_dir, d[2]) for d in v_lst]
                res_dict[modal_name][modal_name.split('_')[-2].lower()] = data_lst_1
                res_dict[modal_name][modal_name.split('_')[-1].lower()] = data_lst_2
                
            elif 'threemodal' in modal_name:
                data_lst_1 = [d[1] for d in v_lst] if 'text' in modal_name.split('_')[-3].lower() else [op.join(self.test_dir, d[1]) for d in v_lst]
                data_lst_2 = [d[2] for d in v_lst] if 'text' in modal_name.split('_')[-2].lower() else [op.join(self.test_dir, d[2]) for d in v_lst]
                data_lst_3 = [d[3] for d in v_lst] if 'text' in modal_name.split('_')[-1].lower() else [op.join(self.test_dir, d[3]) for d in v_lst]
                res_dict[modal_name][modal_name.split('_')[-3].lower()] = data_lst_1
                res_dict[modal_name][modal_name.split('_')[-2].lower()] = data_lst_2
                res_dict[modal_name][modal_name.split('_')[-1].lower()] = data_lst_3
                
            elif 'fourmodal' in modal_name:
                data_lst_1 = [d[1] for d in v_lst] if 'text' in modal_name.split('_')[-4].lower() else [op.join(self.test_dir, d[1]) for d in v_lst]
                data_lst_2 = [d[2] for d in v_lst] if 'text' in modal_name.split('_')[-3].lower() else [op.join(self.test_dir, d[2]) for d in v_lst]
                data_lst_3 = [d[3] for d in v_lst] if 'text' in modal_name.split('_')[-2].lower() else [op.join(self.test_dir, d[3]) for d in v_lst]
                data_lst_4 = [d[4] for d in v_lst] if 'text' in modal_name.split('_')[-1].lower() else [op.join(self.test_dir, d[4]) for d in v_lst]
                res_dict[modal_name][modal_name.split('_')[-4].lower()] = data_lst_1
                res_dict[modal_name][modal_name.split('_')[-3].lower()] = data_lst_2
                res_dict[modal_name][modal_name.split('_')[-2].lower()] = data_lst_3
                res_dict[modal_name][modal_name.split('_')[-1].lower()] = data_lst_4

        # gallery
        g_lst = sorted([i.split('.')[0].zfill(4) for i in os.listdir(os.path.join(self.test_dir, 'gallery'))])
        res_dict['gallery'] = {
            'rgb': [os.path.join(self.test_dir, 'gallery', str(int(i)) + '.jpg') for i in g_lst],
            'idxs': [int(i) for i in g_lst]
        }
        assert [int(i) for i in g_lst] == list(map(int, list(np.arange(1, len(g_lst)+1))))

        return res_dict
    
            
    def _process_anno(self, annos: List[dict], training=False):
        """
        需要构建多种模态组合的情况: 由于text,nir,sk,cp相同id的样本个数不相同, 因此需要选择一下, 先随机选择
        单模态(4种, 全量给出), 
        双模态(6种组合, text-nir, text-sk, text-cp, nir-sk, nir-cp, sk-cp), 
        三模态组合(3种, text-nir-sk, text-cp-sk, nir-sk-cp), 
        4模态组合(1种, 但是, 每种组合会随机选择, !! 组合方式会有非常多种!! 以text-rgb的个数为基准, 其余重复抽样)
        """
        pid_container = set()
        chosen_dict = {
            'nir': {},
            'sk': {},
            'cp': {},
            'rgb': {},
            'text': {},
        }
        
        _pid_real_id_mapping = {}
        for anno_pre in annos:
            pid = int(anno_pre['id'])
            pid_container.add(pid)
            _pid_real_id_mapping[anno_pre['file_path'].split('/')[1]] = pid
            if pid in chosen_dict['rgb']:
                chosen_dict['rgb'][pid].append(op.join(self.train_dir, anno_pre['file_path']))
                chosen_dict['text'][pid].append(anno_pre['caption'])
            else:
                chosen_dict['rgb'][pid] = [op.join(self.train_dir, anno_pre['file_path'])]
                chosen_dict['text'][pid] = [anno_pre['caption']]
        
        _pid_for_val = list(map(int, np.random.choice(list(pid_container), size=10)))
        _val_data_map = {
            'nir': {},
            'sk': {},
            'cp': {},
            'text': {},
        }
        val_all_pid = []
        for _real_id, _pid in _pid_real_id_mapping.items():
            nir_img_lst = [op.join(self.nir_img_dir, _real_id, x) for x in os.listdir(op.join(self.nir_img_dir, _real_id))]
            sk_img_lst = [op.join(self.sk_img_dir, _real_id, x) for x in  os.listdir(op.join(self.sk_img_dir, _real_id))]
            cp_img_lst = [op.join(self.cp_img_dir, _real_id, x) for x in  os.listdir(op.join(self.cp_img_dir, _real_id))]
            rgb_img_lst = chosen_dict['rgb'][_pid]
            text_lst = chosen_dict['text'][_pid]
            _max_len = max(len(nir_img_lst), len(sk_img_lst), len(cp_img_lst), len(rgb_img_lst), len(text_lst))
            _min_len = min(len(nir_img_lst), len(sk_img_lst), len(cp_img_lst), len(rgb_img_lst), len(text_lst))
            if _pid in _pid_for_val:
                _val_data_map['nir'][_pid] = nir_img_lst[-_min_len:]
                _val_data_map['sk'][_pid] = sk_img_lst[-_min_len:]
                _val_data_map['cp'][_pid] = cp_img_lst[-_min_len:]
                _val_data_map['text'][_pid] = text_lst[-_min_len:]
                val_all_pid.extend([_pid] * _min_len)
            
            self._expand_dataset(nir_img_lst, _max_len)
            self._expand_dataset(sk_img_lst, _max_len)
            self._expand_dataset(cp_img_lst, _max_len)
            self._expand_dataset(rgb_img_lst, _max_len)
            self._expand_dataset(text_lst, _max_len)
            
            chosen_dict['rgb'][_pid] = rgb_img_lst
            chosen_dict['text'][_pid] = text_lst
            chosen_dict['nir'][_pid] = nir_img_lst
            chosen_dict['sk'][_pid] = sk_img_lst
            chosen_dict['cp'][_pid] = cp_img_lst
        
        dataset_train = []
        all_nir = list(itertools.chain(*[v for _, v in chosen_dict['nir'].items()]))
        all_sk = list(itertools.chain(*[v for _, v in chosen_dict['sk'].items()]))
        all_cp = list(itertools.chain(*[v for _, v in chosen_dict['cp'].items()]))
        all_rgb = list(itertools.chain(*[v for _, v in chosen_dict['rgb'].items()]))
        all_text = list(itertools.chain(*[v for _, v in chosen_dict['text'].items()]))
        _mapping = {pid: idx for idx, pid in enumerate(pid_container)}
        
        all_pid = list(itertools.chain(*[[_mapping.get(k)] * len(v) for k, v in chosen_dict['rgb'].items()]))
        
        for image_id, (pid, rgb, sk, cp, nir, text) in enumerate(zip(*[all_pid, all_rgb, all_sk, all_cp, all_nir, all_text])):
            dataset_train.append(
                [
                    pid, 
                    image_id, 
                    rgb, 
                    sk,
                    cp,
                    nir,
                    text,
                ]
            )
        val_all_pid = [_mapping.get(i) for i in val_all_pid]
        val_all_nir = list(itertools.chain(*[v for _, v in _val_data_map['nir'].items()]))
        val_all_sk = list(itertools.chain(*[v for _, v in _val_data_map['sk'].items()]))
        val_all_cp = list(itertools.chain(*[v for _, v in _val_data_map['cp'].items()]))
        val_all_text = list(itertools.chain(*[v for _, v in _val_data_map['text'].items()]))
        
        image_ids = [i for i in range(len(val_all_cp))]

        rgb_image_ids = [i for i in range(len(all_rgb))]

        caption_ids = [i for i in range(len(val_all_text))]

        dataset_val = {
            'nir_img_pids': val_all_pid,
            'nir_img_paths': val_all_nir,
            
            'sk_img_pids': val_all_pid,
            'sk_img_paths': val_all_sk,
            
            'cp_img_pids': val_all_pid,
            'cp_img_paths': val_all_cp,
            
            'captions_pids': val_all_pid,
            'captions': val_all_text,
            'caption_ids': caption_ids,  #### 

            'image_ids': image_ids,

            # gallery
            'rgb_img_pids': all_pid,
            'rgb_img_paths': all_rgb,

            'rgb_image_ids': rgb_image_ids
        }
        return dataset_train, dataset_val,  list(_mapping.values()), set(val_all_pid)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        for k, v in self.__dict__.items():
            if k.startswith('_'):
                continue
            if '_img_dir' in k or '_path' in k:
                if not op.exists(v):
                    raise RuntimeError("'{}' is not available".format(v))
