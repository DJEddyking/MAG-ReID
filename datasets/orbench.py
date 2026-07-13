import sys
sys.path.append('/home/dj_2025/competitions/original_code备份_last_H20')

import os.path as op
import random
from utils.iotools import read_json
from datasets.bases import BaseDataset
import os
import numpy as np

class ORBench(BaseDataset):
    dataset_dir = 'ORBench'

    def __init__(self, root='', verbose=True):
        super(ORBench, self).__init__()
        self.dataset_root = op.join(root,self.dataset_dir)
        self.train_anno_path = op.join(self.dataset_root, 'train_annos.json')
        self.test_anno_path = op.join(self.dataset_root, 'test_gallery_and_queries.json')

        self.train_annos = read_json(self.train_anno_path)
        self.test_annos = read_json(self.test_anno_path)
        self.nir_paths, self.cp_paths, self.sk_paths, self.vis_paths, self.txt_paths = self.get_paths()

        # dj : 修改train annos
        self.train_annos = self.random_sampling()
        # self.random_sampling()

        self.train, self.train_id_container = self._process_anno(self.train_annos, training=True)
        self.test, self.test_id_container = self._process_anno(self.test_annos)

        if verbose:
            self.logger.info("=> ORBench Images and Captions are loaded")
            self.show_dataset_info()

    def _expand_dataset(self, img_lst, max_len):
        if max_len > len(img_lst):
            img_lst.extend(
                list(map(str, np.random.choice(img_lst, size=max_len - len(img_lst), replace=max_len > 2 *len(img_lst))))
            )

    def random_sampling(self):
        """
        dj: 通过随机采样一张图像的方式, 避免nir, cp, sk三种模态的图像数量不均衡的问题, 最终所有模态的图像数量均相同
        """
        print("Using Sampling Processing...")
        # for anno in self.train_annos:
        #     real_identity = anno['file_path'].split('/')[1]
        #     anno['nir_path'] = random.sample(self.nir_paths[real_identity],1)[0]
        #     anno['cp_path'] = random.sample(self.cp_paths[real_identity], 1)[0]
        #     anno['sk_path'] = random.sample(self.sk_paths[real_identity], 1)[0]
        # 改为扩充采样方式
        # 先拿到所有id的vis 和 text
        all_vis = {}
        all_text = {}
        all_real_id_maps = {}
        for anno in self.train_annos:
            real_identity = anno['file_path'].split('/')[1]
            all_real_id_maps[anno['id']] = real_identity

            if real_identity not in all_vis.keys():
                all_vis[real_identity] = [anno['file_path']]
                all_text[real_identity] = [anno['caption']]
            all_vis[real_identity].append(anno['file_path'])
            all_text[real_identity].append(anno['caption'])
        # 随机采样
        new_train_annos = []
        # 对于每个id, 每个id都扩充到最长
        for pid, real_identity in all_real_id_maps.items():
            _max_len = max(len(self.nir_paths[real_identity]), len(self.cp_paths[real_identity]), 
                           len(self.sk_paths[real_identity]), len(self.vis_paths[real_identity]),len(self.txt_paths[real_identity]))
            self._expand_dataset(self.nir_paths[real_identity], _max_len)
            self._expand_dataset(self.cp_paths[real_identity], _max_len)
            self._expand_dataset(self.sk_paths[real_identity], _max_len)
            self._expand_dataset(self.vis_paths[real_identity], _max_len)
            self._expand_dataset(self.txt_paths[real_identity], _max_len)
            # 组合为新的anno
            for idx in range(_max_len):
                new_anno = {
                    'id': pid,
                    'file_path': self.vis_paths[real_identity][idx],
                    'caption': self.txt_paths[real_identity][idx],
                    'nir_path': self.nir_paths[real_identity][idx],
                    'cp_path': self.cp_paths[real_identity][idx],
                    'sk_path': self.sk_paths[real_identity][idx],
                    'split': 'train'
                }
                new_train_annos.append(new_anno)
        print("Using Max Sampling Completed!")
        return new_train_annos
        

    def get_paths(self):
        nir_paths = {}
        for nir_identity in os.listdir(os.path.join(self.dataset_root,'nir')):
            if nir_identity not in nir_paths.keys():
                nir_paths[nir_identity] = []
            identity_paths = os.listdir(os.path.join(self.dataset_root,'nir',nir_identity))
            for path in identity_paths:
                nir_paths[nir_identity].append(os.path.join('nir',nir_identity,path))

        cp_paths = {}
        for cp_identity in os.listdir(os.path.join(self.dataset_root,'cp')):
            if cp_identity not in cp_paths.keys():
                cp_paths[cp_identity] = []
            identity_paths = os.listdir(os.path.join(self.dataset_root,'cp', cp_identity))
            for path in identity_paths:
                cp_paths[cp_identity].append(os.path.join('cp', cp_identity, path))

        sk_paths = {}
        for sk_identity in os.listdir(os.path.join(self.dataset_root, 'sk')):
            if sk_identity not in sk_paths.keys():
                sk_paths[sk_identity] = []
            identity_paths = os.listdir(os.path.join(self.dataset_root, 'sk', sk_identity))
            for path in identity_paths:
                sk_paths[sk_identity].append(os.path.join('sk', sk_identity, path))

        vis_paths = {}
        for vis_identity in os.listdir(os.path.join(self.dataset_root, 'vis')):
            if vis_identity not in vis_paths.keys():
                vis_paths[vis_identity] = []
            identity_paths = os.listdir(os.path.join(self.dataset_root, 'vis', vis_identity))
            for path in identity_paths:
                vis_paths[vis_identity].append(os.path.join('vis', vis_identity, path))


        txt_paths = {}
        for anno in self.train_annos:
            real_identity = anno['file_path'].split('/')[1]
            if real_identity not in txt_paths.keys():
                txt_paths[real_identity] = [anno['caption']]
            txt_paths[real_identity].append(anno['caption'])

        return nir_paths, cp_paths, sk_paths, vis_paths, txt_paths

    def _process_anno(self, annos, training=False):
        pid_container = set()
        if training:
            dataset = []
            image_id = 0
            for anno in annos:
                pid = int(anno['id']) - 1  # make pid begin from 0
                pid_container.add(pid)
                rgb_path = op.join(self.dataset_root, anno['file_path'])
                nir_path = op.join(self.dataset_root, anno['nir_path'])
                cp_path = op.join(self.dataset_root, anno['cp_path'])
                sk_path = op.join(self.dataset_root, anno['sk_path'])

                caption = anno['caption']  # caption list
                dataset.append((pid, image_id, rgb_path, nir_path, cp_path, sk_path, caption))
                image_id += 1
            for idx, pid in enumerate(pid_container):
                # check pid begin from 0 and no break
                assert idx == pid, f"idx: {idx} and pid: {pid} are not match"
            return dataset, pid_container

        else:
            gallery_paths = []
            gallery_pids = []

            for anno in annos['RGB_GALLERY']:
                pid = int(anno[0])
                pid_container.add(pid)
                img_path = op.join(self.dataset_root, anno[1])
                gallery_paths.append(img_path)
                gallery_pids.append(pid)

            queries = annos.copy()
            queries.pop('RGB_GALLERY',None)

            for key in queries.keys():
                for i in range(len(queries[key])):
                    for j in range(len(queries[key][i])):
                        if type(queries[key][i][j]) == int:
                            continue
                        if '.jpg' in queries[key][i][j]:
                            queries[key][i][j] = op.join(self.dataset_root,queries[key][i][j])

            dataset = {
                "gallery_pids": gallery_pids,
                "gallery_paths": gallery_paths,
                "queries": queries
            }

            # 把相同的key合并在一起, 比如, CP+SK 和 SK + CP是一起的
            new_queries = {
                'TEXT': dataset['queries']['TEXT'],
                'NIR': dataset['queries']['NIR'],
                'CP': dataset['queries']['CP'],
                'SK': dataset['queries']['SK'],

                'TEXT+NIR': dataset['queries']['TEXT+NIR'] + [[i[0], i[2], i[1]] for i in dataset['queries']['NIR+TEXT']],
                'TEXT+SK': dataset['queries']['TEXT+SK'] + [[i[0], i[2], i[1]] for i in dataset['queries']['SK+TEXT']],
                'TEXT+CP': dataset['queries']['TEXT+CP'] + [[i[0], i[2], i[1]] for i in dataset['queries']['CP+TEXT']],

                'SK+CP': dataset['queries']['SK+CP'] + [[i[0], i[2], i[1]] for i in dataset['queries']['CP+SK']],
                'SK+NIR': dataset['queries']['SK+NIR'] + [[i[0], i[2], i[1]] for i in dataset['queries']['NIR+SK']],
                'CP+NIR': dataset['queries']['CP+NIR'] + [[i[0], i[2], i[1]] for i in dataset['queries']['NIR+CP']],

                'TEXT+CP+SK': dataset['queries']['TEXT+CP+SK'] + [[i[0], i[3], i[1], i[2]] for i in dataset['queries']['CP+SK+TEXT']] + \
                                                                      [[i[0], i[3], i[2], i[1]] for i in dataset['queries']['SK+CP+TEXT']],

                'TEXT+CP+NIR': [[i[0], i[3], i[1], i[2]] for i in dataset['queries']['CP+NIR+TEXT']]+ \
                                [[i[0], i[3], i[2], i[1]] for i in dataset['queries']['NIR+CP+TEXT']] + \
                                    [[i[0], i[1], i[3], i[2]] for i in dataset['queries']['TEXT+NIR+CP']],

                'TEXT+SK+NIR': [[i[0], i[3], i[1], i[2]] for i in dataset['queries']['SK+NIR+TEXT']] + \
                                [[i[0], i[3], i[2], i[1]] for i in dataset['queries']['NIR+SK+TEXT']] + \
                                [[i[0], i[1], i[3], i[2]] for i in dataset['queries']['TEXT+NIR+SK']],

                'CP+SK+NIR': [[i[0], i[1], i[3], i[2]] for i in dataset['queries']['CP+NIR+SK'] ] + \
                                [[i[0], i[2], i[3], i[1]] for i in dataset['queries']['NIR+CP+SK']] + \
                                [[i[0], i[3], i[1], i[2]] for i in dataset['queries']['SK+NIR+CP']],

                'TEXT+CP+SK+NIR': [[i[0], i[1], i[3], i[4], i[2]] for i in dataset['queries']['TEXT+NIR+CP+SK']] + \
                                  [[i[0], i[4], i[3], i[1], i[2]] for i in dataset['queries']['SK+NIR+CP+TEXT']] + \
                                  [[i[0], i[4], i[1], i[3], i[2]] for i in dataset['queries']['CP+NIR+SK+TEXT']] + \
                                  [[i[0], i[4], i[2], i[3], i[1]] for i in dataset['queries']['NIR+CP+SK+TEXT']]
            }
            assert len(new_queries['TEXT+NIR']) == len(dataset['queries']['TEXT+NIR']) + len(dataset['queries']['NIR+TEXT'])
            assert len(new_queries['TEXT+SK']) == len(dataset['queries']['TEXT+SK']) + len(dataset['queries']['SK+TEXT'])
            assert len(new_queries['TEXT+CP']) == len(dataset['queries']['TEXT+CP']) + len(dataset['queries']['CP+TEXT'])
            assert len(new_queries['SK+CP']) == len(dataset['queries']['SK+CP']) + len(dataset['queries']['CP+SK']) 
            assert len(new_queries['SK+NIR']) == len(dataset['queries']['SK+NIR']) + len(dataset['queries']['NIR+SK'])
            assert len(new_queries['CP+NIR']) == len(dataset['queries']['CP+NIR']) + len(dataset['queries']['NIR+CP'])
            assert len(new_queries['TEXT+CP+SK']) == len(dataset['queries']['TEXT+CP+SK']) + len(dataset['queries']['CP+SK+TEXT']) + len(dataset['queries']['SK+CP+TEXT'])
            assert len(new_queries['TEXT+CP+NIR']) == len(dataset['queries']['TEXT+NIR+CP']) + len(dataset['queries']['CP+NIR+TEXT']) + len(dataset['queries']['NIR+CP+TEXT'])
            assert len(new_queries['TEXT+SK+NIR']) == len(dataset['queries']['TEXT+NIR+SK']) + len(dataset['queries']['SK+NIR+TEXT']) + len(dataset['queries']['NIR+SK+TEXT'])
            assert len(new_queries['CP+SK+NIR']) == len(dataset['queries']['CP+NIR+SK']) + len(dataset['queries']['NIR+CP+SK']) + len(dataset['queries']['SK+NIR+CP'])
            assert len(new_queries['TEXT+CP+SK+NIR']) == len(dataset['queries']['NIR+CP+SK+TEXT']) + len(dataset['queries']['TEXT+NIR+CP+SK']) + len(dataset['queries']['SK+NIR+CP+TEXT']) + len(dataset['queries']['CP+NIR+SK+TEXT'])

            dataset['queries'] = new_queries

            return dataset, pid_container

# d = ORBench(root='/data/dj_2025')
# print(d.show_dataset_info())
