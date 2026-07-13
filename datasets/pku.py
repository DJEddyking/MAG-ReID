# import sys
# sys.path.append('/home/dj_2025/competitions/original_code备份_last_H20')

from json.encoder import py_encode_basestring
import os.path as op
from typing import List

from utils.iotools import read_json
from .bases_unireid import BaseDataset
# from bases_prcv import BaseDataset
from collections import  defaultdict

import numpy as np
import pdb

class PKU(BaseDataset):
    """
    OKU sketch
    """
    dataset_dir = 'PKUSketchRE-ID_V1'

    def __init__(self, root='', nlp_aug=False, verbose=True, trial=1):  ### pku要计算10次结果取平均 trial: 1-10
        super(PKU, self).__init__()
        self.dataset_dir = op.join(root, self.dataset_dir)
        self.img_dir = op.join(self.dataset_dir, 'photo/')
        self.simg_dir = op.join(self.dataset_dir, 'sketch/')
        # if nlp_aug:
        #     self.anno_path = op.join(self.dataset_dir, 'nlp_aug.json') 
        # else:
        #     self.anno_path = op.join(self.dataset_dir, 'reid_raw.json')   ### dj: 注意, 原始rgb有部分是bmp格式文件, 而sketch全部是jpg
        self.anno_path = op.join(self.dataset_dir, 'TriReID.txt')
        self._check_before_run()


        # self.train = {}
        self.train = defaultdict(list)
        self.train2 = []
        self.gallery = []
        self.query = []
        train_visible_path = op.join(self.dataset_dir,'idx/train_visible_{}.txt'.format(trial))  # 一共有10种划分, 然后取平均
        train_sketch_path = op.join(self.dataset_dir,'idx/train_sketch_{}.txt'.format(trial))
        test_visible_path = op.join(self.dataset_dir,'idx/test_visible_{}.txt'.format(trial))
        test_sketch_path = op.join(self.dataset_dir,'idx/test_sketch_{}.txt'.format(trial))

        ## 读取text: {1: ['xx', 'xx'], 2: [], ..}
        txt_file_list = open(self.anno_path, 'rt').read().splitlines()
        txt_file_id_txt_dict = {}
        for s in txt_file_list:
            _id = int(s.split('/')[0])
            if _id in txt_file_id_txt_dict:
                txt_file_id_txt_dict[_id].append(s.split('@')[-1])
            else:
                txt_file_id_txt_dict[_id] = [s.split('@')[-1]]

        #trainset RGB-0 sketch-1
        data_file_list = open(train_visible_path, 'rt').read().splitlines()
        file_label = [int(s.split(' ')[1]) for s in data_file_list]
        pid2label_img = {pid: label for label, pid in enumerate(np.unique(file_label))}
        file_cam = [int(s.split(' ')[2]) for s in data_file_list]
        pid2cam_img = {pid: label for label, pid in enumerate(np.unique(file_cam))}

        # 新建一个字典记录选择哪个text
        _train_count_dict = {}
        for j in range(len(data_file_list)):
            img = op.join(self.dataset_dir, data_file_list[j].split(' ')[0])
            id = int(data_file_list[j].split(' ')[1])
            cam = int(data_file_list[j].split(' ')[2])
            sty = int(data_file_list[j].split(' ')[3])
            if id in _train_count_dict:
                txt_id = 1
            else:
                txt_id = 0
            # self.train2.append((img, pid2label_img[id], 0 ))  # 0表示 rgb? 1表示sketch? 新增text
            # self.train2.append((pid2label_img[id], j, img, txt_file_id_txt_dict[txt_id]))  # pid, img id, img, text
            # self.train[pid2label_img[id]].append((img, pid2label_img[id], 0))
            self.train[pid2label_img[id]].append([pid2label_img[id], j, img, txt_file_id_txt_dict[id][txt_id]])  # {1: [[], []]}

        data_file_list = open(train_sketch_path, 'rt').read().splitlines()
        for j in range(len(data_file_list)):
            img = op.join(self.dataset_dir, data_file_list[j].split(' ')[0])
            id = int(data_file_list[j].split(' ')[1])
            cam = int(data_file_list[j].split(' ')[2])
            sty = int(data_file_list[j].split(' ')[3])
            # self.train2.append((img, pid2label_img[id], 1))
            # self.train[pid2label_img[id]].append((img, pid2label_img[id], 1))   ## 放在一个列表中
            self.train[pid2label_img[id]][0].append(img)
            # simg 和 text调换一下顺序
            self.train[pid2label_img[id]][0][-2], self.train[pid2label_img[id]][0][-1] = self.train[pid2label_img[id]][0][-1], self.train[pid2label_img[id]][0][-2]

            self.train[pid2label_img[id]][1].append(img)
            self.train[pid2label_img[id]][1][-2], self.train[pid2label_img[id]][1][-1] = self.train[pid2label_img[id]][1][-1], self.train[pid2label_img[id]][1][-2]
            

        # self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train2)


        # 测试的时候, gallery都是rgb, query是text or sketch,
        #query set: 每个id只有一张sketch, copy一份, 因为text和rgb都是有两份
        query_dict = {}
        data_file_list = open(test_sketch_path, 'rt').read().splitlines()
        for j in range(len(data_file_list)):
            img = op.join(self.dataset_dir, data_file_list[j].split(' ')[0])
            id = int(data_file_list[j].split(' ')[1])
            # cam = int(data_file_list[j].split(' ')[2])
            # sty = int(data_file_list[j].split(' ')[3])
            # self.query.append((img, id, 1))
            query_dict[id] = img

        all_query_and_gallery = {}
        _test_count_dict = {}
        #gallery set
        data_file_list = open(test_visible_path, 'rt').read().splitlines()
        file_cam = [int(s.split(' ')[2]) for s in data_file_list]
        # pid2cam_img = {pid: label for label, pid in enumerate(np.unique(file_cam))}
        for j in range(len(data_file_list)):
            img = op.join(self.dataset_dir, data_file_list[j].split(' ')[0])
            id = int(data_file_list[j].split(' ')[1])
            if id in _train_count_dict:
                txt_id = 1
            else:
                txt_id = 0
            # cam = int(data_file_list[j].split(' ')[2])
            # sty = int(data_file_list[j].split(' ')[3])
            # self.gallery.append((img, id, 0))
            if id in all_query_and_gallery:
                all_query_and_gallery[id].append((id, j, img, query_dict[id], txt_file_id_txt_dict[id][txt_id]))
            else:
                all_query_and_gallery[id] = [(id, j, img, query_dict[id], txt_file_id_txt_dict[id][txt_id])]
            # self.gallery.append((img, id, txt_file_id_txt_dict[txt_id]))
            

        
        # self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)

        
        # self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams  = self.get_imagedata_info(self.gallery)

        # self.train_annos, self.test_annos, self.val_annos = self._split_anno(self.anno_path)
        self.train_annos = self.train
        self.test_annos = all_query_and_gallery  ### 注意, 实际上是有100个query和gallery, 因为把sketch copy了一份
        self.val_annos = self.test_annos

        self.train, self.train_id_container = self._train_process_anno()
        self.test, self.test_id_container = self._test_process_anno(all_query_and_gallery)
        # self.val, self.val_id_container = self._process_anno(self.val_annos)
        self.val, self.val_id_container = self.test, self.test_id_container

        if verbose:
            self.logger.info("=> CUHK-PEDES Images and Captions are loaded")
            self.show_dataset_info()


    # def _split_anno(self, anno_path: str):
    #     train_annos, test_annos, val_annos = [], [], []
    #     annos = read_json(anno_path)
    #     for anno in annos:
    #         if anno['split'] == 'train':
    #             train_annos.append(anno)
    #         elif anno['split'] == 'test':
    #             test_annos.append(anno)
    #         else:
    #             val_annos.append(anno)
    #     return train_annos, test_annos, val_annos

    def _train_process_anno(self):
        dataset = []
        pid_container = set()
        for _id, tp in self.train.items():
            pid_container.add(_id)
            dataset.append(tp[0])
            dataset.append(tp[1])
        return dataset, pid_container

    def _test_process_anno(self, all_query_and_gallery):
        pid_container = set()
        # if training:
        #     dataset = []
            # image_id = 0

            # for anno in annos:
            #     pid = int(anno['id']) - 1 # make pid begin from 0
            #     img_path = op.join(self.img_dir, anno['file_path'])

            #     str_lst = anno['file_path'].split('.')
            #     if str_lst[-1] == 'bmp':
            #         sim_anno_file_name = ''.join(str_lst[: -1]) + '.jpg'
            #     else:
            #         sim_anno_file_name = anno['file_path']

            #     captions = anno['captions'] # caption list
            #     # if pid not in pid_container:
            #     simg_path = op.join(self.simg_dir, sim_anno_file_name)
            #     pid_container.add(pid)

            #     for caption in captions:
            #         dataset.append((pid, image_id, img_path, simg_path, caption))
            #     image_id += 1


            # for idx, pid in enumerate(pid_container):
            #     # check pid begin from 0 and no break
            #     assert idx == pid, f"idx: {idx} and pid: {pid} are not match"

            
            # return dataset, pid_container
        # else:
        dataset = {}
        img_paths = []
        simg_paths = []
        captions = []
        image_pids = []
        caption_pids = []
        image_id = 0
        image_ids = []
        simage_ids = []
        simage_pids = []

        for _id, tp in all_query_and_gallery.items():
            # pid = int(_id)
            for each_t in tp:
                pid, image_id, img_path, simg_path, caption = each_t
                # img_path = op.join(self.img_dir, anno['file_path'])
                img_paths.append(img_path)

                image_pids.append(pid)
                # caption_list = anno['captions'] # caption list
                image_ids.append(image_id)
            
                # str_lst = anno['file_path'].split('.')
                # if str_lst[-1] == 'bmp':
                #     sim_anno_file_name = ''.join(str_lst[: -1]) + '.jpg'
                # else:
                #     sim_anno_file_name = anno['file_path']

                # if pid not in pid_container:
                # simg_path =  op.join(self.simg_dir, sim_anno_file_name) 
                    #  simage_id = image_id
                    #  simage_pid = pid

                pid_container.add(pid)

                # for caption in caption_list:
                captions.append(caption)
                caption_pids.append(pid)

                simg_paths.append(simg_path)
                simage_ids.append(image_id)
                simage_pids.append(pid)

                # image_id += 1

        dataset = {
            "image_pids": image_pids,
            "img_paths": img_paths,
            "image_ids": image_ids,
            "simage_pids": simage_pids,
            "simg_paths": simg_paths,
            "simage_ids": simage_ids,
            "caption_pids": caption_pids,
            "captions": captions 
        }
        return dataset, pid_container


    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not op.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not op.exists(self.img_dir):
            raise RuntimeError("'{}' is not available".format(self.img_dir))
        if not op.exists(self.anno_path):
            raise RuntimeError("'{}' is not available".format(self.anno_path))



# PKU(root='/data/dj_2025')
