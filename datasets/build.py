import logging
import torch
import math
import random
import torchvision.transforms as T
from torch.utils.data import DataLoader
from .bases import (
    ImageTextDataset, GalleryDataset, SingleQueryTextDataset, SingleQueryVisionDataset,
    TwoQueryTextVisionDataset, TwoQueryVisionTextDataset, TwoQueryAllVisionDataset,
    ThreeQueryTextVisionVisionDataset, ThreeQueryVisionVisionTextDataset, ThreeQueryAllVisionDataset,
    FourQueryTextVisionVisionVisionDataset, FourQueryVisionVisionVisionTextDataset
)
from .bases_unireid import ImageTextDataset as ImageTextDatasetTri
from .bases_unireid import ImageDataset as ImageDatasetTri
from .bases_unireid import SketchTextDataset as SketchTextDatasetTri
from .bases_unireid import SketchDataset as SketchDatasetTri
from .bases_unireid import TextDataset as TextDatasetTri

from .bases_rnt import ImageDataset as ImageDatasetRNT
from .bases_rnt_text import ImageDataset as ImageDatasetRNT_Text

from .orbench import ORBench
from .sampler import RandomIdentitySampler
from .autoaugment import AutoAugment

from .cuhkpedes import CUHKPEDES
from .icfgpedes import ICFGPEDES
from .rstpreid import RSTPReid

from .rbgnt201 import RGBNT201
from .RGBNT201_Text import RGBNT201_Text

from .pku import PKU

__factory = {'ORBench': ORBench, 'CUHK-PEDES': CUHKPEDES, 'ICFG-PEDES': ICFGPEDES, 'RSTPReid': RSTPReid, 'RGBNT201': RGBNT201, 'RGBNT201_Text': RGBNT201_Text,
                'PKU-Sketch': PKU}



def build_transforms(img_size=(384, 128), aug=False, is_train=True):
    height, width = img_size

    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]

    transform_mix_aug_nir = [T.ColorJitter(brightness=0.3,contrast=0.3), T.GaussianBlur(21, sigma=(0.1, 3))]
    transform_mix_aug_rgb = [T.Grayscale(num_output_channels=3), T.GaussianBlur(21, sigma=(0.1, 3))]

    if not is_train:
        transform = T.Compose([
            T.Resize((height, width)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        return transform

    # transform for training
    if aug:
        # rgb/cp transform: 增加随机变为灰度图的增强
        transform = T.Compose([
            T.Resize((height, width)),
            T.RandomHorizontalFlip(0.5),
            T.Pad(10),
            T.RandomCrop((height, width)),
            # 给每个图像模态引入随机增强策略
            # T.RandomApply([AutoAugment()], p=0.5), # 效果不太行
            # 尝试加入颜色变化? 比如sketch就加彩色, rgb就灰度变化 ? 
            # T.RandomGrayscale(0.5),
            # T.RandomChoice(transform_mix_aug_rgb),

            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            T.RandomErasing(scale=(0.02, 0.4), value=mean),
        ])
        # nir/sk transform: 增加颜色抖动和高斯模糊
        transform_nir_sk = T.Compose([
            T.Resize((height, width)),
            T.RandomHorizontalFlip(0.5),
            T.Pad(10),
            T.RandomCrop((height, width)),

            T.RandomChoice(transform_mix_aug_nir), # 这个应该是有效的

            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            T.RandomErasing(scale=(0.02, 0.4), value=mean),
        ])
        
    else:
        transform = T.Compose([
            T.Resize((height, width)),
            T.RandomHorizontalFlip(0.5),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        transform_nir_sk = T.Compose([
            T.Resize((height, width)),
            T.RandomHorizontalFlip(0.5),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    return transform, transform_nir_sk


def collate(batch):
    keys = set([key for b in batch for key in b.keys()])
    # turn list of dicts data structure to dict of lists data structure
    dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

    batch_tensor_dict = {}
    for k, v in dict_batch.items():
        if isinstance(v[0], int):
            batch_tensor_dict.update({k: torch.tensor(v)})
        elif torch.is_tensor(v[0]):
            batch_tensor_dict.update({k: torch.stack(v)})
        else:
            raise TypeError(f"Unexpect data type: {type(v[0])} in a batch.")

    return batch_tensor_dict


def _create_query_datasets(test_queries, val_transforms):
    """创建所有查询类型的Dataset"""
    query_datasets = {}

    # 单模态查询
    single_modalities = ['NIR', 'CP', 'SK', 'TEXT']
    for modality in single_modalities:
        if modality == 'TEXT':
            query_datasets[modality] = SingleQueryTextDataset(test_queries[modality])
        else:
            query_datasets[modality] = SingleQueryVisionDataset(test_queries[modality], val_transforms)

    # 双模态查询
    two_modality_queries = {
        # 'NIR+CP': TwoQueryAllVisionDataset,
        'CP+NIR': TwoQueryAllVisionDataset,
        # 'NIR+SK': TwoQueryAllVisionDataset,
        'SK+NIR': TwoQueryAllVisionDataset,
        # 'NIR+TEXT': TwoQueryVisionTextDataset,
        'TEXT+NIR': TwoQueryTextVisionDataset,
        # 'CP+SK': TwoQueryAllVisionDataset,
        'SK+CP': TwoQueryAllVisionDataset,
        # 'CP+TEXT': TwoQueryVisionTextDataset,
        'TEXT+CP': TwoQueryTextVisionDataset,
        # 'SK+TEXT': TwoQueryVisionTextDataset,
        'TEXT+SK': TwoQueryTextVisionDataset,
    }

    for query_key, dataset_class in two_modality_queries.items():
        query_datasets[query_key] = dataset_class(test_queries[query_key], val_transforms)

    # 三模态查询
    three_modality_queries = {
        # 'NIR+CP+SK': ThreeQueryAllVisionDataset,
        'CP+SK+NIR': ThreeQueryAllVisionDataset,
        # 'SK+NIR+CP': ThreeQueryAllVisionDataset,
        # 'NIR+CP+TEXT': ThreeQueryVisionVisionTextDataset,
        # 'CP+NIR+TEXT': ThreeQueryVisionVisionTextDataset,
        'TEXT+CP+NIR': ThreeQueryTextVisionVisionDataset,
        # 'NIR+SK+TEXT': ThreeQueryVisionVisionTextDataset,
        # 'SK+NIR+TEXT': ThreeQueryVisionVisionTextDataset,
        'TEXT+SK+NIR': ThreeQueryTextVisionVisionDataset,
        # 'CP+SK+TEXT': ThreeQueryVisionVisionTextDataset,
        # 'SK+CP+TEXT': ThreeQueryVisionVisionTextDataset,
        'TEXT+CP+SK': ThreeQueryTextVisionVisionDataset,
    }

    for query_key, dataset_class in three_modality_queries.items():
        query_datasets[query_key] = dataset_class(test_queries[query_key], val_transforms)

    # 四模态查询
    four_modality_queries = {
        # 'NIR+CP+SK+TEXT': FourQueryVisionVisionVisionTextDataset,
        # 'CP+NIR+SK+TEXT': FourQueryVisionVisionVisionTextDataset,
        # 'SK+NIR+CP+TEXT': FourQueryVisionVisionVisionTextDataset,
        # 'TEXT+NIR+CP+SK': FourQueryTextVisionVisionVisionDataset,
        'TEXT+CP+SK+NIR': FourQueryTextVisionVisionVisionDataset,
    }

    for query_key, dataset_class in four_modality_queries.items():
        query_datasets[query_key] = dataset_class(test_queries[query_key], val_transforms)

    return query_datasets


def _create_dataloaders(query_datasets, test_batch_size, num_workers):
    """为所有查询Dataset创建DataLoader"""
    dataloaders = {}

    for query_key, dataset in query_datasets.items():
        dataloaders[query_key] = DataLoader(
            dataset,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=num_workers
        )

    return dataloaders


def build_dataloader(args, tranforms=None):
    logger = logging.getLogger("ORBench.dataset")

    num_workers = args.num_workers
    dataset = __factory[args.dataset_name](root=args.root_dir)
    num_classes = len(dataset.train_id_container)

    # if tranforms:
    #     val_transforms = tranforms
    # else:
    val_transforms = build_transforms(img_size=args.img_size, is_train=False)

    if args.training:
        train_transforms, train_nir_sk_tranforms = build_transforms(
            img_size=args.img_size, aug=args.img_aug, is_train=True
        )

        train_set = ImageTextDataset(
            dataset.train, train_transforms,  train_nir_sk_tranforms,  text_length=args.text_length,
            data_use_inverse=args.data_use_inverse  ### 增加是否用inversenet
        )

        logger.info('using random sampler')
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True if args.sampler == 'random' else False,
            num_workers=num_workers,
            collate_fn=collate,

            sampler=RandomIdentitySampler(dataset.train, args.batch_size, args.num_instance) if args.sampler == 'identity' else None
        )

        # use test set as validate set
        ds = dataset.test
        test_gallery_set = GalleryDataset(
            ds['gallery_pids'], ds['gallery_paths'], val_transforms
        )
        test_gallery_loader = DataLoader(
            test_gallery_set, batch_size=args.test_batch_size,
            shuffle=False, num_workers=num_workers
        )

        # 创建所有查询的Dataset和DataLoader
        query_datasets = _create_query_datasets(ds['queries'], val_transforms)
        query_dataloaders = _create_dataloaders(
            query_datasets, args.test_batch_size, num_workers
        )
        # 合并相同模态
        return_order = [
            'train_loader', 'test_gallery_loader',
            # 单模态
            'NIR', 'CP', 'SK', 'TEXT',
            # 双模态
            'CP+NIR', 'SK+NIR', 'TEXT+NIR', 'SK+CP',  'TEXT+CP', 'TEXT+SK',
            # 三模态
            'CP+SK+NIR', 'TEXT+CP+NIR',  'TEXT+SK+NIR', 'TEXT+CP+SK',
            # 四模态
            'TEXT+CP+SK+NIR',
            'num_classes'
        ]

        result = [train_loader, test_gallery_loader]
        for key in return_order[2:-1]:  # 跳过train_loader, test_gallery_loader和num_classes
            result.append(query_dataloaders[key])
        result.append(num_classes)

        return tuple(result)

    else:
        # build dataloader for testing
        ds = dataset.test

        test_gallery_set = GalleryDataset(
            ds['gallery_pids'], ds['gallery_paths'], val_transforms
        )
        test_gallery_loader = DataLoader(
            test_gallery_set, batch_size=args.test_batch_size,
            shuffle=False, num_workers=num_workers
        )

        # 创建所有查询的Dataset和DataLoader
        query_datasets = _create_query_datasets(ds['queries'], val_transforms)
        query_dataloaders = _create_dataloaders(
            query_datasets, args.test_batch_size, num_workers
        )
        # 合并相同模态
        return_order = [
            'test_gallery_loader',
            # 单模态
            'NIR', 'CP', 'SK', 'TEXT',
            # 双模态
            'CP+NIR', 'SK+NIR', 'TEXT+NIR', 'SK+CP',  'TEXT+CP', 'TEXT+SK',
            # 三模态
            'CP+SK+NIR', 'TEXT+CP+NIR',  'TEXT+SK+NIR', 'TEXT+CP+SK',
            # 四模态
            'TEXT+CP+SK+NIR',
            'num_classes'
        ]

        result = [test_gallery_loader]
        for key in return_order[1:-1]:  # 跳过test_gallery_loader和num_classes
            result.append(query_dataloaders[key])
        result.append(num_classes)

        return tuple(result)


def build_dataloader_Tri_reid(args, tranforms=None):
    logger = logging.getLogger("ORBench.dataset")

    num_workers = args.num_workers
    dataset = __factory[args.dataset_name](root=args.root_dir)

    # if args.training:
    train_transforms, train_transforms_sk  = build_transforms(img_size=args.img_size, aug=args.img_aug, is_train=True)
    val_transforms = build_transforms(img_size=args.img_size,  is_train=False)

    train_set = ImageTextDatasetTri(dataset.train, train_transforms, train_transforms_sk, text_length=args.text_length)

    num_classes = len(dataset.train_id_container)

    logger.info('using random sampler')
    train_loader = DataLoader(train_set,
                                batch_size=args.batch_size,
                                shuffle=True if args.sampler == 'random' else False,
                                num_workers=num_workers,
                                collate_fn=collate)

    # use test set as validate set
    ds = dataset.val if args.val_dataset == 'val' else dataset.test

    ## rgb作为gallery
    val_img_set = ImageDatasetTri(ds['image_pids'], ds['img_paths'], ds['image_ids'],
                                val_transforms)
    # text和sketch一起作为query
    val_txt_sketch_set = SketchTextDatasetTri(ds['simg_paths'], ds['simage_ids'], ds['caption_pids'],
                                ds['captions'], val_transforms,
                                text_length=args.text_length)# 返回 pid, img, caption
    # 单独的sketch作为query
    val_sketch_set = SketchDatasetTri(ds['simg_paths'], ds['simage_ids'], ds['simage_pids'], val_transforms, 
                                        is_pku=True if args.dataset_name == 'PKU-Sketch' else False)   # pid, img

    # 单独的text作为query
    val_txt_set = TextDatasetTri(ds['caption_pids'], ds['captions'], text_length=args.text_length)  ## 返回pid, caption
                            

    val_img_loader = DataLoader(val_img_set,
                                batch_size=args.test_batch_size,
                                shuffle=False,
                                num_workers=num_workers)
    val_txt_loader = DataLoader(val_txt_set,
                                batch_size=args.test_batch_size,
                                shuffle=False,
                                num_workers=num_workers)
    val_sketch_loader = DataLoader(val_sketch_set,
                                batch_size=args.test_batch_size,
                                shuffle=False,
                                num_workers=num_workers)

    val_txt_sketch_loader = DataLoader(val_txt_sketch_set,
                                batch_size=args.test_batch_size,
                                shuffle=False,
                                num_workers=num_workers)

    return train_loader, val_img_loader, val_txt_loader, val_sketch_loader, val_txt_sketch_loader, num_classes

    ## test的代码暂时没补充
    # else:
    #     # build dataloader for testing
    #     if tranforms:
    #         test_transforms = tranforms
    #     else:
    #         test_transforms = build_transforms(img_size=args.img_size,
    #                                            is_train=False)

    #     ds = dataset.test
    #     test_img_set = ImageDataset(ds['image_pids'], ds['img_paths'], ds['image_ids'],
    #                                 test_transforms)
    #     test_txt_set = SketchTextDataset(ds['simg_paths'], ds['simage_ids'], ds['caption_pids'],
    #                                ds['captions'], test_transforms, 
    #                                text_length=args.text_length)
    #     test_sketch_set = SketchDataset(ds['simg_paths'], ds['simage_ids'], ds['simage_pids'], test_transforms)

    #     test_img_loader = DataLoader(test_img_set,
    #                                  batch_size=args.test_batch_size,
    #                                  shuffle=False,
    #                                  num_workers=num_workers)
    #     test_txt_loader = DataLoader(test_txt_set,
    #                                  batch_size=args.test_batch_size,
    #                                  shuffle=False,
    #                                  num_workers=num_workers)
    #     test_sketch_loader = DataLoader(test_sketch_set,
    #                                 batch_size=args.batch_size,
    #                                 shuffle=False,
    #                                 num_workers=num_workers)

    #     return test_img_loader, test_txt_loader, test_sketch_loader


def train_rnt_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs, pids, camids, viewids, _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    RGB_list = []
    NI_list = []
    TI_list = []

    for img in imgs:
        RGB_list.append(img[0])
        NI_list.append(img[1])
        TI_list.append(img[2])

    RGB = torch.stack(RGB_list, dim=0)
    NI = torch.stack(NI_list, dim=0)
    TI = torch.stack(TI_list, dim=0)
    imgs = {'RGB': RGB, "NI": NI, "TI": TI}
    return imgs, pids, camids, viewids,_


def val_rnt_collate_fn(batch):
    imgs, pids, camids, viewids, img_paths = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    camids_batch = camids
    RGB_list = []
    NI_list = []
    TI_list = []

    for img in imgs:
        RGB_list.append(img[0])
        NI_list.append(img[1])
        TI_list.append(img[2])

    RGB = torch.stack(RGB_list, dim=0)
    NI = torch.stack(NI_list, dim=0)
    TI = torch.stack(TI_list, dim=0)
    imgs = {'RGB': RGB, "NI": NI, "TI": TI}
    # return imgs, pids, camids, camids_batch, viewids, img_paths  ### 在collate的时候, 把一个batch的imgs拼接起来, 然后改为一个字典装
    return pids, imgs, camids


def train_rnt_text_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs, pids, camids, viewids, _, r_text, n_text, t_text = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    RGB_list = []
    NI_list = []
    TI_list = []

    for img in imgs:
        RGB_list.append(img[0])
        NI_list.append(img[1])
        TI_list.append(img[2])

    RGB = torch.stack(RGB_list, dim=0)
    NI = torch.stack(NI_list, dim=0)
    TI = torch.stack(TI_list, dim=0)
    imgs = {'RGB': RGB, "NI": NI, "TI": TI}
    text = {'rgb_text': torch.stack(r_text),
            'ni_text': torch.stack(n_text),
            'ti_text': torch.stack(t_text)}
    return imgs, pids, camids, viewids, _, text


def val_rnt_text_collate_fn(batch):
    imgs, pids, camids, viewids, img_paths, r_text, n_text, t_text = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    camids_batch = camids
    RGB_list = []
    NI_list = []
    TI_list = []

    for img in imgs:
        RGB_list.append(img[0])
        NI_list.append(img[1])
        TI_list.append(img[2])

    RGB = torch.stack(RGB_list, dim=0)
    NI = torch.stack(NI_list, dim=0)
    TI = torch.stack(TI_list, dim=0)
    imgs = {'RGB': RGB, "NI": NI, "TI": TI}
    text = {'rgb_text': torch.stack(r_text),
            'ni_text': torch.stack(n_text),
            'ti_text': torch.stack(t_text)}  ## 推理暂时不用texts
    # return imgs, pids, camids, camids_batch, viewids, img_paths, text
    return  pids, imgs,camids, text



def build_dataloader_rnt(args):
    """
    RNT -> RNT
    """
    num_workers = args.num_workers
    dataset = __factory[args.dataset_name](root=args.root_dir)
    # num_classes = len(dataset.train_id_container)

    val_transforms = build_transforms(img_size=args.img_size, is_train=False)

    # if args.training:
    train_transforms, train_nir_sk_tranforms = build_transforms(
        img_size=args.img_size, aug=args.img_aug, is_train=True
    )

    train_set = ImageDatasetRNT_Text(dataset.train, train_transforms, train_nir_sk_tranforms, mask_token=True) \
                if '_Text' in args.dataset_name else ImageDatasetRNT(dataset.train, train_transforms, train_nir_sk_tranforms)
    # train_set_normal = ImageDatasetRNT(dataset.train, val_transforms)  ### 训练数据, 但是用的测试的transforms
    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids

    # if 'triplet' in cfg.DATALOADER.SAMPLER:
    #         train_loader = DataLoader(
    #             train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
    #             sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
    #             num_workers=num_workers, collate_fn=train_rnt_collate_fn,
    #         )
    # elif cfg.DATALOADER.SAMPLER == 'softmax':
    # print('using softmax sampler')
    train_loader = DataLoader(
        train_set, 
        batch_size=args.batch_size, 
        num_workers=num_workers, 
        shuffle=True if args.sampler == 'random' else False,
        collate_fn=train_rnt_text_collate_fn if '_Text' in args.dataset_name else train_rnt_collate_fn,

        sampler=RandomIdentitySampler(dataset.train, args.batch_size, args.num_instance) if args.sampler == 'identity' else None # 数量太少的数据集, 有必要加上triplet loss 
    )
    # else:
    #     print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Only use for grad-cam when fixed samples need for different modalities
    # train_loader = DataLoader(
    #     train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
    #     collate_fn=train_collate_fn
    # )
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    val_query_set = ImageDatasetRNT_Text(dataset.query, val_transforms, val_transforms) \
        if '_Text' in args.dataset_name else ImageDatasetRNT(dataset.query, val_transforms, val_transforms)
    val_gallery_set = ImageDatasetRNT_Text(dataset.gallery, val_transforms, val_transforms) \
        if '_Text' in args.dataset_name else ImageDatasetRNT(dataset.gallery, val_transforms, val_transforms)

    val_query_loader = DataLoader(
        val_query_set, 
        batch_size=args.test_batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        collate_fn=val_rnt_text_collate_fn if '_Text' in args.dataset_name else val_rnt_collate_fn
    )
    val_gallery_loader = DataLoader(
        val_gallery_set, 
        batch_size=args.test_batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        collate_fn=val_rnt_text_collate_fn if '_Text' in args.dataset_name else val_rnt_collate_fn
    )
    # train_loader_normal = DataLoader(
    #     train_set_normal, batch_size=args.test_batch_size, shuffle=False, num_workers=num_workers,
    #     collate_fn=val_rnt_collate_fn
    # )
    # return train_loader, train_loader_normal, val_loader, len(dataset.query), num_classes, cam_num, view_num
    return train_loader, val_query_loader, val_gallery_loader, num_classes, len(dataset.query)  ### 三个Loader都包含RNT三个模态, 以字典的形式存放
