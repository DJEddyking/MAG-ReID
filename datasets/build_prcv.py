import logging
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from datasets.sampler import RandomIdentitySampler
from datasets.sampler_ddp import RandomIdentitySampler_DDP
from torch.utils.data.distributed import DistributedSampler

from utils.comm import get_world_size

from .bases_prcv import (ImageDataset, SketchDataset, ImageTextMSMDataset, ImageTextMSMMLMDataset, 
                    TextDataset, ImageTextDataset, ImageTextMCQDataset, ImageTextMaskColorDataset,
                      ImageTextMLMDataset, ImageTextMCQMLMDataset, ImageImageDataset, ImageImageTextDataset,
                      ImageImageImageDataset, ImageImageImageTextDataset,
                    SketchTextDataset)

from .f30k import F30K
from .cuhkpedes import CUHKPEDES
from .icfgpedes import ICFGPEDES
from .rstpreid import RSTPReid
from .autoaugment import AutoAugment

from .orbench_prcv import ORBench_PRCV
__factory = {'CUHK-PEDES': CUHKPEDES, 'ICFG-PEDES': ICFGPEDES, 'F30K': F30K, 'RSTPReid': RSTPReid, 'ORBench_PRCV': ORBench_PRCV}


def build_transforms(img_size=(384, 128), aug=False, is_train=True):
    height, width = img_size

    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]

    if not is_train:
        transform = T.Compose([
            T.Resize((height, width)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        return transform

    # transform for training
    if aug:
        transform = T.Compose([
            T.Resize((height, width)),
            T.RandomHorizontalFlip(0.5),
            T.Pad(10),
            T.RandomCrop((height, width)),
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
    return transform


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

def build_dataloader(args, tranforms=None, combs_select=None, train_only=False, training=True):
    logger = logging.getLogger("CLIP2ReID.dataset")

    num_workers = args.num_workers
    dataset = __factory[args.dataset_name](root=args.root_dir, nlp_aug=args.nlp_aug)

    if training:
        pass
    else:
        if tranforms:
            test_transforms = tranforms
        else:
            test_transforms = build_transforms(img_size=args.img_size,
                                               is_train=False)
        ds = dataset.test

        new_ds = {}
        for k, v in ds.items():
            k_s = sorted(k.lower().split('_'))
            new_k_s = '_'.join(k_s)
            if new_k_s in new_ds:
                for k2, v2 in v.items():
                    new_ds[new_k_s][k2].extend(v2)
            else:
                new_ds[new_k_s] = v
        for name, res_dict in new_ds.items():
            if 'onemodal' in name:
                if 'text' in name.lower():
                    test_text_set = TextDataset(res_dict['query_idxs'], res_dict['text'], res_dict['query_idxs'], text_length=args.text_length)  ###### 77 -> 248
                elif 'sk' in name.lower():
                    test_simg_set = ImageDataset(res_dict['query_idxs'], res_dict['sk'], res_dict['query_idxs'],
                                   test_transforms)
                elif 'cp' in name.lower():
                    test_cimg_set = ImageDataset(res_dict['query_idxs'], res_dict['cp'], res_dict['query_idxs'],
                                   test_transforms)
                elif 'nir' in name.lower():
                    test_nimg_set = ImageDataset(res_dict['query_idxs'], res_dict['nir'], res_dict['query_idxs'],
                                    test_transforms)
            elif 'twomodal' in name:
                if 'text' in name.lower() and 'sk' in name.lower():
                    test_text_sk_set = SketchTextDataset(res_dict['sk'], res_dict['query_idxs'], res_dict['query_idxs'],
                                  res_dict['text'], test_transforms,
                                  text_length=args.text_length)
                elif 'text' in name.lower() and 'cp' in name.lower():
                    test_text_cp_set = SketchTextDataset(res_dict['cp'], res_dict['query_idxs'], res_dict['query_idxs'],
                                    res_dict['text'], test_transforms,
                                    text_length=args.text_length)
                elif 'text' in name.lower() and 'nir' in name.lower():
                    test_text_nir_set = SketchTextDataset(res_dict['nir'], res_dict['query_idxs'], res_dict['query_idxs'],
                                    res_dict['text'], test_transforms,
                                    text_length=args.text_length)
                elif 'sk' in name.lower() and 'cp' in name.lower():
                    test_sk_cp_set = ImageImageDataset(res_dict['query_idxs'], res_dict['sk'], res_dict['query_idxs'],
                                    res_dict['query_idxs'], res_dict['cp'], res_dict['query_idxs'],
                                   test_transforms)
                elif 'sk' in name.lower() and 'nir' in name.lower():
                    test_sk_nir_set = ImageImageDataset(res_dict['query_idxs'], res_dict['sk'], res_dict['query_idxs'],
                                        res_dict['query_idxs'], res_dict['nir'], res_dict['query_idxs'],
                                        test_transforms)
                elif 'cp' in name.lower() and 'nir' in name.lower():
                    test_cp_nir_set = ImageImageDataset(res_dict['query_idxs'], res_dict['cp'], res_dict['query_idxs'],
                                        res_dict['query_idxs'], res_dict['nir'], res_dict['query_idxs'],
                                        test_transforms)
                    
            elif 'threemodal' in name:
                if 'text' in name.lower() and 'sk' in name.lower() and 'cp' in name.lower():
                    test_text_cp_sk_set = ImageImageTextDataset(res_dict['cp'], res_dict['query_idxs'], 
                                res_dict['sk'], res_dict['query_idxs'],
                                res_dict['query_idxs'],
                                  res_dict['text'], test_transforms,
                                  text_length=args.text_length)
                elif 'text' in name.lower() and 'sk' in name.lower() and 'nir' in name.lower():
                    test_text_sk_nir_set = ImageImageTextDataset(res_dict['sk'], res_dict['query_idxs'],   ###### 
                                            res_dict['nir'], res_dict['query_idxs'],
                                            res_dict['query_idxs'],
                                            res_dict['text'], test_transforms,
                                            text_length=args.text_length)
                elif 'text' in name.lower() and 'cp' in name.lower() and 'nir' in name.lower():
                    test_text_cp_nir_set = ImageImageTextDataset(res_dict['cp'], res_dict['query_idxs'],   ####### 
                                            res_dict['nir'], res_dict['query_idxs'],
                                            res_dict['query_idxs'],
                                            res_dict['text'], test_transforms,
                                            text_length=args.text_length)
                elif 'sk' in name.lower() and 'cp' in name.lower() and 'nir' in name.lower():
                    test_cp_sk_nir_set = ImageImageImageDataset(res_dict['query_idxs'], res_dict['cp'], res_dict['query_idxs'],
                                            res_dict['query_idxs'], res_dict['sk'], res_dict['query_idxs'],
                                            res_dict['query_idxs'], res_dict['nir'], res_dict['query_idxs'],
                                            test_transforms)
            elif 'fourmodal' in name:
                test_text_cp_sk_nir_set = ImageImageImageTextDataset(res_dict['cp'], res_dict['query_idxs'], 
                                res_dict['sk'], res_dict['query_idxs'], 
                                res_dict['nir'], res_dict['query_idxs'],
                                res_dict['query_idxs'],
                                res_dict['text'], test_transforms,
                                text_length=args.text_length)
            elif 'gallery' in name:
                test_img_set = ImageDataset(res_dict['idxs'], res_dict['rgb'], res_dict['idxs'], test_transforms)
        # gallery
        test_img_loader = DataLoader(test_img_set,
                                    batch_size=args.test_batch_size,
                                    shuffle=False,
                                    num_workers=num_workers)
        # query
        test_text_loader = DataLoader(test_text_set, batch_size=args.test_batch_size, shuffle=False, num_workers=num_workers)
        test_sketch_loader = DataLoader(test_simg_set, batch_size=args.test_batch_size, shuffle=False, num_workers=num_workers)
        test_color_pencil_loader = DataLoader(test_cimg_set, batch_size=args.test_batch_size, shuffle=False, num_workers=num_workers)
        test_nir_loader = DataLoader(test_nimg_set, batch_size=args.test_batch_size,shuffle=False,num_workers=num_workers)
        
        test_text_sk_loader = DataLoader(test_text_sk_set,batch_size=args.test_batch_size,shuffle=False,num_workers=num_workers)
        test_text_cp_loader = DataLoader(test_text_cp_set, batch_size=args.test_batch_size,shuffle=False,num_workers=num_workers)
        test_text_nir_loader = DataLoader(test_text_nir_set,batch_size=args.test_batch_size,shuffle=False,num_workers=num_workers)
        test_sk_cp_loader = DataLoader(test_sk_cp_set,batch_size=args.test_batch_size,shuffle=False,num_workers=num_workers)
        test_sk_nir_loader = DataLoader(test_sk_nir_set,batch_size=args.test_batch_size,shuffle=False,num_workers=num_workers)
        test_cp_nir_loader = DataLoader(test_cp_nir_set,batch_size=args.test_batch_size,shuffle=False,num_workers=num_workers)
        
        test_text_cp_sk_loader = DataLoader(test_text_cp_sk_set,batch_size=args.test_batch_size,shuffle=False,num_workers=num_workers)
        test_text_cp_nir_loader = DataLoader(test_text_cp_nir_set,batch_size=args.test_batch_size,shuffle=False,num_workers=num_workers)
        test_text_sk_nir_loader = DataLoader(test_text_sk_nir_set,batch_size=args.test_batch_size,shuffle=False,num_workers=num_workers)
        test_cp_sk_nir_loader = DataLoader(test_cp_sk_nir_set,batch_size=args.test_batch_size,shuffle=False,num_workers=num_workers)
        
        test_text_cp_sk_nir_loader = DataLoader(test_text_cp_sk_nir_set,batch_size=args.test_batch_size,shuffle=False,num_workers=num_workers)
        
        return test_img_loader, test_text_loader, test_sketch_loader, test_color_pencil_loader, test_nir_loader,\
            test_text_sk_loader, test_text_cp_loader, test_text_nir_loader, test_sk_cp_loader, test_sk_nir_loader, test_cp_nir_loader,\
            test_text_cp_sk_loader, test_text_cp_nir_loader, test_text_sk_nir_loader, test_cp_sk_nir_loader,\
            test_text_cp_sk_nir_loader
