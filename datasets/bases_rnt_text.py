# import sys
# sys.path.append('/home/dj_2025/competitions/original_code备份_last_H20')
from PIL import Image, ImageFile
import logging
from torch.utils.data import Dataset
import os.path as osp
from utils.simple_tokenizer import SimpleTokenizer
import torch
from prettytable import PrettyTable
import random
import copy

ImageFile.LOAD_TRUNCATED_IMAGES = True



def read_image(img_list):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    if type(img_list) == type("This is a str"):
        img_path = img_list
        got_img = False
        if not osp.exists(img_path):
            raise IOError("{} does not exist".format(img_path))
        while not got_img:
            try:
                img = Image.open(img_path).convert('RGB')
                RGB = img.crop((0, 0, 256, 128))
                NI = img.crop((256, 0, 512, 128))
                TI = img.crop((512, 0, 768, 128))
                img3 = [RGB, NI, TI]
                got_img = True
            except IOError:
                print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
                pass
    else:
        img3 = []
        for i in img_list:
            img_path = i
            got_img = False
            if not osp.exists(img_path):
                raise IOError("{} does not exist".format(img_path))
            while not got_img:
                try:
                    img = Image.open(img_path).convert('RGB')
                    img3.append(img)
                    got_img = True
                except IOError:
                    print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
                    pass
    return img3


class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def get_imagedata_info(self, data):
        pids, cams, tracks = [], [], []

        for _, pid, camid, trackid, _, _, _ in data:
            pids += [pid]
            cams += [camid]
            tracks += [trackid]
        pids = set(pids)
        cams = set(cams)
        tracks = set(tracks)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        num_views = len(tracks)
        return num_pids, num_imgs, num_cams, num_views

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """
    logger = logging.getLogger("ORBench.dataset")

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams, num_train_views = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams, num_train_views = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams, num_train_views = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        print("  ----------------------------------------")

        self.logger.info(f"{self.__class__.__name__} Dataset statistics:")
        table = PrettyTable(['subset', 'ids', 'images', 'cameras'])
        table.add_row(
            ['train', num_train_pids, num_train_imgs, num_train_cams])
        table.add_row(
            ['query', num_query_pids, num_query_imgs, num_query_cams])
        table.add_row(
            ['gallery', num_gallery_pids, num_gallery_imgs, num_gallery_cams])
        self.logger.info('\n' + str(table))



def tokenize(caption: str, tokenizer, text_length=77, truncate=True) -> torch.LongTensor:
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    tokens = [sot_token] + tokenizer.encode(caption) + [eot_token]

    result = torch.zeros(text_length, dtype=torch.long)
    if len(tokens) > text_length:
        if truncate:
            tokens = tokens[:text_length]
            tokens[-1] = eot_token
        else:
            raise RuntimeError(
                f"Input {caption} is too long for context length {text_length}"
            )
    result[:len(tokens)] = torch.tensor(tokens)
    return result


class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None, transform_nir=None, text_length: int = 77,
                 truncate: bool = True, mask_token=False, #mask_ratio: float = 0.
                 ):
        self.dataset = dataset
        self.transform = transform
        self.transform_nir = transform_nir

        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()
        # self.mask_ratio = mask_ratio
        self.mask_token = mask_token

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, trackid, r_text, n_text, t_text = self.dataset[index]
        img3 = read_image(img_path)
        r_tokens = tokenize(r_text, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)
        n_tokens = tokenize(n_text, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)
        t_tokens = tokenize(t_text, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

        if self.mask_token:
            r_tokens, _ = self._build_random_masked_tokens_and_labels(copy.deepcopy(r_tokens.cpu().numpy()))
            n_tokens, _ = self._build_random_masked_tokens_and_labels(copy.deepcopy(n_tokens.cpu().numpy()))
            t_tokens, _ = self._build_random_masked_tokens_and_labels(copy.deepcopy(t_tokens.cpu().numpy()))

        img = []
        if self.transform is not None:
            # img = [self.transform(img) for img in img3]
            img.append(self.transform(img3[0]))
            img.append(self.transform_nir(img3[1]))
            img.append(self.transform(img3[2]))

        if type(img_path) == type("This is a str"):
            return img, pid, camid, trackid, img_path.split('/')[-1], r_tokens, n_tokens, t_tokens
        else:
            return img, pid, camid, trackid, img_path[0].split('/')[-1], r_tokens, n_tokens, t_tokens

    def _build_random_masked_tokens_and_labels(self, tokens):
        """
        Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
        :param tokens: list of int, tokenized sentence.
        :return: (list of int, list of int), masked tokens and related labels for MLM prediction
        """
        mask = self.tokenizer.encoder["<|mask|>"]
        token_range = list(range(1, len(self.tokenizer.encoder)-3)) # 1 ~ 49405
        
        labels = []
        for i, token in enumerate(tokens):
            if 0 < token < 49405:
                prob = random.random()
                # mask token with 15% probability
                if prob < 0.20:  ### 0.15
                    prob /= 0.20  ### 0.15

                    # 80% randomly change token to mask token
                    if prob < 0.6: ### 0.8
                        tokens[i] = mask

                    # 10% randomly change token to random token
                    elif prob < 0.8:  ## 0.9
                        tokens[i] = random.choice(token_range)

                    # -> rest 10% randomly keep current token

                    # append current token to output (we will predict these later)
                    labels.append(token)
                else:
                    # no masking token (will be ignored by loss function later)
                    labels.append(0)
            else:
                labels.append(0)
        
        if all(l == 0 for l in labels):
            # at least mask 1
            labels[1] = tokens[1]
            tokens[1] = mask

        return torch.tensor(tokens), torch.tensor(labels)
