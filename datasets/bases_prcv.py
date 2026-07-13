from typing import List
from pyparsing import col
from torch.utils.data import Dataset
import os.path as osp
import logging
import torch
from utils.iotools import read_image
from utils.simple_tokenizer import SimpleTokenizer
from prettytable import PrettyTable
import random
import nltk
import regex as re
import copy


class BaseDataset(object):
    """
    Base class of text to image reid dataset
    """
    logger = logging.getLogger("ORBench.dataset")  ### 改为一样的名称才会显示logger

    def show_dataset_info(self):
        num_train_pids, num_train_imgs, num_train_captions = len(
            self.train_id_container), len(self.train_annos), len(self.train)
        num_test_pids, num_test_imgs, num_test_captions = len(
            self.test_id_container), len(self.test_annos), len(
                self.test['captions'])
        num_val_pids, num_val_imgs, num_val_captions = len(
            self.val_id_container), len(self.val_annos), len(
                self.val['captions'])

        # TODO use prettytable print comand line table

        self.logger.info(f"{self.__class__.__name__} Dataset statistics:")
        table = PrettyTable(['subset', 'ids', 'images', 'captions'])
        table.add_row(
            ['train', num_train_pids, num_train_imgs, num_train_captions])
        table.add_row(
            ['test', num_test_pids, num_test_imgs, num_test_captions])
        table.add_row(['val', num_val_pids, num_val_imgs, num_val_captions])
        self.logger.info('\n' + str(table))

        print('\n' + str(table))


def tokenize(caption: str, tokenizer, text_length=248, truncate=True) -> torch.LongTensor:
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

import os
class ImageTextDataset(Dataset):
    '''修改为支持输入五个模态'''
    def __init__(self,
                 dataset,
                 transform=None,
                 text_length: int = 248,
                 truncate: bool = True,
                 ## 增加一个可选项: 随机选择的组合
                 combs_select: list = None
                 ):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()
        
        self.combs_select = combs_select  #### [1,1,0,0]假如是这样, 那么只会选择sk, cp; nir, text不选

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pid, image_id, rgb_img_path, sk_img_path, cp_img_path, nir_img_path, caption = self.dataset[index]
        img = read_image(rgb_img_path)
        # simg_path = os.path.join('/data0/data_ccq/CUHK-PEDES/CUHK-PEDES/imgs-sketch2', img_path.split('/')[-2],img_path.split('/')[-1])
        img_sk = read_image(sk_img_path)
        img_cp = read_image(cp_img_path)
        img_nir = read_image(nir_img_path)

        if self.transform is not None:
            '''这里要考虑是否对其他图像也用transform,，目前全部都用吧: 不影响, 因为没有对颜色进行transform的内容'''
            img = self.transform(img)
            simg = self.transform(img_sk)
            cimg = self.transform(img_cp)
            nimg = self.transform(img_nir)
        tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

        ret = {
            'pids': pid,
            'image_ids': image_id,
            'images': img,
            'simages': simg,
            'cimages': cimg,
            'nimages': nimg,
            'caption_ids': tokens,
        }

        ### 根据所选的重新更新dict
        new_ret = {}
        combs_select_dict = {n: self.combs_select[idx] for idx, n in enumerate(['simages', 'cimages', 'nimages', 'cpation_ids'])}
        for k, v in ret.items():
            if k in combs_select_dict:
                if v == 0:
                    continue
            new_ret[k] = v

        return new_ret


class ImageImageImageDataset(Dataset):
    def __init__(self, image_pids, img_paths, image_ids, image_pids2, img_paths2, image_ids2,
                 image_pids3, img_paths3, image_ids3, transform=None):
        self.image_pids = image_pids
        self.img_paths = img_paths
        self.image_ids = image_ids

        self.image_pids2 = image_pids2
        self.img_paths2 = img_paths2
        self.image_ids2 = image_ids2

        self.image_pids3 = image_pids3
        self.img_paths3 = img_paths3
        self.image_ids3 = image_ids3

        self.transform = transform

    def __len__(self):
        return len(self.image_pids)

    def __getitem__(self, index):
        pid, img_path, image_id = self.image_pids[index], self.img_paths[index], self.image_ids[index]
        pid2, img_path2, image_id2 = self.image_pids2[index], self.img_paths2[index], self.image_ids2[index]
        pid3, img_path3, image_id3 = self.image_pids3[index], self.img_paths3[index], self.image_ids3[index]
        img = read_image(img_path)
        img2 = read_image(img_path2)
        img3 = read_image(img_path3)
        if self.transform is not None:
            img = self.transform(img)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        return pid, img, image_id,pid2, img2, image_id2,pid3, img3, image_id3
    

class ImageImageDataset(Dataset):
    def __init__(self, image_pids, img_paths, image_ids, image_pids2, img_paths2, image_ids2,transform=None):
        self.image_pids = image_pids
        self.img_paths = img_paths
        self.image_ids = image_ids

        self.image_pids2 = image_pids2
        self.img_paths2 = img_paths2
        self.image_ids2 = image_ids2

        self.transform = transform

    def __len__(self):
        return len(self.image_pids)

    def __getitem__(self, index):
        pid, img_path, image_id = self.image_pids[index], self.img_paths[index], self.image_ids[index]
        pid2, img_path2, image_id2 = self.image_pids2[index], self.img_paths2[index], self.image_ids2[index]
        img = read_image(img_path)
        img2 = read_image(img_path2)
        if self.transform is not None:
            img = self.transform(img)
            img2 = self.transform(img2)
        return pid, img, image_id,pid2, img2, image_id2
    

class ImageDataset(Dataset):
    def __init__(self, image_pids, img_paths, image_ids, transform=None):
        self.image_pids = image_pids
        self.img_paths = img_paths
        self.image_ids = image_ids
        self.transform = transform

    def __len__(self):
        return len(self.image_pids)

    def __getitem__(self, index):
        pid, img_path, image_id = self.image_pids[index], self.img_paths[index], self.image_ids[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return pid, img, image_id

class SketchDataset(Dataset):
    def __init__(self, simg_paths, simage_ids, simage_pids, transform=None):
        self.simg_paths = list(set(simg_paths))
        self.simg_paths.sort(key=simg_paths.index)
        self.simage_ids = list(set(simage_ids))
        self.simage_ids.sort(key=simage_ids.index)
        # self.simage_pids = list(set(simage_pids))
        # self.simage_pids.sort(key=simage_pids.index)
        self.simage_pids = simage_pids
        self.transform = transform

    def __len__(self):
        return len(self.simage_ids)

    def __getitem__(self, index):
        pid = self.simage_pids[index]
        simg_path = self.simg_paths[index]
        simage_id = self.simage_ids[index]
        simg = read_image(simg_path)
        if self.transform is not None:
            simg = self.transform(simg)

        return pid, simg, simage_id
        
class SketchTextDataset(Dataset):
    def __init__(self, simg_paths, simage_ids,
                 caption_pids,
                 captions,  transform=None,
                 text_length: int = 248, 
                 truncate: bool = True):
        self.simg_paths = simg_paths
        self.simage_ids = simage_ids
        self.caption_pids = caption_pids
        self.captions = captions
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()
        self.transform = transform

    def __len__(self):
        return len(self.caption_pids)

    def __getitem__(self, index):
        pid, caption, simg_path, simage_id = self.caption_pids[index], self.captions[index], self.simg_paths[index], self.simage_ids[index]

        simg = read_image(simg_path)
        if self.transform is not None:
            simg = self.transform(simg)
        caption = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

        return simg, simage_id, pid, caption


class ImageImageTextDataset(Dataset):
    def __init__(self, simg_paths, simage_ids,simg_paths2, simage_ids2,
                 caption_pids,
                 captions,  transform=None,
                 text_length: int = 248, 
                 truncate: bool = True):
        self.simg_paths = simg_paths
        self.simage_ids = simage_ids
        self.simg_paths2 = simg_paths2
        self.simage_ids2 = simage_ids2

        self.caption_pids = caption_pids
        self.captions = captions
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()
        self.transform = transform

    def __len__(self):
        return len(self.caption_pids)

    def __getitem__(self, index):
        pid, caption, simg_path, simage_id = self.caption_pids[index], self.captions[index],\
              self.simg_paths[index], self.simage_ids[index]
        simg_path2, simage_id2 = self.simg_paths2[index], self.simage_ids2[index]

        simg = read_image(simg_path)
        simg2 = read_image(simg_path2)

        if self.transform is not None:
            simg = self.transform(simg)
            simg2 = self.transform(simg2)

        caption = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

        return simg, simage_id, simg2, simage_id2,pid, caption

class ImageImageImageTextDataset(Dataset):
    def __init__(self, simg_paths, simage_ids,simg_paths2, simage_ids2,simg_paths3, simage_ids3,
                 caption_pids,
                 captions,  transform=None,
                 text_length: int = 248, 
                 truncate: bool = True):
        self.simg_paths = simg_paths
        self.simage_ids = simage_ids
        self.simg_paths2 = simg_paths2
        self.simage_ids2 = simage_ids2

        self.simg_paths3 = simg_paths3
        self.simage_ids3 = simage_ids3

        self.caption_pids = caption_pids
        self.captions = captions
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()
        self.transform = transform

    def __len__(self):
        return len(self.caption_pids)

    def __getitem__(self, index):
        pid, caption, simg_path, simage_id = self.caption_pids[index], self.captions[index],\
              self.simg_paths[index], self.simage_ids[index]
        simg_path2, simage_id2 = self.simg_paths2[index], self.simage_ids2[index]
        simg_path3, simage_id3 = self.simg_paths3[index], self.simage_ids3[index]

        simg = read_image(simg_path)
        simg2 = read_image(simg_path2)
        simg3 = read_image(simg_path3)

        if self.transform is not None:
            simg = self.transform(simg)
            simg2 = self.transform(simg2)
            simg3 = self.transform(simg3)

        caption = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

        return simg, simage_id, simg2, simage_id2,simg3, simage_id3,pid, caption


class TextDataset(Dataset):
    def __init__(self,
                 caption_pids,
                 captions,
                 caption_ids,  # 增加一个caption ids
                 text_length: int = 248,
                 truncate: bool = True):
        self.caption_pids = caption_pids
        self.captions = captions

        self.caption_ids = caption_ids


        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.caption_pids)

    def __getitem__(self, index):
        pid, caption, caption_id = self.caption_pids[index], self.captions[index], self.caption_ids[index]

        caption = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

        return pid, caption, caption_id


class ImageTextMaskColorDataset(Dataset):
    def __init__(self,
                 dataset,
                 transform=None,
                 text_length: int = 248,
                 masked_token_rate: float = 0.8,
                 masked_token_unchanged_rate: float = 0.1,
                 truncate: bool = True):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate
        self.masked_token_rate = masked_token_rate
        self.masked_token_unchanged_rate = masked_token_unchanged_rate

        self.tokenizer = SimpleTokenizer()


        self.colors_set = [
            'multicolored', 'gray', 'black', 'redish', 'blueish', 'brown', 'lime',
            'golden', 'nude', 'purplish', 'teal', 'magenta', 'whit', 'tan', 'grassy',
            'red', 'brownish', 'yellow', 'olive', 'metallic', 'orange', 'aqua', 'kaki',
            'bluish', 'purple', 'grayish', 'beige', 'pinkish', 'cream', 'auburn',
            'bleached', 'straw', 'orangish', 'brunette', 'white', 'yellowish', 'blue',
            'reddish', 'green', 'coral', 'greenish', 'blond', 'taupe', 'pink', 'grey',
            'colorful', 'neutral', 'transparent', 'gold', 'greyish', 'whitish',
            'orangey', 'salmon', 'dark', 'blonde', 'burgundy', 'khaki', 'silver',
            'navy'
        ]

        self.color2id = {color: idx+1 for idx, color in enumerate(self.colors_set)}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pid, image_id, img_path, caption = self.dataset[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        
        masked_color_caption_tokens, color_labels = self._build_masked_color_question_and_label(caption)

        caption_tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

        ret = {
            'pids': pid,
            'image_ids': image_id,
            'images': img,
            'caption_ids': caption_tokens,
            'masked_caption_ids': masked_color_caption_tokens,
            'mcm_labels': color_labels
        }

        return ret

    def _build_masked_color_question_and_label(self, caption):

        sot_token = self.tokenizer.encoder["<|startoftext|>"]
        eot_token = self.tokenizer.encoder["<|endoftext|>"]
        mask_token = self.tokenizer.encoder["<|mask|>"]


        words = re.findall(self.tokenizer.pat, caption.lower())

        # color_2_tokens = {color: self.tokenizer.encode(color) for color in self.colors_set}

        bpe_tokens = []
        color_label = []
        for i, word in enumerate(words):
            if word in self.colors_set:
                # color_tokens = color_2_tokens[word]
                # bpe_tokens.extend([mask_token] * len(color_tokens))
                # color_label.extend([self.color2id[word]] * len(color_tokens))
                if random.random() < self.masked_token_rate:
                    mask_id = mask_token
                else:
                    if random.random() < self.masked_token_unchanged_rate:
                        mask_id = self.get_bpe_tokens(word)
                    else:
                        mask_id = random.randint(0, len(self.tokenizer.encoder)-3)
                
                if isinstance(mask_id, list):
                    bpe_tokens.extend(mask_id)
                    color_label.extend([self.color2id[word]] *len(mask_id))
                else:
                    bpe_tokens.append(mask_id)
                    color_label.append(self.color2id[word])
            else:
                b_tokens = self.get_bpe_tokens(word)
                bpe_tokens.extend(b_tokens)
                color_label.extend([0]* len(b_tokens))
        
        masked_color_caption_tokens = [sot_token] + bpe_tokens + [eot_token]
        color_label = [0] + color_label + [0]

        result = torch.zeros(self.text_length, dtype=torch.long)
        label = torch.zeros(self.text_length, dtype=torch.long)
        if len(masked_color_caption_tokens) > self.text_length:
            if self.truncate:
                masked_color_caption_tokens = masked_color_caption_tokens[:self.text_length]
                masked_color_caption_tokens[-1] = eot_token

                color_label = color_label[:self.text_length]
                color_label[-1] = 0
            else:
                raise RuntimeError(
                    f"Input {caption} is too long for context length {self.text_length}"
                )
        result[:len(masked_color_caption_tokens)] = torch.tensor(masked_color_caption_tokens)
        label[:len(color_label)] = torch.tensor(color_label)
        return result, label

    def get_bpe_tokens(self, word):
        token = ''.join(self.tokenizer.byte_encoder[b] for b in word.encode('utf-8'))
        bpe_tokens = [self.tokenizer.encoder[bpe_token] for bpe_token in self.tokenizer.bpe(token).split(' ')]
        return bpe_tokens


class ImageTextMCQDataset(Dataset):
    def __init__(self,
                 dataset,
                 transform=None,
                 text_length: int = 248,
                 truncate: bool = True):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pid, image_id, img_path, caption = self.dataset[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)

        question_tokens, answer_tokens = self.build_question(caption)

        caption_tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

        ret = {
            'pids': pid,
            'image_ids': image_id,
            'images': img,
            'caption_ids': caption_tokens,
            'question_ids': question_tokens,
            'answer_ids': answer_tokens
        }

        return  ret

    def build_question(self, caption):

        sot_token = self.tokenizer.encoder["<|startoftext|>"]
        eot_token = self.tokenizer.encoder["<|endoftext|>"]
        mask_token = self.tokenizer.encoder["<|mask|>"]

        caption_tokens = [sot_token] + self.tokenizer.encode(caption) + [eot_token]

        question_tokens = torch.zeros(self.text_length, dtype=torch.long)

        if len(caption_tokens) > self.text_length:
            if self.truncate:
                caption_tokens = caption_tokens[:self.text_length]
                caption_tokens[-1] = eot_token
            else:
                raise RuntimeError(
                    f"Input {caption} is too long for context length {self.text_length}"
                )
        question_tokens[:len(caption_tokens)] = torch.tensor(caption_tokens)

        def get_mask_index(haystack, needle):
            n, m = len(haystack), len(needle)
            i = 0
            while i + m <= n:
                flag = True
                for j in range(m):
                    if haystack[i + j] != needle[j]:
                        flag = False
                        break
                if flag:
                    return i
                i += 1
            return -1

        noun_phrases = self._get_noun_phrases(caption)
        assert len(noun_phrases) > 0, f"Can't found noun phrases in the question: {caption}"

        ans_idx = -1
        while ans_idx == -1:
            answer = random.sample(noun_phrases, 1)[0]
            answer_tokens = self.tokenizer.encode(answer)
            # mask answer tokens in question_tokens
            # brute match, sometime will use KMP to improve speed
            ans_idx = get_mask_index(question_tokens, answer_tokens)

        # assert ans_idx != -1, f"The answer: {answer} is not in the question, maybe truncate from the question: {caption}"

        question_tokens[ans_idx:(ans_idx + len(answer_tokens))] = mask_token

        # answer_tokens = [sot_token] + answer_tokens + [eot_token]
        # add prompt before answer tokens
        # answer_tokens = [sot_token] + answer_tokens + [mask_token, mask_token, mask_token, eot_token]
        answer_tokens = [sot_token, mask_token, mask_token, mask_token] + answer_tokens + [eot_token]
        # pad answer tokens
        answer_tokens_tensor = torch.zeros(self.text_length, dtype=torch.long)
        answer_tokens_tensor[:len(answer_tokens)] = torch.tensor(answer_tokens)

        return question_tokens, answer_tokens_tensor

    def _get_noun_phrases(self, text):
        # grammar that matches noun phrases
        grammar = "NP: {(<JJ>)*(<V\w+>|<NN\w?>)*.*<NN\w?>}"
        # grammar="NP: {(<JJ>|<V\w+>|<NN\w?>)*<NN\w?>}"
        # grammar="NP: {(<JJ>|<V\w+>|<NN\w?>)*(<IN>|<DT>|<CC>)*<NN\w?>}"
        chunker = nltk.RegexpParser(grammar)
        # chunked = chunker.parse(nltk.pos_tag(nltk.word_tokenize(text)))
        chunked = chunker.parse(nltk.pos_tag(re.findall(self.tokenizer.pat, text.lower())))
        continuous_chunk = []
        current_chunk = []

        for subtree in chunked:
            if type(subtree) == nltk.Tree:
                current_chunk.append(" ".join(
                    [token for token, pos in subtree.leaves()]))
            elif current_chunk:
                named_entity = " ".join(current_chunk)
                if named_entity not in continuous_chunk:
                    continuous_chunk.append(named_entity)
                    current_chunk = []
            else:
                continue

        return continuous_chunk


class ImageTextMSMDataset(Dataset):
    """
    Masked Subsentence Matching
    """
    def __init__(self,
                 dataset,
                 transform=None,
                 text_length: int = 248,
                 truncate: bool = True):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pid, image_id, img_path, caption = self.dataset[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)

        question_tokens, answer_tokens = self.build_question(caption)

        caption_tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

        ret = {
            'pids': pid,
            'image_ids': image_id,
            'images': img,
            'caption_ids': caption_tokens,
            'question_ids': question_tokens,
            'answer_ids': answer_tokens
        }

        return  ret

    def build_question(self, caption):

        sot_token = self.tokenizer.encoder["<|startoftext|>"]
        eot_token = self.tokenizer.encoder["<|endoftext|>"]
        mask_token = self.tokenizer.encoder["<|mask|>"]

        caption_tokens = self.tokenizer.encode(caption)

        # random generate masked subsequence
        cap_len = len(caption_tokens)
        cap_max_len = min(cap_len, self.text_length)
        sub_min_len = cap_max_len * 0.8
        sub_max_len = cap_max_len * 0.95
        
        while True:
            sub_start = random.randint(0, min(cap_len // 2, self.text_length // 2))
            sub_end = random.randint(cap_len // 2 + 1, min(cap_len, self.text_length-1))
            sub_len = sub_end - sub_start
            if sub_min_len < sub_len < sub_max_len:
                break


        # build answer tensor
        answer_tokens = copy.deepcopy(caption_tokens[sub_start:sub_end])
        # no prompt
        answer_tokens = [sot_token] + answer_tokens + [eot_token]
        # with prompt
        # answer_tokens = [sot_token, mask_token, mask_token, mask_token] + answer_tokens + [eot_token]
        answer_tensor = torch.zeros(self.text_length, dtype=torch.long)
        answer_tensor[:len(answer_tokens)] = torch.tensor(answer_tokens)

        # build masked question tensor
        question_tokens = copy.deepcopy(caption_tokens)
        question_tokens = [sot_token] + question_tokens + [eot_token]

        question_tensor = torch.zeros(self.text_length, dtype=torch.long)

        if len(question_tokens) > self.text_length:
            if self.truncate:
                question_tokens = question_tokens[:self.text_length]
                question_tokens[-1] = eot_token
            else:
                raise RuntimeError(
                    f"Input {caption} is too long for context length {self.text_length}"
                )
        
        question_tokens = torch.tensor(question_tokens)
        question_tokens[sub_start+1:sub_end] = mask_token
        question_tensor[:len(question_tokens)] = question_tokens

        return question_tensor, answer_tensor



class ImageTextMLMDataset(Dataset):
    '''修改为支持输入五个模态'''

    def __init__(self,
                 dataset,
                 transform=None,
                 text_length: int = 248,
                 truncate: bool = True,  ## 是否切断长度超过 77 的text,  77 -> 248
                 ## 增加一个可选项: 随机选择的组合
                 combs_select: list = None
                 ):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate

        self.tokenizer = SimpleTokenizer()
        self.combs_select = combs_select  #### [1,1,0,0]假如是这样, 那么只会选择sk, cp; nir, text不选

        # 统计token长度
        # res = {}
        # for d in self.dataset:
        #     length = len(self.tokenizer.encode(d[-1]))
        #     if length not in res:
        #         res[length] = 1
        #     else:
        #         res[length] += 1
        # # 画图
        # import matplotlib.pyplot as plt
        # plt.bar(res.keys(), res.values())
        # # plt.show()
        # plt.savefig('/home/dj_2025/competitions/original_code/token_length.png')
        


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # pid, image_id, img_path, caption = self.dataset[index]
        pid, image_id, rgb_img_path, sk_img_path, cp_img_path, nir_img_path, caption = self.dataset[index]
        # img = read_image(img_path)

        img = read_image(rgb_img_path)
        # simg_path = os.path.join('/data0/data_ccq/CUHK-PEDES/CUHK-PEDES/imgs-sketch2', img_path.split('/')[-2],img_path.split('/')[-1])
        img_sk = read_image(sk_img_path)
        img_cp = read_image(cp_img_path)
        img_nir = read_image(nir_img_path)
        
        if self.transform is not None:
            # img = self.transform(img)
            img = self.transform(img)
            simg = self.transform(img_sk)
            cimg = self.transform(img_cp)
            nimg = self.transform(img_nir)
        
        caption_tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)
        # caption_masked_tokens = tokenize(caption_add_masked, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)
        mlm_tokens, mlm_labels  = self._build_random_masked_tokens_and_labels(caption_tokens.cpu().numpy())
        # mlm_tokens, mlm_labels = self._build_random_masked_tokens_and_labels_except_color_token(caption_tokens.cpu().numpy(), caption_masked_tokens.cpu().numpy())
        # 只 mask color 不行, 必须color + 剩余的也mask一些, 即, rgb和cp的text必须mask掉color, 而sk和cp恰恰相反
        # mlm_tokens_mask, mlm_labels_mask = self._build_rgb_mask_with_res_mask_tokens_and_labels(caption_tokens.cpu().numpy(), caption_masked_tokens.cpu().numpy())  # _build_rgb_mask_with_res_mask_tokens_and_labels
        # caption_tokens1 = self.txt_data_aug(caption_tokens.cpu().numpy())

        ret = {
            'pids': pid,
            'image_ids': image_id,
            'images': img,
            'simages': simg,
            'cimages': cimg,
            'nimages': nimg,
            'caption_ids': caption_tokens,  ##### 替换为mask之后的text token, 与RDE保持一致
            
            'mlm_ids': mlm_tokens,  # 除了color 没有被 mask, 其余的token随机mask
            'mlm_labels': mlm_labels,

            # 'mlm_ids_mask': mlm_tokens_mask,  ### 只 mask 掉 color 
            # 'mlm_labels_mask': mlm_labels_mask,
        }

        ### 根据所选的重新更新dict
        new_ret = {}
        combs_select_dict = {n: self.combs_select[idx] for idx, n in enumerate(['simages', 'cimages', 'nimages', 'caption_ids'])}
        for k, v in ret.items():
            if k in combs_select_dict:
                if combs_select_dict[k] == 0:
                    continue
            new_ret[k] = v

        return new_ret
    
    def txt_data_aug(self, tokens):
        import numpy as np
        mask = self.tokenizer.encoder["<|mask|>"]
        token_range = list(range(1, len(self.tokenizer.encoder)-3)) # 1 ~ 49405
        new_tokens = np.zeros_like(tokens)
        aug_tokens = []
        for i, token in enumerate(tokens):
            if 0 < token < 49405:
                prob = random.random()
                # mask token with 15% probability
                if prob < 0.20:
                    prob /= 0.20
                    # 50% randomly change token to mask token
                    if prob < 0.6:
                        aug_tokens.append(mask) 
                    # 20% randomly change token to random token
                    elif prob < 0.8:
                        aug_tokens.append(random.choice(token_range)) # -> rest 10% randomly keep current token
                    else:
                        None # # 30% randomly remove
                else:
                    # no masking token (will be ignored by loss function later)
                    aug_tokens.append(tokens[i])
            else:
                aug_tokens.append(tokens[i])
        new_tokens[0:len(aug_tokens)] = np.array(aug_tokens)
        return torch.tensor(new_tokens)
    

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
    


    def _build_random_masked_tokens_and_labels_except_color_token(self, tokens, tokens_masked):
        """
        改为: color 保持不变(不 mask), 其他的random mask
        """
        mask = self.tokenizer.encoder["<|mask|>"]
        token_range = list(range(1, len(self.tokenizer.encoder)-3)) # 1 ~ 49405
        
        labels = []
        for i, token in enumerate(tokens):

            if tokens_masked[i] == mask:  ## 如果已经标记mask的颜色已经被mask了, 忽略这部分, 不参与random mask, 其余的参与随机mask
                labels.append(0)
                continue

            # 仅对普通token做随机mask（排除特殊token：sot/eot/mask等）
            if 0 < token < (len(self.tokenizer.encoder) - 3):
                """除了color保持, 剩余的随机mask"""
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
            # 至少mask 1：选择第一个有效的非mask普通token
            for i, token in enumerate(tokens):
                if (token in token_range) and (token != mask):
                    labels[i] = token
                    tokens[i] = mask
                    break

        return torch.tensor(tokens), torch.tensor(labels)
    

    def _build_rgb_masked_tokens_and_labels(self, tokens, tokens_masked):
        """
        input rgb masked captions; tokens: 原始不mask的 token, tokens_masked: mask掉color相关的tokens
        fix bug: 如果已经存在mask, label应该是对应的token, 而不是0, 其余的label才都是 0啊!!!!
        """
        mask = self.tokenizer.encoder["<|mask|>"]
        # token_range = list(range(1, len(self.tokenizer.encoder)-3)) # 1 ~ 49405
        
        labels = []
        for i, token in enumerate(tokens):
            # 跳过已是 <|mask|> 的位置, label要去预测这个token !;
            if tokens_masked[i] == mask:
                labels.append(token)
                continue

            # 仅对普通token做随机mask（排除特殊token：sot/eot/mask等）
            # if 0 < token < (len(self.tokenizer.encoder) - 3):
            #     """剩余的不mask"""
            #     labels.append(0)
            # else:
            labels.append(0)  ### 剩余的都不预测
        return torch.tensor(tokens_masked), torch.tensor(labels)   ##### 注意, 返回的是 mask color 掉之后的token 啊!!!!
    

    def _build_rgb_mask_with_res_mask_tokens_and_labels(self, tokens, tokens_masked):
        """
        rgb mask + rgb剩余token也要 mask
        """
        mask = self.tokenizer.encoder["<|mask|>"]
        token_range = list(range(1, len(self.tokenizer.encoder)-3)) # 1 ~ 49405
        
        labels = []
        for i, token in enumerate(tokens):

            if tokens_masked[i] == mask:  ## 如果已经标记mask的颜色已经被mask了, 忽略这部分, 不参与random mask, 其余的参与随机mask
                tokens[i] = mask   # rgb mask, 剩余的再mask
                labels.append(token)
                continue

            # 仅对普通token做随机mask（排除特殊token：sot/eot/mask等）
            if 0 < token < (len(self.tokenizer.encoder) - 3):
                """除了color保持, 剩余的随机mask"""
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
            # 至少mask 1：选择第一个有效的非mask普通token
            for i, token in enumerate(tokens):
                if (token in token_range) and (token != mask):
                    labels[i] = token
                    tokens[i] = mask
                    break
        return torch.tensor(tokens), torch.tensor(labels)


class ImageTextMCQMLMDataset(ImageTextMLMDataset, ImageTextMCQDataset):
    def __init__(self,
                 dataset,
                 transform=None,
                 text_length: int = 248,
                 truncate: bool = True):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate

        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pid, image_id, img_path, caption = self.dataset[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        
        question_tokens, answer_tokens = self.build_question(caption)
        
        caption_tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

        mlm_tokens, mlm_labels = self._build_random_masked_tokens_and_labels(caption_tokens.cpu().numpy())  

        ret = {
            'pids': pid,
            'image_ids': image_id,
            'images': img,
            'caption_ids': caption_tokens,
            'mlm_ids': mlm_tokens,
            'mlm_labels': mlm_labels,
            'question_ids': question_tokens,
            'answer_ids': answer_tokens
        }

        return ret


class ImageTextMSMMLMDataset(ImageTextMLMDataset, ImageTextMSMDataset):
    def __init__(self,
                 dataset,
                 transform=None,
                 text_length: int = 248,
                 truncate: bool = True):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate

        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pid, image_id, img_path, caption = self.dataset[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        
        question_tokens, answer_tokens = self.build_question(caption)
        
        caption_tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

        mlm_tokens, mlm_labels = self._build_random_masked_tokens_and_labels(caption_tokens.cpu().numpy())  

        ret = {
            'pids': pid,
            'image_ids': image_id,
            'images': img,
            'caption_ids': caption_tokens,
            'mlm_ids': mlm_tokens,
            'mlm_labels': mlm_labels,
            'question_ids': question_tokens,
            'answer_ids': answer_tokens
        }

        return ret