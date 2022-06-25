import json
import random
import zipfile
from io import BytesIO
from functools import partial
from async_timeout import final

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import BertTokenizer

from category_id_map import category_id_to_lv2id


def create_dataloaders(args):
    dataset = MultiModalDataset(
        args, args.train_annotation, args.train_zip_feats)
    size = len(dataset)
    val_size = int(size * args.val_ratio)
    if args.only_train:
        train_dataset = dataset
        # if test_mode is not None:
        #     train_dataset.test_mode = test_mode
    else:
        if args.val_annotation:
            print(f'use val: {args.val_annotation}')
            train_dataset = MultiModalDataset(args, args.train_annotation, args.train_zip_feats)
            val_dataset =  MultiModalDataset(args, args.val_annotation, args.train_zip_feats)
        
        else:
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [size - val_size, val_size],
                                                                    generator=torch.Generator().manual_seed(args.split_seed))

        # if test_mode is not None:
        #     train_dataset.test_mode = test_mode
        #     val_dataset.test_mode = test_mode

        
    # train_dataset = torch.utils.data.Subset(dataset, range(size-val_size))
    # val_dataset = torch.utils.data.Subset(dataset, range(size-val_size, size))
    if args.num_workers > 0:
        dataloader_class = partial(
            DataLoader, pin_memory=True, num_workers=args.num_workers, prefetch_factor=args.prefetch)
    else:
        # single-thread reading does not support prefetch_factor arg
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=0)

    if args.only_train:
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = dataloader_class(train_dataset,
                                            batch_size=args.batch_size,
                                            sampler=train_sampler,
                                            drop_last=True)
        
        return train_dataloader
    
    else:
        train_sampler = RandomSampler(train_dataset)
        val_sampler = SequentialSampler(val_dataset)
        # print(f'test mode: ', train_dataset.test_mode, val_dataset.test_mode)
        train_dataloader = dataloader_class(train_dataset,
                                            batch_size=args.batch_size,
                                            sampler=train_sampler,
                                            drop_last=True)
        val_dataloader = dataloader_class(val_dataset,
                                        batch_size=args.val_batch_size,
                                        sampler=val_sampler,
                                        drop_last=False)
        return train_dataloader, val_dataloader


class MultiModalDataset(Dataset):
    """ A simple class that supports multi-modal inputs.
    For the visual features, this dataset class will read the pre-extracted
    features from the .npy files. For the title information, it
    uses the BERT tokenizer to tokenize. We simply ignore the ASR & OCR text in this implementation.
    Args:
        ann_path (str): annotation file path, with the '.json' suffix.
        zip_feats (str): visual feature zip file path.
        test_mode (bool): if it's for testing.
    """

    def __init__(self,
                 args,
                 ann_path: str,
                 zip_feats: str,
                 test_mode: bool = False):
        self.max_frame = args.max_frames
        self.bert_seq_length = args.bert_seq_length
        print(f'bert seq length:{self.bert_seq_length}')
        
        self.test_mode = test_mode
        if args.pretrain:
            self.test_mode = True

        self.zip_feat_path = zip_feats
        self.num_workers = args.num_workers
        if self.num_workers > 0:
            # lazy initialization for zip_handler to avoid multiprocessing-reading error
            self.handles = [None for _ in range(args.num_workers)]
        else:
            self.handles = zipfile.ZipFile(self.zip_feat_path, 'r')
        # load annotations
        with open(ann_path, 'r', encoding='utf8') as f:
            self.anns = json.load(f)
        # initialize the text tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(
            args.bert_dir, use_fast=True, cache_dir=args.bert_cache)

    def __len__(self) -> int:
        return len(self.anns)

    def get_visual_feats(self, idx: int) -> tuple:
        # read data from zipfile
        vid = self.anns[idx]['id']
        if self.num_workers > 0:
            worker_id = torch.utils.data.get_worker_info().id
            if self.handles[worker_id] is None:
                self.handles[worker_id] = zipfile.ZipFile(
                    self.zip_feat_path, 'r')
            handle = self.handles[worker_id]
        else:
            handle = self.handles
        raw_feats = np.load(
            BytesIO(handle.read(name=f'{vid}.npy')), allow_pickle=True)
        raw_feats = raw_feats.astype(np.float32)  # float16 to float32
        num_frames, feat_dim = raw_feats.shape

        feat = np.zeros((self.max_frame, feat_dim), dtype=np.float32)
        mask = np.ones((self.max_frame,), dtype=np.int32)
        if num_frames <= self.max_frame:
            feat[:num_frames] = raw_feats
            mask[num_frames:] = 0
        else:
            # if the number of frames exceeds the limitation, we need to sample
            # the frames.
            if self.test_mode:
                # uniformly sample when test mode is True
                step = num_frames // self.max_frame
                select_inds = list(range(0, num_frames, step))
                select_inds = select_inds[:self.max_frame]
            else:
                # randomly sample when test mode is False
                select_inds = list(range(num_frames))
                random.shuffle(select_inds)
                select_inds = select_inds[:self.max_frame]
                select_inds = sorted(select_inds)
            for i, j in enumerate(select_inds):
                feat[i] = raw_feats[j]
        feat = torch.FloatTensor(feat)
        mask = torch.LongTensor(mask)
        return feat, mask

    def tokenize_text(self, text: str) -> tuple:
        encoded_inputs = self.tokenizer(
            text, max_length=self.bert_seq_length, padding='max_length', truncation=True)
        input_ids = torch.LongTensor(encoded_inputs['input_ids'])
        # print(len(input_ids))
        mask = torch.LongTensor(encoded_inputs['attention_mask'])
        return input_ids, mask

    def __getitem__(self, idx: int) -> dict:
        # Step 1, load visual features from zipfile.
        frame_input, frame_mask = self.get_visual_feats(idx)
        ocr_texts = []
        for item in self.anns[idx]['ocr']:
            ocr_texts.append(item['text'])

        # Step 2, load title tokens

        # 单独截断+拼接
        # title_input, title_mask = self.tokenize_text(self.anns[idx]['title'])
        # asr_input, asr_mask = self.tokenize_text(self.anns[idx]['asr'])
        # ocr_texts = []
        # for item in self.anns[idx]['ocr']:
        #     ocr_texts.append(item['text'])

        # ocr_text = '｜'.join(ocr_texts)
        # ocr_input, ocr_mask = self.tokenize_text(ocr_text)

        # sep_tensor = torch.LongTensor([self.tokenizer.sep_token_id])
        # mask1 = torch.LongTensor([1])
        # final_input = torch.cat([title_input, sep_tensor, asr_input, sep_tensor, ocr_input], dim=-1)
        # final_mask = torch.cat([title_mask, mask1, asr_mask, mask1, ocr_mask], dim=-1)

        # 拼接起来再截断

        text = self.anns[idx]['title'] + self.tokenizer.sep_token + \
            self.anns[idx]['asr'] + \
            self.tokenizer.sep_token + '｜'.join(ocr_texts)
        # text = self.anns[idx]['title'] + self.tokenizer.sep_token + \
        #     self.anns[idx]['asr'] + \
        #     self.tokenizer.sep_token + ''.join(ocr_texts)
        final_input, final_mask = self.tokenize_text(text)

        # 更精细的截断
        # title_ids = self.tokenizer.encode(self.anns[idx]['title'], add_special_tokens=False)
        # asr_ids = self.tokenizer.encode(self.anns[idx]['asr'], add_special_tokens=False)
        # ocr_ids = self.tokenizer.encode('｜'.join(ocr_texts), add_special_tokens=False)
        # # print('titile,asr,ocr length:', len(title_ids), len(asr_ids), len(ocr_ids))

        # max_length = self.bert_seq_length
        # pad_id = self.tokenizer.pad_token_id
        # sep_id = self.tokenizer.sep_token_id
        # cls_id = self.tokenizer.cls_token_id

        # if len(title_ids) + len(asr_ids) + len(ocr_ids) <= max_length - 4:
        #     l = len(title_ids) + len(asr_ids) + len(ocr_ids)
        #     final_input = torch.LongTensor([cls_id] + title_ids + [sep_id] + asr_ids + [sep_id] + ocr_ids + [sep_id] + [pad_id] * (max_length - l - 4))
        #     final_mask = torch.LongTensor([1] * (l + 4) + [0] * (max_length - l - 4))

        # else:
        #     exceed_length = len(title_ids) + len(asr_ids) + len(ocr_ids) - (max_length - 4)

        #     title_chunk_len, asr_chunk_len, ocr_chunk_len = 0, 0, 0
        #     ratios = np.array([len(title_ids) if len(title_ids) >= (max_length - 4) / 3 else 0, len(asr_ids) if len(asr_ids) >= (max_length - 4) / 3 else 0,
        #               len(ocr_ids) if len(ocr_ids) >= (max_length - 4) / 3 else 0])

        #     ratios = ratios / ratios.sum() * (exceed_length + 3)
        #     ratios = ratios.astype(int)

        #     title_chunk_len, asr_chunk_len, ocr_chunk_len = ratios
        #     # print(f'title,asr,ocr chunk length:', title_chunk_len, asr_chunk_len, ocr_chunk_len)
        #     title_ids = title_ids[:len(title_ids) - title_chunk_len]
        #     asr_ids = asr_ids[:len(asr_ids) - asr_chunk_len]
        #     ocr_ids = ocr_ids[:len(ocr_ids) - ocr_chunk_len]

        #     l = len(title_ids) + len(asr_ids) + len(ocr_ids)
        #     final_input = torch.LongTensor([cls_id] + title_ids + [sep_id] + asr_ids + [sep_id] + ocr_ids + [sep_id] + [pad_id] * (max_length - l - 4))
        #     final_mask = torch.LongTensor([1] * (l + 4) + [0] * (max_length - l - 4))

        # Step 3, summarize into a dictionary
        data = dict(
            frame_input=frame_input,
            frame_mask=frame_mask,
            title_input=final_input,
            title_mask=final_mask
        )

        # Step 4, load label if not test mode
        if not self.test_mode and 'category_id' in self.anns[idx]:
            # print('get label', self.test_mode)
            label = category_id_to_lv2id(self.anns[idx]['category_id'])
            data['label'] = torch.LongTensor([label])

        return data
