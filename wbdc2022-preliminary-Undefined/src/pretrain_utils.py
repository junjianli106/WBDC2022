import math
import random
import shutil
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple, Optional
from argparse import ArgumentParser
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from collections import defaultdict
import transformers
transformers.logging.set_verbosity_error()
from transformers import BertTokenizer, AdamW
from transformers.models.bert.modeling_bert import  BertOnlyMLMHead
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertEmbeddings, BertEncoder
from transformers import AutoTokenizer, AutoModel, AutoConfig, BertTokenizer,AutoModelForMaskedLM, BertConfig

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast as autocast, GradScaler

from models.nezha.modeling_nezha import NeZhaConfig, NeZhaForMaskedLM
from masklm import MaskLM, MaskVideo, ShuffleVideo

import gc
import zipfile
from io import BytesIO
warnings.filterwarnings('ignore')

class WarmupLinearSchedule(LambdaLR):
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))

def get_scheduler(args, optimizer, num_train_steps):
    if args.scheduler=='linear':
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.num_warmup_steps, num_training_steps=num_train_steps
        )
    elif args.scheduler=='cosine':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=args.num_warmup_steps, num_training_steps=num_train_steps, num_cycles=args.num_cycles
        )
    return scheduler


def build_optimizer(args, model, train_steps):
    no_decay = ['bias', 'LayerNorm.weight']

    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.eps)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=train_steps * args.warmup_ratio, t_total=train_steps)

    return optimizer, scheduler


def batch2cuda(args, batch):
    '''
    batch -> dict
    '''
    return {item: value.to(args.device) for item, value in list(batch.items())}


def create_dirs(path):
    os.makedirs(path, exist_ok=True)


def save_model(args, model, tokenizer, global_steps, is_last=False):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model_to_save = model.module if hasattr(model, 'module') else model
    if is_last:
        model_save_path = os.path.join(args.save_path, f'checkpoint-{global_steps}')
    else:
        model_save_path = os.path.join(args.record_save_path, f'checkpoint-{global_steps}')
    
    model_to_save.save_pretrained(model_save_path)
    tokenizer.save_vocabulary(model_save_path)

    print(f'\n>> model saved in : {model_save_path} .')
    

def load_dataset(args, i):
    with open(f'./process_data/pkl/pretrain_{i}.pkl', 'rb') as f:
        train_data = pickle.load(f)
    
    print(f'Define dataset：{i}')
    train_dataset = WBDCDataset(train_data)
    return train_dataset

def load_data(args, tokenizer):
    for i in tqdm(range(1, 21)):
        if i == 1:
            train_dataset = load_dataset(args, i)
        else:
            train_dataset = train_dataset + load_dataset(args, i)
    collate_fn = WBDCCollator(args)
    print('Define DataLoader')
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    return train_dataloader

class MultiModal_Pretrain_Dataset(Dataset):
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
                 ann,
                 zip_feats: str,
                 test_mode: bool = False):
        self.max_frame = args.max_frames
        self.bert_seq_length = args.bert_seq_length
        self.test_mode = test_mode
        # lazy initialization for zip_handler to avoid multiprocessing-reading error
        self.zip_feat_path = zip_feats
        self.handles = [None for _ in range(args.num_workers)]

        self.anns = ann
        
        # initialize the text tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_dir, use_fast=True, cache_dir=args.bert_cache)

    def __len__(self) -> int:
        return len(self.anns)

    def get_visual_feats(self, worker_id, idx: int) -> tuple:
        # read data from zipfile
        vid = self.anns[idx]['id']
        if self.handles[worker_id] is None:
            self.handles[worker_id] = zipfile.ZipFile(self.zip_feat_path, 'r')
        raw_feats = np.load(BytesIO(self.handles[worker_id].read(name=f'{vid}.npy')), allow_pickle=True)
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
        encoded_inputs = self.tokenizer(text, max_length=self.bert_seq_length, padding='max_length', truncation=True)
        input_ids = np.array(encoded_inputs['input_ids'])
        mask = np.array(encoded_inputs['attention_mask'])

        return input_ids, mask

    def __getitem__(self, idx: int) -> dict:
        # Step 1, load visual features from zipfile.
        worker_info = torch.utils.data.get_worker_info()
        frame_input, frame_mask = self.get_visual_feats(worker_info.id, idx)
            
        # Step 2, load title tokens
        title_replace_p = random.random()
        asr_replace_p = random.random()
        ocr_replace_p = random.random()
        
        title = self.anns[idx]['title']
        asr = self.anns[idx]['asr']
        ocr_dict = self.anns[idx]['ocr']
        
        ocr = ''
        for i, tmp in enumerate(ocr_dict):
            if i == 0:
                ocr = ocr + tmp['text']
            else:
                ocr = ocr + ',' + tmp['text']

        if  title == '':
                title = '无'
            
        if  asr == '':
                asr = '无'
                
        if  ocr == '':
                ocr = '无'
            
        text_asr = title + '[SEP]' + asr + '[SEP]' + ocr
        title_input, title_mask = self.tokenize_text(text_asr)

        # Step 3, summarize into a dictionary
        data = dict(
            frame_input=frame_input,
            frame_mask=frame_mask,
            text_input=title_input,
            text_mask=title_mask
        )
        
        return data
    
def create_dataloaders(args):
    
    with open(args.train_unlabeled, 'r', encoding='utf8') as f:
        anns = json.load(f)
    Data = np.array(anns)
    
    train_dataset = MultiModal_Pretrain_Dataset(args, Data, args.train_zip_nolabels_feats)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  sampler=train_sampler,
                                  drop_last=True,
                                  pin_memory=True,
                                  num_workers=args.num_workers,
                                  prefetch_factor=args.prefetch)
    
    return train_dataloader
    
