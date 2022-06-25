# coding:utf-8

import gc
import re
import os
import sys
import json
import pickle
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
from transformers import BertTokenizer, AdamW
from transformers.models.bert.modeling_bert import BertConfig, BertOnlyMLMHead
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertEmbeddings, BertEncoder
from transformers import AutoTokenizer, AutoModel, AutoConfig, BertTokenizer,AutoModelForMaskedLM, VisualBertForPreTraining, BertConfig
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from models.bert.visual_modeling_bert import BertModel as Visual_BertModel

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast as autocast, GradScaler

from models.nezha.modeling_nezha import NeZhaConfig, NeZhaForMaskedLM
from masklm import MaskLM, MaskVideo, ShuffleVideo
import transformers
transformers.logging.set_verbosity_error()
import gc
import zipfile
from io import BytesIO
#sys.path.append('src')
warnings.filterwarnings('ignore')

class VisualOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = VisualLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores

def gelu(x):
    """Implementation of the gelu activation function.
    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}

class VisualPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states
    
class VisualLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = VisualPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, 768, bias=False)
        self.bias = nn.Parameter(torch.zeros(768))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

    
class VisualBertForMaskedLM(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        #self.bert_config = BertConfig.from_pretrained(config)
        self.bert = Visual_BertModel.from_pretrained('../challenge/prev_pretrained_models/chinese-roberta-wwm-ext')
        #self.bert = Visual_BertModel(config)#.from_pretrained(config)
        self.cls = BertOnlyMLMHead(config)
        
    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    def forward(self, text_input_ids, 
                    attention_mask, 
                    token_type_ids, 
                    visual_embedding, 
                    visual_attention_mask, 
                    visual_token_type_ids, 
                    gather_index=None, 
                    return_mlm=False):
    
        encoder_outputs = self.bert(input_ids=text_input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            visual_embeds=visual_embedding,
                            visual_attention_mask=visual_attention_mask,
                            visual_token_type_ids=visual_token_type_ids,
                            output_hidden_states=True).last_hidden_state  # bs, seq_len + frame_len, 768
        
        if return_mlm:
            return encoder_outputs, self.cls(encoder_outputs)[:, 1:-visual_embedding.size()[1] , :]
        else:
            return encoder_outputs, None    
        
class VisualBertMultiModal(nn.Module):
    def __init__(self, args, task=['mlm'], init_from_pretrain=True):
        super().__init__()
        self.bert_config = BertConfig.from_pretrained(args.bert_dir)
        self.bert = VisualBertForMaskedLM(self.bert_config)
        self.task = set(task)

        if 'large' in args.bert_dir:
            args.bert_output_size = 1024
            
        if 'mlm' in task:
            self.lm = MaskLM(tokenizer_path=args.bert_dir)
            self.num_class = 200
            self.vocab_size = self.bert_config.vocab_size
            
        if 'mfm' in task:
            self.vm = MaskVideo()
            self.roberta_mvm_lm_header = VisualOnlyMLMHead(self.bert_config) 

        if 'itm' in task:
            self.sv = ShuffleVideo()
            self.newfc_itm = torch.nn.Linear(self.bert_config.hidden_size, 1) 
            
        self.newfc_hidden = torch.nn.Linear(args.bert_output_size, args.bert_output_size)
        
        self.classifier = nn.Linear(args.bert_output_size, 200)
        #self.ClassificationHead = ClassificationHead(args)

#         self.project_linear = nn.Sequential(
#                                 nn.Linear(args.frame_embedding_size, args.bert_output_size),
#                                 nn.ReLU(),
#                                 nn.LayerNorm(args.bert_output_size)
#                                 )
        
    def forward(self, text_input_ids, 
                        attention_mask, 
                        token_type_ids, 
                        visual_embedding, 
                        visual_attention_mask, 
                        visual_token_type_ids, 
                        inference=False, task=None):
    
        loss, pred = 0, None
        if task is None:
            sample_task = self.task
        elif type(task) == str:
            sample_task = [task]
        elif type(task) == list:
            sample_task = task
            
        
        return_mlm = False
        if 'mlm' in sample_task:
            input_ids, lm_label = self.lm.torch_mask_tokens(text_input_ids.cpu())
            text_input_ids = input_ids.to(text_input_ids.device)
            lm_label = lm_label[:, 1:].to(text_input_ids.device) # [SEP] 卡 MASK 大师 [SEP]
            return_mlm = True
            
        if 'mfm' in sample_task:
            vm_input = visual_embedding
            input_feature, video_label = self.vm.torch_mask_frames(visual_embedding.cpu(), visual_attention_mask.cpu())
            video_feature = input_feature.to(visual_embedding.device)
            video_label = video_label.to(visual_embedding.device)
            
        if 'itm' in sample_task:
            input_feature, video_text_match_label = self.sv.torch_shuf_video(visual_embedding.cpu())
            video_feature = input_feature.to(visual_embedding.device)
            video_text_match_label = video_text_match_label.to(visual_embedding.device)
        
#         visual_embedding = self.project_linear(visual_embedding)

        features, lm_prediction_scores = self.bert(text_input_ids, 
                                                   attention_mask, 
                                                   token_type_ids, 
                                                   visual_embedding, 
                                                   visual_attention_mask, 
                                                   visual_token_type_ids, 
                                                   return_mlm=return_mlm)
        #features = torch.mean(features, 1)
        #embedding = self.newfc_hidden(features)

        if 'mlm' in sample_task:
            pred = lm_prediction_scores.contiguous().view(-1, self.vocab_size)
            masked_lm_loss = nn.CrossEntropyLoss()(pred, lm_label.contiguous().view(-1))
            loss += masked_lm_loss  / len(sample_task)
            
        if 'mfm' in sample_task:
            #print(features[:, -video_feature.size()[1]:, :].shape)
            vm_output = self.roberta_mvm_lm_header(features[:, -video_feature.size()[1]:, :])

            masked_vm_loss = self.calculate_mfm_loss(vm_output, vm_input, 
                                                     visual_attention_mask, video_label, normalize=False)
            loss += masked_vm_loss / 3 / len(sample_task)
            
        if 'itm' in sample_task:
            pred = self.newfc_itm(features[:, 0, :])
            itm_loss = nn.BCEWithLogitsLoss()(pred.view(-1), video_text_match_label.view(-1))
            loss += itm_loss #/ 100 / len(sample_task)

        #return (pred, loss, masked_lm_loss, itm_loss)
        return (pred, loss, masked_lm_loss)
    # calc mfm loss 
    def calculate_mfm_loss(self, video_feature_output, video_feature_input, 
                           video_mask, video_labels_index, normalize=False, temp=0.1):
        if normalize:
            video_feature_output = torch.nn.functional.normalize(video_feature_output, p=2, dim=2)
            video_feature_input = torch.nn.functional.normalize(video_feature_input, p=2, dim=2)

        # 512, 1536
        # torch.Size([16, 32, 1536]) torch.Size([16, 32, 768])
        afm_scores_tr = video_feature_output.view(-1, video_feature_output.shape[-1])
        
        # 768, 512
        video_tr = video_feature_input.permute(2, 0, 1)
        video_tr = video_tr.view(video_tr.shape[0], -1)
        
        
        
        logits_matrix = torch.mm(afm_scores_tr, video_tr)
        if normalize:
            logits_matrix = logits_matrix / temp

        video_mask_float = video_mask.to(dtype=torch.float)
        mask_matrix = torch.mm(video_mask_float.view(-1, 1), video_mask_float.view(1, -1))
        masked_logits = logits_matrix + (1. - mask_matrix) * -1e8

        logpt = F.log_softmax(masked_logits, dim=-1)
        logpt = torch.diag(logpt)
        nce_loss = -logpt

        video_labels_index_mask = (video_labels_index != -100)
        nce_loss = nce_loss.masked_select(video_labels_index_mask.view(-1))
        nce_loss = nce_loss.mean()
        return nce_loss
    

def save_pickle(dic, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(dic, f)


def load_pickle(load_path):
    with open(load_path, 'rb') as f:
        message_dict = pickle.load(f)
    return message_dict


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# def read_data(args, tokenizer: BertTokenizer) -> dict:
#     labeled_text = os.path.join(args.text_path, 'labeled.json')
#     unlabeled_text = os.path.join(args.text_path, 'unlabeled.json')

#     labeled_frame = os.path.join(args.frame_fea, 'labeled.zip')
#     unlabeled_frame = os.path.join(args.frame_fea, 'unlabeled.zip')

#     inputs = defaultdict(list)

#     with open(labeled_text, 'r', encoding='utf8') as f:
#         anns = json.load(f)

#     handles = zipfile.ZipFile(labeled_frame, 'r')
    
#     i = 0
#     for idx, line in enumerate(tqdm(anns, desc='Processing train data ... ...', total=len(anns))):

# #         if idx == 1000:
# #             break

#         vid = line['id']
#         title = line['title']
#         asr = line['asr']
#         ocr = '。'.join([d['text'] for d in line['ocr']])
        
#         raw_feats = np.load(BytesIO(handles.read(name=f'{vid}.npy')), allow_pickle=True)
        
#         build_model_inputs(args, inputs, tokenizer, title, asr, ocr, raw_feats)
        
#     with open(unlabeled_text, 'r', encoding='utf8') as f:
#         anns = json.load(f)

#     handles = zipfile.ZipFile(unlabeled_frame, 'r')
    
#     for idx, line in enumerate(tqdm(anns, desc='Processing train data ... ...', total=len(anns))):

#         if idx == 500000:
#             break

#         vid = line['id']
#         title = line['title']
#         asr = line['asr']
#         ocr = '。'.join([d['text'] for d in line['ocr']])
        
#         raw_feats = np.load(BytesIO(handles.read(name=f'{vid}.npy')), allow_pickle=True)
        
#         build_model_inputs(args, inputs, tokenizer, title, asr, ocr, raw_feats)
#     os.makedirs(os.path.dirname(args.data_cache_path), exist_ok=True)
#     save_pickle(inputs, args.data_cache_path)
    
#     return inputs

def read_data(args, tokenizer: BertTokenizer) -> dict:
    labeled_text = os.path.join(args.text_path, 'labeled.json')
    unlabeled_text = os.path.join(args.text_path, 'unlabeled.json')

    labeled_frame = os.path.join(args.frame_fea, 'labeled.zip')
    unlabeled_frame = os.path.join(args.frame_fea, 'unlabeled.zip')

    with open(labeled_text, 'r', encoding='utf8') as f:
        anns = json.load(f)

    handles = zipfile.ZipFile(labeled_frame, 'r')
    
    i = 1
    inputs = defaultdict(list)
    for idx, line in enumerate(tqdm(anns, desc='Processing train data ... ...', total=len(anns))):

#         if idx == 1000:
#             break
        
        if i % 55000 == 0:
            print('save_pickle:{}')
            save_pickle(inputs, f'./process_data/pkl/pretrain_{i//55000}.pkl')
            inputs = defaultdict(list)
        i += 1
        vid = line['id']
        title = line['title']
        asr = line['asr']
        ocr = '。'.join([d['text'] for d in line['ocr']])
        
        raw_feats = np.load(BytesIO(handles.read(name=f'{vid}.npy')), allow_pickle=True)
        
        build_model_inputs(args, inputs, tokenizer, title, asr, ocr, raw_feats)
        
    with open(unlabeled_text, 'r', encoding='utf8') as f:
        anns = json.load(f)

    handles = zipfile.ZipFile(unlabeled_frame, 'r')
    
    for idx, line in enumerate(tqdm(anns, desc='Processing train data ... ...', total=len(anns))):

#         if idx == 500000:
#             break
        if i % 55000 == 0:
            save_pickle(inputs, f'./process_data/pkl/pretrain_{i//55000}.pkl')
            inputs = defaultdict(list)
        i += 1
        
        vid = line['id']
        title = line['title']
        asr = line['asr']
        ocr = '。'.join([d['text'] for d in line['ocr']])
        
        raw_feats = np.load(BytesIO(handles.read(name=f'{vid}.npy')), allow_pickle=True)
        
        build_model_inputs(args, inputs, tokenizer, title, asr, ocr, raw_feats)
    #os.makedirs(os.path.dirname(args.data_cache_path), exist_ok=True)
    #save_pickle(inputs, args.data_cache_path)
    
    return inputs

def build_model_inputs(args, inputs, tokenizer, title, asr, ocr, raw_feats,
                       labels=None, level1_labels=None, test_mode=False):

    if title == '':
        title = '空'
    if asr == '':
        asr = '空'
    if ocr == '':
        ocr = '空'
    text =  title + '[SEP]' + asr + '[SEP]' + ocr
    bert_inputs = tokenizer.encode_plus(text, max_length=args.bert_seq_length, padding='max_length', truncation=True)
        
    text_input_ids = bert_inputs['input_ids']
    attention_mask = bert_inputs['attention_mask']
    token_type_ids = bert_inputs['token_type_ids']
    
    raw_feats = raw_feats.astype(np.float32)  # float16 to float32
    num_frames, feat_dim = raw_feats.shape

    feat = np.zeros((args.max_frames, feat_dim), dtype=np.float32)

    if num_frames <= args.max_frames:
        feat[:num_frames] = raw_feats
    else:
        # if the number of frames exceeds the limitation, we need to sample the frames.
        if test_mode:
            # uniformly sample when test mode is True
            step = num_frames // args.max_frames
            select_inds = list(range(0, num_frames, step))
            select_inds = select_inds[:args.max_frames]
        else:
            # randomly sample when test mode is False
            select_inds = list(range(num_frames))
            random.shuffle(select_inds)
            select_inds = select_inds[:args.max_frames]
            select_inds = sorted(select_inds)

        for i, j in enumerate(select_inds):
            feat[i] = raw_feats[j]

    visual_embedding = torch.FloatTensor(feat)
    visual_attention_mask = torch.ones(visual_embedding.shape[:-1], dtype=torch.float)
    visual_token_type_ids = torch.zeros(visual_embedding.shape[:-1], dtype=torch.long)
        
    inputs['text_input_ids'].append(text_input_ids)
    inputs['attention_mask'].append(attention_mask)
    inputs['token_type_ids'].append(token_type_ids)
    inputs['visual_embedding'].append(visual_embedding)
    inputs['visual_attention_mask'].append(visual_attention_mask)
    inputs['visual_token_type_ids'].append(visual_token_type_ids)


class WBDCCollator:
    def __init__(self, args):

        self.args = args
        self.max_seq_len = args.bert_seq_length

        self.batch_same_length = True

    def pad_and_truncate(self, input_ids_list, token_type_ids_list, attention_mask_list,
                         visual_embedding_list, visual_attention_mask_list, visual_token_type_ids_list,
                         max_seq_len):

        input_ids = torch.zeros((len(input_ids_list), max_seq_len), dtype=torch.long)
        token_type_ids = torch.zeros_like(input_ids)
        attention_mask = torch.zeros_like(input_ids)

        for i in range(len(input_ids_list)):

            seq_len = len(input_ids_list[i])

            if seq_len <= max_seq_len:
                input_ids[i, :seq_len] = torch.tensor(input_ids_list[i], dtype=torch.long)
                token_type_ids[i, :seq_len] = torch.tensor(token_type_ids_list[i], dtype=torch.long)
                attention_mask[i, :seq_len] = torch.tensor(attention_mask_list[i], dtype=torch.float)

            else:
                input_ids[i] = torch.tensor(input_ids_list[i][:max_seq_len - 1] + [self.tokenizer.sep_token_id],
                                            dtype=torch.long)
                token_type_ids[i] = torch.tensor(token_type_ids_list[i][:max_seq_len], dtype=torch.long)
                attention_mask[i] = torch.tensor(attention_mask_list[i][:max_seq_len], dtype=torch.float)

        visual_embedding = torch.zeros((len(visual_embedding_list), 32, 768), dtype=torch.float)
        visual_attention_mask = torch.zeros((len(visual_attention_mask_list), 32), dtype=torch.float)
        visual_token_type_ids = torch.zeros((len(visual_token_type_ids_list), 32), dtype=torch.long)

        for i in range(len(visual_embedding_list)):
            visual_embedding[i] = torch.tensor(visual_embedding_list[i], dtype=torch.float)
            visual_attention_mask[i] = torch.tensor(visual_attention_mask_list[i], dtype=torch.float)
            visual_token_type_ids[i] = torch.tensor(visual_token_type_ids_list[i], dtype=torch.long)

        return input_ids, token_type_ids, attention_mask, \
                   visual_embedding, visual_attention_mask, visual_token_type_ids

    def __call__(self, examples: list) -> dict:

        input_ids_list, token_type_ids_list, attention_mask_list, \
            visual_embedding_list, visual_attention_mask_list, visual_token_type_ids_list = \
                list(zip(*examples))


        if self.batch_same_length:
            cur_max_seq_len = max(len(input_id) for input_id in input_ids_list)
            max_seq_len = min(cur_max_seq_len, self.max_seq_len)
        else:
            max_seq_len = self.max_seq_len

        input_ids, token_type_ids, attention_mask, visual_embedding, visual_attention_mask, visual_token_type_ids = \
                self.pad_and_truncate(input_ids_list, token_type_ids_list, attention_mask_list,
                                      visual_embedding_list, visual_attention_mask_list, visual_token_type_ids_list,
                                      max_seq_len)


        data_dict = {
                'text_input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids,
                'visual_embedding': visual_embedding,
                'visual_attention_mask': visual_attention_mask,
                'visual_token_type_ids': visual_token_type_ids
            }


        return data_dict


class WBDCDataset(Dataset):
    def __init__(self, data_dict: dict):
        super(Dataset, self).__init__()
        self.data_dict = data_dict

    def __getitem__(self, index: int) -> tuple:
        data = (self.data_dict['text_input_ids'][index],
                self.data_dict['token_type_ids'][index],
                self.data_dict['attention_mask'][index],
                self.data_dict['visual_embedding'][index],
                self.data_dict['visual_attention_mask'][index],
                self.data_dict['visual_token_type_ids'][index])

        return data

    def __len__(self) -> int:
        return len(self.data_dict['text_input_ids'])


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

# def load_data(args, tokenizer):
#     print('Load dataset')
#     with open(args.data_cache_path, 'rb') as f:
#         train_data = pickle.load(f)

#     collate_fn = WBDCCollator(args)
#     print('Define dataset')
#     train_dataset = WBDCDataset(train_data)
#     del train_data
#     gc.collect()
#     print('Define DataLoader')
#     train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
#                                   num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=False)
#     return train_dataloader


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

    
def build_model_and_tokenizer(args):

    model = VisualBertMultiModal(args).to(args.device)
    tokenizer = BertTokenizer.from_pretrained(args.bert_dir)
    
    return tokenizer, model


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


def sorted_checkpoints(args, best_model_checkpoint, checkpoint_prefix=PREFIX_CHECKPOINT_DIR, use_mtime=False):
    ordering_and_checkpoint_path = []

    glob_checkpoints = [str(x) for x in Path(args.record_save_path).glob(f"{checkpoint_prefix}-*")]

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(f".*{checkpoint_prefix}-([0-9]+)", path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    # Make sure we don't delete the best model.
    if best_model_checkpoint is not None:
        best_model_index = checkpoints_sorted.index(str(Path(best_model_checkpoint)))
        checkpoints_sorted[best_model_index], checkpoints_sorted[-1] = (
            checkpoints_sorted[-1],
            checkpoints_sorted[best_model_index],
        )
    return checkpoints_sorted


def rotate_checkpoints(args, best_model_checkpoint, use_mtime=False) -> None:
    if args.save_total_limit is None or args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = sorted_checkpoints(args, best_model_checkpoint, use_mtime=use_mtime)
    if len(checkpoints_sorted) <= args.save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        shutil.rmtree(checkpoint)


def pretrain(args):
    print('\n>> start pretraining ... ...')
    print(f'\n>> loading from pretrain model path -> {args.bert_dir}')

    tokenizer, model = build_model_and_tokenizer(args)
    
    if args.device == 'cuda':
        model = torch.nn.parallel.DataParallel(model.to(args.device))
        
#     if not os.path.exists(os.path.join(args.data_cache_path)):
#         read_data(args, tokenizer)

    train_dataloader = load_data(args, tokenizer)

    total_steps = args.num_epochs * len(train_dataloader)

    optimizer, scheduler = build_optimizer(args, model, total_steps)

    total_loss, cur_avg_loss, global_steps = 0., 0., 0

    if args.fp16:
        scaler = GradScaler()

    pretrain_loss_list, global_steps_list = [], []

    for epoch in range(1, args.num_epochs + 1):
        
        train_iterator = tqdm(train_dataloader, desc=f'Epoch : {epoch}', total=len(train_dataloader))
        
        model.train()

        for step, batch in enumerate(train_iterator):
            batch_cuda = batch2cuda(args, batch)
            
            if args.fp16:
                with autocast():
                    #pred, loss, masked_lm_loss, itm_loss = model(**batch_cuda)
                    pred, loss, masked_lm_loss = model(**batch_cuda)
                    loss = loss.mean()
                    masked_lm_loss = masked_lm_loss.mean()
                    #itm_loss = itm_loss.mean()
                scaler.scale(loss).backward()
            else:
                #pred, loss, masked_lm_loss, itm_loss = model(**batch_cuda)
                pred, loss, masked_lm_loss = model(**batch_cuda)
                loss = loss.mean()
                masked_lm_loss = masked_lm_loss.mean()
                #itm_loss = itm_loss.mean()
                loss.backward()

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            pretrain_loss_list.append(loss.item())
            global_steps_list.append(global_steps + 1)

            total_loss += loss.item()
            cur_avg_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                if args.fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()

                if (global_steps + 1) % args.logging_step == 0:
                    epoch_avg_loss = cur_avg_loss / args.logging_step
                    global_avg_loss = total_loss / (global_steps + 1)

                    print(f"\n>> epoch - {epoch},  global steps - {global_steps + 1}, "
                          f"epoch avg loss - {epoch_avg_loss:.4f}, global avg loss - {global_avg_loss:.4f}.")

                    cur_avg_loss = 0.0
                global_steps += 1

                lr = scheduler.get_last_lr()[0]
#                 train_iterator.set_postfix_str(f'loss : {loss.item():.4f}, masked_lm_loss : {masked_lm_loss.item():.4f}, itm_loss : {itm_loss.item():.4f}, lr : {lr}, global steps : {global_steps} .')
                train_iterator.set_postfix_str(f'loss : {loss.item():.4f}, masked_lm_loss : {masked_lm_loss.item():.4f}, lr : {lr}, global steps : {global_steps} .')

            if (global_steps + 1) % args.save_steps == 0:
                torch.save({'epoch': epoch, 'model_state_dict': model.module.state_dict()},
                           f'{args.record_save_path}/model_{step}.bin')
#                 save_model(args, model, tokenizer, global_steps)
#                 last_checkpoint_save_path = os.path.join(args.record_save_path, f'checkpoint-{global_steps}')
#                 rotate_checkpoints(args, last_checkpoint_save_path, use_mtime=False)
        torch.save({'epoch': epoch, 'model_state_dict': model.module.state_dict()},
                           f'{args.record_save_path}/model_{epoch}.bin')
    print('\n>> saving model at last epoch ... ...')
    torch.save({'epoch': epoch, 'model_state_dict': model.module.state_dict()},
                           f'{args.record_save_path}/model_{step}.bin')

    fig, ax = plt.subplots()
    ax.plot(global_steps_list, pretrain_loss_list, 'k', label='pretrain_loss')
    legend = ax.legend(loc='best', shadow=True, fontsize='large')
    legend.get_frame().set_facecolor('#00FFCC')

    fig_save_path = os.path.join(args.save_path, 'train_loss_curve.jpg')
    plt.savefig(fig_save_path)
    plt.show()

    del model, tokenizer, optimizer, scheduler
    torch.cuda.empty_cache()
    gc.collect()


def main():
    parser = ArgumentParser()
    
    # ========================= Data Configs ==========================
    parser.add_argument('--train_annotation', type=str, default='../challenge/data/annotations/labeled.json')
    parser.add_argument('--test_annotation', type=str, default='../challenge/data/annotations/test_a.json')
    parser.add_argument('--text_path', type=str, default='../challenge/data/annotations')
    parser.add_argument('--frame_fea', type=str, default='../challenge/data/zip_feats')
    parser.add_argument('--train_zip_feats', type=str, default='../challenge/data/zip_feats/labeled.zip')
    parser.add_argument('--test_zip_feats', type=str, default='../challenge/data/zip_feats/test_a.zip')
    parser.add_argument('--test_output_csv', type=str, default='data/result.csv')
    parser.add_argument('--batch_size', default=64*2, type=int, help="use for training duration per worker")
    parser.add_argument('--val_batch_size', default=256, type=int, help="use for validation duration per worker")
    parser.add_argument('--test_batch_size', default=256, type=int, help="use for testing duration per worker")
    parser.add_argument('--num_workers', default=4, type=int, help="num_workers for dataloaders")

    # ========================= Learning Configs ==========================
    parser.add_argument('--max_steps', default=50000, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--print_steps', type=int, default=20, help="Number of steps to log training metrics.")
    parser.add_argument('--warmup_steps', default=1000, type=int, help="warm ups for parameters not in bert or vit")
    parser.add_argument('--minimum_lr', default=0., type=float, help='minimum learning rate')
    parser.add_argument('--learning_rate', default=5e-5, type=float, help='initial learning rate')
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--patience", default=1, type=float, help="Early Stop.")
    parser.add_argument("--scheduler", default='linear', type=str, help="scheduler") # cosine

    # ========================== Title BERT =============================
    parser.add_argument('--bert_dir', type=str, default='../challenge/prev_pretrained_models/chinese-roberta-wwm-ext')
    parser.add_argument('--bert_tokenizer_dir', type=str, default='../challenge/prev_pretrained_models/chinese-roberta-wwm-ext')
    parser.add_argument('--bert_cache', type=str, default='data/cache')
    parser.add_argument('--bert_seq_length', type=int, default=256)
    parser.add_argument('--bert_embeddding_size', type=int, default=768)
    parser.add_argument('--bert_output_size', type=int, default=768)
    parser.add_argument('--bert_warmup_steps', type=int, default=5000)
    parser.add_argument('--bert_max_steps', type=int, default=30000)
    parser.add_argument("--bert_hidden_dropout_prob", type=float, default=0.1)

    # ========================== Video =============================
    parser.add_argument('--frame_embedding_size', type=int, default=768)
    parser.add_argument('--max_frames', type=int, default=32)
    parser.add_argument('--vlad_cluster_size', type=int, default=64)
    parser.add_argument('--vlad_groups', type=int, default=8)
    parser.add_argument('--vlad_hidden_size', type=int, default=1024, help='nextvlad output size using dense')
    parser.add_argument('--se_ratio', type=int, default=8, help='reduction factor in se context gating')
    parser.add_argument('--seed', type=int, default=32)
    # ========================== Fusion Layer =============================
    parser.add_argument('--fc_size', type=int, default=512, help="linear size before final linear")

    parser.add_argument('--save_path', type=str,
                        default='./new_self_pretrained_model/chinese-roberta-wwm-ext-e10_10_1_110W_mlm')
    parser.add_argument('--record_save_path', type=str,
                        default='./new_self_pretrained_model_record/chinese-roberta-wwm-ext-e10_10_1_110W_mlm')

    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--eps', type=float, default=1e-8)

    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)

    parser.add_argument('--warmup_ratio', type=float, default=0.1)

    parser.add_argument('--save_steps', type=int, default=4000)
    parser.add_argument('--save_total_limit', type=int, default=10)

    parser.add_argument('--logging_step', type=int, default=2000)

    parser.add_argument('--device', type=str, default='cuda')

    # ========================== text masked pro =============================
    parser.add_argument('--mask_modal', type=bool, default=True, help='mask_modal')
    parser.add_argument('--masked_title_pro', type=float, default=0.15)
    parser.add_argument('--masked_asr_pro', type=float, default=0.17)
    parser.add_argument('--masked_ocr_pro', type=float, default=0.25)
    
    # ========================== HMC =============================
    parser.add_argument('--hmc_loss', type=bool, default=False, help='use the hmc loss')

    parser.add_argument('--hmc_lambda', type=float, default=0.5)
    parser.add_argument('--hmc_beta', type=float, default=0.5)
    
    # ========================== Tricks =============================
    parser.add_argument('--fp16', type=str, default=True, help='use the fp16')
    parser.add_argument('--data_cache_path', type=str,
                        default=f'./process_data/pkl/pretrain.pkl')

    parser.add_argument('--do_fgm', type=bool, default=True, help='use the fgm')
    parser.add_argument('--fgm_epsilon', type=float, default=0.5)
    parser.add_argument('--do_pgd', type=bool, default=False, help='use the pgd')
    parser.add_argument('--do_ema', type=bool, default=True, help='use the pgd')
    parser.add_argument('--clear_text', type=bool, default=True, help='clear_text')
    

    warnings.filterwarnings('ignore')
    args = parser.parse_args()


    args.n_gpus = torch.cuda.device_count()
    #args.batch_size *= args.n_gpus

    create_dirs(args.save_path)
    create_dirs(args.record_save_path)

    seed_everything(args.seed)

    pretrain(args)


if __name__ == '__main__':
    main()
