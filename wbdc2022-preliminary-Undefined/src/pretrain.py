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
from pretrain_utils import WarmupLinearSchedule, get_scheduler, build_optimizer, batch2cuda, create_dirs, save_model, create_dataloaders

import gc
import zipfile
from io import BytesIO
warnings.filterwarnings('ignore')
    
def pretrain(args):
    
    print('Define the train dataloader')
    train_dataloader = create_dataloaders(args)
    model = WCUniModel(args, task=['mlm', 'itm'])
    
    if args.device == 'cuda':
        model = torch.nn.parallel.DataParallel(model.to(args.device))
        
#     if not os.path.exists(os.path.join(args.data_cache_path)):
#         read_data(args, tokenizer)

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
            
            input_ids, attention_mask = batch_cuda['text_input'], batch_cuda['text_mask']
            visual_embeds, visual_attention_mask = batch_cuda['frame_input'], batch_cuda['frame_mask']
            if args.fp16:
                with autocast():
                    loss, masked_lm_loss, itm_loss = model(visual_embeds, visual_attention_mask, input_ids, attention_mask)
                    loss = loss.mean()
                    masked_lm_loss = masked_lm_loss.mean()
                    #masked_vm_loss = masked_lm_loss.mean()
                    itm_loss = itm_loss.mean()
                scaler.scale(loss).backward()
            else:
                loss, masked_lm_loss, itm_loss = model(visual_embeds, visual_attention_mask, input_ids, attention_mask)
                loss = loss.mean()
                masked_lm_loss = masked_lm_loss.mean()
                #masked_vm_loss = masked_lm_loss.mean()
                itm_loss = itm_loss.mean()
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
                
                train_iterator.set_postfix_str(f'loss : {loss.item():.4f}, masked_lm_loss : {masked_lm_loss.item():.4f}, global steps : {global_steps} .')

        torch.save({'epoch': epoch, 'model_state_dict': model.module.state_dict()},
                           f'{args.record_save_path}/model_{epoch}.bin')

    del model, tokenizer, optimizer, scheduler
    torch.cuda.empty_cache()
    gc.collect()
    
def main():
    args = parse_args()
    setup_device(args)
    setup_seed(args)

    os.makedirs(args.savedmodel_path, exist_ok=True)
    
    pretrain(args)

if __name__ == '__main__':
    main()