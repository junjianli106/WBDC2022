import json
import random
import zipfile
from io import BytesIO

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import BertTokenizer, AutoTokenizer
from sklearn.model_selection import KFold, StratifiedKFold

from category_id_map import category_id_to_lv2id
import random

def create_dataloaders(args):
    with open(args.train_annotation, 'r', encoding='utf8') as f:
        anns = json.load(f)
    label = [a['category_id'] for a in anns]
    X = np.array(anns)
    y = np.array(label)
    
    fold = int(1 / args.val_ratio)
    skf = StratifiedKFold(n_splits=fold, random_state=args.seed, shuffle=True)
    
    for idx, (train_index, val_index) in enumerate(skf.split(X, y)):
        if idx != 0:
            break
        print('Fold:', idx, 'TRAIN:', len(train_index), "VAL:", len(val_index))

        X_train, X_val = X[train_index], X[val_index]

    train_dataset = MultiModalDataset(args, X_train, args.train_zip_feats, text_mask=True)
    val_dataset = MultiModalDataset(args, X_val, args.train_zip_feats)
        
    train_sampler = RandomSampler(train_dataset)
    val_sampler = SequentialSampler(val_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  sampler=train_sampler,
                                  drop_last=True,
                                  pin_memory=True,
                                  num_workers=args.num_workers)
    val_dataloader = DataLoader(val_dataset, 
                                batch_size=args.val_batch_size,
                                sampler=val_sampler,
                                drop_last=False,
                                pin_memory=True,
                                num_workers=args.num_workers)
    return train_dataloader, val_dataloader


class MultiModalDataset(Dataset):

    def __init__(self,
                 args,
                 anns,
                 zip_feats,
                 text_mask=False,
                 test_mode=False):

        self.max_frame = args.max_frames
        self.bert_seq_length = args.bert_seq_length
        self.text_mask = text_mask
        self.test_mode = test_mode
        
        self.mask_modal = args.mask_modal
        self.masked_title_pro = args.masked_title_pro
        self.masked_asr_pro = args.masked_asr_pro
        self.masked_ocr_pro = args.masked_ocr_pro

        # lazy initialization for zip_handler to avoid multiprocessing-reading error
        self.zip_feat_path = zip_feats
        self.handles = [None for _ in range(args.num_workers)]

        # load annotations
        self.anns = anns

        # initialize the text tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(args.bert_tokenizer_dir, use_fast=True, cache_dir=args.bert_cache)

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
        token_type = torch.zeros(feat.shape[:-1], dtype=torch.long)
        return feat, mask, token_type

    def tokenize_text(self, text: str) -> tuple:
        encoded_inputs = self.tokenizer(text, max_length=self.bert_seq_length, padding='max_length', truncation=True)
        input_ids = np.array(encoded_inputs['input_ids'])
        mask = np.array(encoded_inputs['attention_mask'])
        token_type = np.array(encoded_inputs['token_type_ids'])
        return input_ids, mask, token_type

    def __getitem__(self, idx: int) -> dict:
        # Step 1, load visual features from zipfile.
        worker_info = torch.utils.data.get_worker_info()
        frame_input, frame_mask, frame_token_type = self.get_visual_feats(worker_info.id, idx)
            
        # Step 2, load title tokens
        title_replace_p = random.random()
        asr_replace_p = random.random()
        ocr_replace_p = random.random()
        
        title = self.anns[idx]['title']
        asr = self.anns[idx]['asr']
        ocr = '。'.join([d['text'] for d in self.anns[idx]['ocr']])
        
        if not self.test_mode and self.mask_modal:
        
            if title_replace_p < self.masked_title_pro:
                titie = ''

            if asr_replace_p < self.masked_asr_pro:
                asr = ''

            if ocr_replace_p < self.masked_ocr_pro:
                ocr = ''

#         if not self.test_mode:
        
#             if title_replace_p < 0.17:
#                 titie = ''

#             if asr_replace_p < 0.25:
#                 asr = ''

#             if ocr_replace_p < 0.25:
#                 ocr = ''
#         title = title[:64] + title[-64:]
#         asr = asr[:64] + asr[-64:]
#         ocr = ocr[:64] + ocr[-64:]
        text =  title + '[SEP]' + asr + '[SEP]' + ocr
#         if self.clear_text:
#             text = text.replace(';;', ';').replace(' ', '').replace('…', '').replace('。;', ';').replace('JosephPilates', '')
#             text = text.replace('\xa0', ' ').replace('⋛', '').replace('㇀', '').replace('嗯', '').replace('但。', '')
        text_input, text_mask, text_token_type = self.tokenize_text(text)


        # Step 3, summarize into a dictionary
        data = dict(
            frame_input=frame_input,
            frame_mask=frame_mask,
            frame_token_type=frame_token_type,
            text_input=text_input,
            text_mask=text_mask,
            text_token_type=text_token_type
        )

        # Step 4, load label if not test mode
        if not self.test_mode:
            label = category_id_to_lv2id(self.anns[idx]['category_id'])
            data['label'] = torch.LongTensor([label])

        return data




# import json
# import random
# import zipfile
# from io import BytesIO

# import numpy as np
# import torch
# from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
# from transformers import BertTokenizer

# from category_id_map import category_id_to_lv2id, category_id_to_lv1id
# import random

# def create_dataloaders(args):
#     dataset = MultiModalDataset(args, args.train_annotation, args.train_zip_feats)
#     size = len(dataset)
#     val_size = int(size * args.val_ratio)
#     train_dataset, val_dataset = torch.utils.data.random_split(dataset, [size - val_size, val_size],
#                                                                generator=torch.Generator().manual_seed(args.seed))

#     train_sampler = RandomSampler(train_dataset)
#     val_sampler = SequentialSampler(val_dataset)
#     train_dataloader = DataLoader(train_dataset,
#                                   batch_size=args.batch_size,
#                                   sampler=train_sampler,
#                                   drop_last=True,
#                                   pin_memory=True,
#                                   num_workers=args.num_workers)
#     val_dataloader = DataLoader(val_dataset, 
#                                 batch_size=args.val_batch_size,
#                                 sampler=val_sampler,
#                                 drop_last=False,
#                                 pin_memory=True,
#                                 num_workers=args.num_workers)
#     return train_dataloader, val_dataloader


# class MultiModalDataset(Dataset):
#     """ A simple class that supports multi-modal inputs.

#     For the visual features, this dataset class will read the pre-extracted
#     features from the .npy files. For the title information, it
#     uses the BERT tokenizer to tokenize. We simply ignore the ASR & OCR text in this implementation.

#     Args:
#         ann_path (str): annotation file path, with the '.json' suffix.
#         zip_feats (str): visual feature zip file path.
#         test_mode (bool): if it's for testing.
#     """

#     def __init__(self,
#                  args,
#                  ann_path: str,
#                  zip_feats: str,
#                  test_mode: bool = False):
#         self.max_frame = args.max_frames
#         self.bert_seq_length = args.bert_seq_length
#         self.test_mode = test_mode
#         self.mask_modal = args.mask_modal
#         self.clear_text = args.clear_text
        
#         self.masked_title_pro = args.masked_title_pro
#         self.masked_asr_pro = args.masked_asr_pro
#         self.masked_ocr_pro = args.masked_ocr_pro
#         # lazy initialization for zip_handler to avoid multiprocessing-reading error
#         self.zip_feat_path = zip_feats
#         self.handles = [None for _ in range(args.num_workers)]
#         self.hmc_loss = args.hmc_loss
#         # load annotations
#         with open(ann_path, 'r', encoding='utf8') as f:
#             self.anns = json.load(f)

#         # initialize the text tokenizer
#         self.tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer_dir, use_fast=True, cache_dir=args.bert_cache)

#     def __len__(self) -> int:
#         return len(self.anns)

#     def get_visual_feats(self, worker_id, idx: int) -> tuple:
#         # read data from zipfile
#         vid = self.anns[idx]['id']
#         if self.handles[worker_id] is None:
#             self.handles[worker_id] = zipfile.ZipFile(self.zip_feat_path, 'r')
#         raw_feats = np.load(BytesIO(self.handles[worker_id].read(name=f'{vid}.npy')), allow_pickle=True)
#         raw_feats = raw_feats.astype(np.float32)  # float16 to float32
#         num_frames, feat_dim = raw_feats.shape

#         feat = np.zeros((self.max_frame, feat_dim), dtype=np.float32)
#         mask = np.ones((self.max_frame,), dtype=np.int32)
#         if num_frames <= self.max_frame:
#             feat[:num_frames] = raw_feats
#             mask[num_frames:] = 0
#         else:
#             # if the number of frames exceeds the limitation, we need to sample
#             # the frames.
#             if self.test_mode:
#                 # uniformly sample when test mode is True
#                 step = num_frames // self.max_frame
#                 select_inds = list(range(0, num_frames, step))
#                 select_inds = select_inds[:self.max_frame]
#             else:
#                 # randomly sample when test mode is False
#                 select_inds = list(range(num_frames))
#                 random.shuffle(select_inds)
#                 select_inds = select_inds[:self.max_frame]
#                 select_inds = sorted(select_inds)
#             for i, j in enumerate(select_inds):
#                 feat[i] = raw_feats[j]
#         feat = torch.FloatTensor(feat)
#         mask = torch.LongTensor(mask)
#         token_type = torch.zeros(feat.shape[:-1], dtype=torch.long)
#         return feat, mask, token_type

#     def tokenize_text(self, text: str) -> tuple:
#         encoded_inputs = self.tokenizer(text, max_length=self.bert_seq_length, padding='max_length', truncation=True)
#         input_ids = np.array(encoded_inputs['input_ids'])
#         mask = np.array(encoded_inputs['attention_mask'])
#         token_type = np.array(encoded_inputs['token_type_ids'])
#         return input_ids, mask, token_type

#     def __getitem__(self, idx: int) -> dict:
#         # Step 1, load visual features from zipfile.
#         worker_info = torch.utils.data.get_worker_info()
#         frame_input, frame_mask, frame_token_type = self.get_visual_feats(worker_info.id, idx)
            
#         # Step 2, load title tokens
#         title_replace_p = random.random()
#         asr_replace_p = random.random()
#         ocr_replace_p = random.random()
        
#         title = self.anns[idx]['title']
#         asr = self.anns[idx]['asr']
#         ocr = '。'.join([d['text'] for d in self.anns[idx]['ocr']])
        
#         if not self.test_mode and self.mask_modal:
        
#             if title_replace_p < self.masked_title_pro:
#                 titie = ''

#             if asr_replace_p < self.masked_asr_pro:
#                 asr = ''

#             if ocr_replace_p < self.masked_ocr_pro:
#                 ocr = ''

# #         if not self.test_mode:
        
# #             if title_replace_p < 0.17:
# #                 titie = ''

# #             if asr_replace_p < 0.25:
# #                 asr = ''

# #             if ocr_replace_p < 0.25:
# #                 ocr = ''
# #         title = title[:64] + title[-64:]
# #         asr = asr[:64] + asr[-64:]
# #         ocr = ocr[:64] + ocr[-64:]
#         text =  title + '[SEP]' + asr + '[SEP]' + ocr
# #         if self.clear_text:
# #             text = text.replace(';;', ';').replace(' ', '').replace('…', '').replace('。;', ';').replace('JosephPilates', '')
# #             text = text.replace('\xa0', ' ').replace('⋛', '').replace('㇀', '').replace('嗯', '').replace('但。', '')
#         text_input, text_mask, text_token_type = self.tokenize_text(text)

#         # Step 3, summarize into a dictionary
#         data = dict(
#             frame_input=frame_input,
#             frame_mask=frame_mask,
#             frame_token_type=frame_token_type,
#             text_input=text_input,
#             text_mask=text_mask,
#             text_token_type=text_token_type
#         )

#         # Step 4, load label if not test mode
#         if not self.test_mode:
#             label = category_id_to_lv2id(self.anns[idx]['category_id'])
#             data['label'] = torch.LongTensor([label])
            
#         if self.hmc_loss:
#             l1_label = category_id_to_lv1id(self.anns[idx]['category_id'])
#             l2_label = category_id_to_lv2id(self.anns[idx]['category_id'])
            
#             data['l1_label'] = torch.LongTensor([l1_label])
#             data['l2_label'] = torch.LongTensor([l2_label])

#         return data

