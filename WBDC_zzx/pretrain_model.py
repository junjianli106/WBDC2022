import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import BertModel, AutoModel, VisualBertConfig, VisualBertModel, BertTokenizer

from category_id_map import CATEGORY_ID_LIST
import random


class MyVisualBert(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model = AutoModel.from_pretrained(args.bert_dir)
        print(type(self.model))
        config = self.model.config
        self.visual_proj = nn.Linear(768, config.hidden_size)
        self.args = args
        # self.activate = nn.LeakyReLU()
        # self.activate = nn.Tanh()
        # self.activate = nn.ReLU()
        # self.activate = nn.GELU()
        self.classifier = nn.Linear(config.hidden_size, len(CATEGORY_ID_LIST))
        # self.classifier = nn.Linear(2 * config.hidden_size, len(CATEGORY_ID_LIST))
        self.tokenizer = BertTokenizer.from_pretrained(
            args.bert_dir, use_fast=True, cache_dir=args.bert_cache)
        
        self.itm_head = nn.Linear(config.hidden_size, 1)
    
    def mask_text(self, input_ids, attention_mask):
        mask_input_ids = []
        mask_output_ids = []
        mask_ratio = 0.15
        mask_token_id = self.tokenizer.mask_token_id
        for ids, masks in zip(input_ids.tolist(), attention_mask.tolist()):
            input_id = []
            output_id = []
            for id, mask in zip(ids, masks):
                
                if mask == 0:
                    input_id.append(id)
                    output_id.append(-100)
                
                else:
                
                    p = random.random()
                    
                    if p < mask_ratio * 0.8:
                        input_id.append(mask_token_id)
                        output_id.append(id)
                        
                    elif p < mask_ratio * 0.9:
                        input_id.append(id)
                        output_id.append(id)
                        
                    elif p < mask_ratio:
                        input_id.append(np.random.choice(len(self.tokenizer)))
                        output_id.append(id)
                        
                    else:
                        input_id.append(id)
                        output_id.append(-100)
                        
            mask_input_ids.append(input_id)
            mask_output_ids.append(output_id)
            
        return torch.tensor(mask_input_ids, dtype=torch.long, device=input_ids.device), torch.tensor(mask_output_ids, dtype=torch.long, device=input_ids.device)             
    
    def get_mlm_loss(self, mask_input_ids, mask_output_ids, attention_mask, visual_embeds, visual_attention_mask):
        emb_layer = self.model.get_input_embeddings()
        word_emb = emb_layer(mask_input_ids)
        img_emb = self.visual_proj(visual_embeds)
        # img_emb = self.activate(img_emb)
        inputs_embeds = torch.cat([word_emb, img_emb], dim=1)
        
        mask = torch.cat([attention_mask, visual_attention_mask], dim=1)
        output = self.model(inputs_embeds=inputs_embeds, attention_mask=mask)

        text_len = mask_output_ids.size(1)
        last_hid = output.last_hidden_state[:, :text_len, :]
        
        trans_weight = emb_layer.weight
        logits = torch.einsum('abc,cd->abd', [last_hid, trans_weight.T])
        pred = logits.view(-1, self.model.config.vocab_size)
        target = mask_output_ids.view(-1)
        loss = nn.CrossEntropyLoss()(pred, target)
        return loss
    
    
    def get_itm_loss(self, input_ids, attention_mask, visual_embeds, visual_attention_mask, label):
        
        emb_layer = self.model.get_input_embeddings()
        word_emb = emb_layer(input_ids)
        img_emb = self.visual_proj(visual_embeds)
        # img_emb = self.activate(img_emb)
        inputs_embeds = torch.cat([word_emb, img_emb], dim=1)
        
        mask = torch.cat([attention_mask, visual_attention_mask], dim=1)
        output = self.model(inputs_embeds=inputs_embeds, attention_mask=mask)
        
        logits = self.itm_head(output.last_hidden_state[:, 0, :])
        loss = nn.BCEWithLogitsLoss()(logits.view(-1), label.view(-1))
        return loss
    
    def get_mfm_loss(self, input_ids, attention_mask, visual_embeds, visual_attention_mask, label):
        emb_layer = self.model.get_input_embeddings()
        word_emb = emb_layer(input_ids)
        img_emb = self.visual_proj(visual_embeds)
        # img_emb = self.activate(img_emb)
        inputs_embeds = torch.cat([word_emb, img_emb], dim=1)
        
        mask = torch.cat([attention_mask, visual_attention_mask], dim=1)
        output = self.model(inputs_embeds=inputs_embeds, attention_mask=mask)
        
        text_len = input_ids.size(1)
        out_feature = output.last_hidden_state[:, text_len:, :]
        out_feature = out_feature.reshape(-1, out_feature.size(-1))
        in_feature = visual_embeds.view(visual_embeds.size(-1), -1)
        logits_matrix = torch.mm(out_feature, in_feature)
        
        video_mask_float = visual_attention_mask.float()
        mask_matrix = torch.mm(video_mask_float.view(-1, 1), video_mask_float.view(1, -1))
        masked_logits = logits_matrix + (1. - mask_matrix) * -1e8 # [bs*video_len, bs*video_len]
        
        logpt = F.log_softmax(masked_logits, dim=-1)
        logpt = torch.diag(logpt)
        nce_loss = -logpt

        video_labels_index_mask = (label != -100)
        nce_loss = nce_loss.masked_select(video_labels_index_mask.view(-1))
        nce_loss = nce_loss.mean()
        return nce_loss
    
    def shuffle_video(self, video_feature, video_mask):
        bs = video_feature.size(0)
        shuf_index = torch.tensor(list(range(bs // 2)) + list(range(bs // 2, bs))[::-1])
        label = (torch.tensor(list(range(bs))) == shuf_index).float().to(video_feature.device)
        # print(label)

        new_feature = video_feature[shuf_index]
        new_mask = video_mask[shuf_index]
        
        return new_feature, new_mask, label
    
    def mask_frame(self, video_feature, video_mask):
        mlm_probability = 0.15
        probability_matrix = torch.full(video_mask.shape, 0.9 * mlm_probability, device=video_feature.device)
        probability_matrix = probability_matrix * video_mask
        
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        video_labels_index = torch.arange(video_feature.size(0) * video_feature.size(1), device=video_feature.device).view(-1, video_feature.size(1))
        video_labels_index = -100 * ~masked_indices + video_labels_index * masked_indices

        # 90% mask video fill all 0.0
        masked_indices_unsqueeze = masked_indices.unsqueeze(-1).expand_as(video_feature)
        inputs = video_feature.data.masked_fill(masked_indices_unsqueeze, 0.0)
        # labels = video_feature[masked_indices_unsqueeze].contiguous().view(-1, video_feature.size(2)) 

        return inputs, video_labels_index
        
        
    
    def forward(self, inputs, inference=False):
        
        input_ids = inputs['title_input']
        attention_mask = inputs['title_mask']
        visual_embeds = inputs['frame_input']
        visual_attention_mask = inputs['frame_mask']
        
        
        # mlm task
        mask_input_ids, mask_output_ids = self.mask_text(input_ids, attention_mask)  
        
        mlm_loss = self.get_mlm_loss(mask_input_ids, mask_output_ids, attention_mask, visual_embeds, visual_attention_mask)
        
        # itm (image text matching)
        itm_visual_embeds, itm_visual_attention_mask, itm_label = self.shuffle_video(visual_embeds, visual_attention_mask)
        itm_loss = self.get_itm_loss(input_ids, attention_mask, itm_visual_embeds, itm_visual_attention_mask, itm_label)
        
        # mfm
        mask_visual_embeds, mfm_labels = self.mask_frame(visual_embeds, visual_attention_mask)
        mfm_loss = self.get_mfm_loss(input_ids, attention_mask, mask_visual_embeds, visual_attention_mask, mfm_labels)

        if inference:
            return torch.argmax(prediction, dim=1)
        else:
            # return self.cal_loss(prediction, inputs['label'])
            # loss = itm_loss
            # loss = mfm_loss
            # loss = (mlm_loss + itm_loss) / 2
            mlm_loss *= self.args.mlm_co
            itm_loss *= self.args.itm_co
            mfm_loss *= self.args.mfm_co
            
            loss = mlm_loss / 3 + itm_loss / 3 + mfm_loss / 3
            return loss, mlm_loss, itm_loss, mfm_loss #{'mlm_loss': mlm_loss.item(), 'itm_loss': itm_loss.item(), 'mfm_loss': mfm_loss.item()}

    @staticmethod
    def cal_loss(prediction, label):
        label = label.squeeze(dim=1)
        loss = F.cross_entropy(prediction, label)
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label
