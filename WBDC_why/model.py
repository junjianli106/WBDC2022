import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel, BertConfig
from transformers import LxmertModel, LxmertConfig
from transformers import AutoModel, AutoConfig

from timm.models.vision_transformer import VideoTransformer
from models.bert.modeling_bert import UniBertModel
from models.bert.modeling_mm_bert import BertModel as ALBEFModel

from category_id_map import CATEGORY_ID_LIST
from loss import FocalLoss
    
class UniBertMultiModal(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.bert = UniBertModel.from_pretrained(args.bert_dir, cache_dir=args.bert_cache)
        self.classifier = nn.Linear(self.bert.config.hidden_size, len(CATEGORY_ID_LIST))
        self.crit = nn.CrossEntropyLoss()
        # self.crit = FocalLoss(num_class=len(CATEGORY_ID_LIST))
        
    def forward(self, inputs, inference=False):
        bert_embedding = self.bert(inputs['frame_input'], inputs['frame_mask'], 
                          inputs['text_input'], inputs['text_mask'], inputs['text_token_type_ids'])
        bert_embedding = torch.mean(bert_embedding, dim=1)
        prediction = self.classifier(bert_embedding)

        if inference:
            return prediction
        else:
            return self.cal_loss(prediction, inputs['label'], self.crit)

    @staticmethod
    def cal_loss(prediction, label, crit):
        label = label.squeeze(dim=1)
        loss = crit(prediction, label)
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label
    
class ALBEFMultiModal(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.bert_config = AutoConfig.from_pretrained(args.bert_dir, cache_dir=args.bert_cache)
        self.bert_config.fusion_layer = int(self.bert_config.num_hidden_layers // 2)
        self.bert_config.encoder_width = self.bert_config.hidden_size
        self.bert = ALBEFModel.from_pretrained(args.bert_dir, cache_dir=args.bert_cache, config=self.bert_config)
        self.classifier = nn.Linear(self.bert.config.hidden_size, len(CATEGORY_ID_LIST))
        self.crit = nn.CrossEntropyLoss()
        # self.crit = FocalLoss(num_class=len(CATEGORY_ID_LIST))
        
    def forward(self, inputs, inference=False):
        bert_embedding = self.bert(inputs['frame_input'], inputs['frame_mask'], 
                          inputs['text_input'], inputs['text_mask'], inputs['text_token_type_ids'])
        bert_embedding = torch.mean(bert_embedding, dim=1)
        prediction = self.classifier(bert_embedding)

        if inference:
            return prediction
        else:
            return self.cal_loss(prediction, inputs['label'], self.crit)

    @staticmethod
    def cal_loss(prediction, label, crit):
        label = label.squeeze(dim=1)
        loss = crit(prediction, label)
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label
    
class DualBertMultiModal(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        # text_layer bert initalize
        self.text_bert = AutoModel.from_pretrained(args.bert_dir, cache_dir=args.bert_cache)
        self.text_proj = nn.Sequential(
            nn.Linear(self.text_bert.config.hidden_size, args.fusion_proj_size, bias=False),
            nn.Dropout(args.dropout),
            nn.LayerNorm(args.fusion_proj_size, eps=1e-6))
        
        # video_layer vit initalize
        self.video_proj = nn.Sequential(
                    nn.Linear(args.frame_embedding_size, args.fusion_proj_size, bias=False),
                    nn.Dropout(args.dropout),
                    nn.LayerNorm(args.fusion_proj_size, eps=1e-6))
        
        # fusion bert random initialize
        self.fusion_bert_config = BertConfig()
        self.fusion_bert_config.num_hidden_layers = args.fusion_num_hidden_layers
        self.fusion_bert_config.num_attention_heads = args.fusion_num_attention_heads
        self.fusion_bert_config.hidden_size = args.fusion_proj_size
        self.fusion_bert_config.vocab_size = 1
        self.fusion_bert = BertModel(self.fusion_bert_config)
        
        self.classifier = nn.Linear(args.fusion_proj_size, len(CATEGORY_ID_LIST))
        self.crit = nn.CrossEntropyLoss()
        # self.crit = FocalLoss(num_class=len(CATEGORY_ID_LIST))
        
    def forward(self, inputs, inference=False):
        # text embedding
        text_embedding = self.text_bert(input_ids=inputs['text_input'], attention_mask=inputs['text_mask']).last_hidden_state
        text_embedding = self.text_proj(text_embedding)
        
        # video embedding
        video_embedding = inputs['frame_input']
        video_embedding = self.video_proj(video_embedding)
        
        text_token_type_ids = torch.ones(text_embedding.size()[:-1], dtype=torch.long, device=text_embedding.device)
        video_token_type_ids = torch.ones(video_embedding.size()[:-1], dtype=torch.long, device=video_embedding.device)
        
        fusion_embedding = torch.cat([text_embedding, video_embedding], 1)
        fusion_mask = torch.cat([inputs['text_mask'], inputs['frame_mask']], 1)
        fusion_token_type_ids = torch.cat([text_token_type_ids, video_token_type_ids], -1)
        
        bert_embedding = self.fusion_bert(inputs_embeds=fusion_embedding, attention_mask=fusion_mask, token_type_ids=fusion_token_type_ids).last_hidden_state
        bert_embedding = torch.mean(bert_embedding, dim=1)
        prediction = self.classifier(bert_embedding)

        if inference:
            return prediction
        else:
            return self.cal_loss(prediction, inputs['label'], self.crit)

    @staticmethod
    def cal_loss(prediction, label, crit):
        label = label.squeeze(dim=1)
        loss = crit(prediction, label)
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label
    
    
class LXMERTMultiModal(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        # text_layer bert initalize
        self.text_bert = AutoModel.from_pretrained(args.bert_dir, cache_dir=args.bert_cache)
        self.text_proj = nn.Sequential(
            nn.Linear(self.text_bert.config.hidden_size, args.fusion_proj_size, bias=False),
            nn.Dropout(args.dropout),
            nn.LayerNorm(args.fusion_proj_size, eps=1e-6))
        
        # video_layer random initalize
        self.video_proj = nn.Sequential(
                    nn.Linear(args.frame_embedding_size, args.fusion_proj_size, bias=False),
                    nn.Dropout(args.dropout),
                    nn.LayerNorm(args.fusion_proj_size, eps=1e-6))
        
        # fusion bert random initialize
        self.fusion_config = LxmertConfig()
        self.fusion_config.x_layers = args.fusion_num_hidden_layers
        self.fusion_config.num_attention_heads = args.fusion_num_attention_heads
        self.fusion_config.hidden_size = args.fusion_proj_size
        self.fusion_model = LxmertModel(self.fusion_config).encoder.x_layers
        
        self.classifier = nn.Linear(args.fusion_proj_size, len(CATEGORY_ID_LIST))
        self.crit = nn.CrossEntropyLoss()
        # self.crit = FocalLoss(num_class=len(CATEGORY_ID_LIST))
        
    def forward(self, inputs, inference=False):
        # text embedding
        text_embedding = self.text_bert(input_ids=inputs['text_input'], attention_mask=inputs['text_mask']).last_hidden_state
        text_embedding = self.text_proj(text_embedding)
        
        # video embedding
        video_embedding = inputs['frame_input']
        video_embedding = self.video_proj(video_embedding)
        
        text_mask = inputs['text_mask'][:, None, None, :]
        text_mask = (1.0 - text_mask) * -10000.0
        
        video_mask = inputs['frame_mask'][:, None, None, :]
        video_mask = (1.0 - video_mask) * -10000.0

        for layer_module in self.fusion_model:
            x_outputs = layer_module(
                text_embedding,
                text_mask,
                video_embedding,
                video_mask
            )
            text_embedding, video_embedding = x_outputs[:2]
            
        bert_embedding = torch.cat([text_embedding, video_embedding], 1)
        bert_embedding = torch.mean(bert_embedding, dim=1)
        prediction = self.classifier(bert_embedding)

        if inference:
            return prediction
        else:
            return self.cal_loss(prediction, inputs['label'], self.crit)

    @staticmethod
    def cal_loss(prediction, label, crit):
        label = label.squeeze(dim=1)
        loss = crit(prediction, label)
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label
    
