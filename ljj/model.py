import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig
from transformers import VisualBertModel
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertEmbeddings, BertEncoder

from models.nezha.modeling_nezha import NeZhaForSequenceClassification, NeZhaModel
from models.nezha.configuration_nezha import NeZhaConfig

from models.nezha.visual_modeling_nezha import NeZhaModel as Visual_NeZhaModel
from models.bert.visual_modeling_bert import BertModel as Visual_BertModel
from models.bert.modeling_bert_vlbert import VLBertModel as VLBertModel
from models.bert.modeling_bert import UniBertModel

from category_id_map import CATEGORY_ID_LIST
from loss import FocalLoss, HMC
from module import NeXtVLAD, SENet, ConcatDenseSE
import os

class VisualBertMultiModal(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.config = BertConfig.from_pretrained(args.bert_dir)
        #self.bert = Visual_BertModel.from_pretrained(args.bert_dir, cache_dir=args.bert_cache)
        self.bert = Visual_BertModel(self.config)
        self.bert = self.rename_dict(args, self.bert)
        self.classifier = nn.Linear(args.bert_output_size, len(CATEGORY_ID_LIST))
        #self.ClassificationHead = ClassificationHead(args)
        if 'large' in args.bert_dir:
            args.bert_output_size = 1024
#         self.project_linear = nn.Sequential(
#                                 nn.Linear(args.frame_embedding_size, args.bert_output_size),
#                                 nn.ReLU(),
#                                 nn.LayerNorm(args.bert_output_size)
#                                 )
        self.FocalLoss = FocalLoss(num_class=len(CATEGORY_ID_LIST))
        self.CrossEntropy = nn.CrossEntropyLoss()
        
    def forward(self, inputs, inference=False):
        input_ids, attention_mask, token_type_ids = inputs['text_input'], inputs['text_mask'], inputs['text_token_type']
        visual_embeds, visual_attention_mask, visual_token_type_ids = inputs['frame_input'], inputs['frame_mask'], inputs['frame_token_type']
#         visual_embeds = self.project_linear(visual_embeds)
        
        bert_embedding = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            visual_embeds=visual_embeds,
                            visual_attention_mask=visual_attention_mask,
                            visual_token_type_ids=visual_token_type_ids,
                            output_hidden_states=True).last_hidden_state
        bert_embedding = torch.mean(bert_embedding, dim=1)
        prediction = self.classifier(bert_embedding)
        #prediction = self.ClassificationHead(bert_embedding)
        if inference:
            return torch.argmax(prediction, dim=1)
        else:
            return self.cal_loss(prediction, inputs['label'], self.CrossEntropy)
        
    def rename_dict(self, args, transformer):
        dropKey = ['roberta_mvm_lm_header.predictions.bias', 'roberta_mvm_lm_header.predictions.transform.dense.weight',\
                   'roberta_mvm_lm_header.predictions.transform.dense.bias', 'roberta_mvm_lm_header.predictions.transform.LayerNorm.weight', \
                   'roberta_mvm_lm_header.predictions.transform.LayerNorm.bias', 'roberta_mvm_lm_header.predictions.decoder.weight', \
                   'roberta_mvm_lm_header.predictions.decoder.bias', 'newfc_itm.weight', 'newfc_itm.bias', 'newfc_hidden.weight','newfc_hidden.bias',\
                    'classifier.weight','classifier.bias']
        model_dict = transformer.state_dict()

        checkpoint_dict = {}
        model_loaded = torch.load(os.path.join(args.bert_dir, 'pytorch_model.bin'))['model_state_dict']
        for key, value in model_loaded.items():
            if key in dropKey:
                break
            if key not in dropKey:
                checkpoint_dict[self.rename_key(key)] = value
        model_dict.update(checkpoint_dict)
        transformer.load_state_dict(model_dict, strict=False)
        return transformer
                                  
    def rename_key(self, key):
        if 'bert' in key:
            return  key.split('bert.')[-1]
    @staticmethod
    def cal_loss(prediction, label, crit):
        label = label.squeeze(dim=1)
        loss = crit(prediction, label)
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label
        
# class VisualBertMultiModal(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         self.config = BertConfig.from_pretrained(args.bert_dir)
#         self.bert = Visual_BertModel.from_pretrained(args.bert_dir, cache_dir=args.bert_cache)
#         #self.bert = Visual_BertModel(self.config)
#         #self.bert = self.rename_dict(args, self.bert)
#         self.classifier = nn.Linear(args.bert_output_size, len(CATEGORY_ID_LIST))
#         #self.ClassificationHead = ClassificationHead(args)
#         if 'large' in args.bert_dir:
#             args.bert_output_size = 1024
# #         self.project_linear = nn.Sequential(
# #                                 nn.Linear(args.frame_embedding_size, args.bert_output_size),
# #                                 nn.ReLU(),
# #                                 nn.LayerNorm(args.bert_output_size)
# #                                 )
#         self.FocalLoss = FocalLoss(num_class=len(CATEGORY_ID_LIST))
#         self.CrossEntropy = nn.CrossEntropyLoss()
        
#     def forward(self, inputs, inference=False):
#         input_ids, attention_mask, token_type_ids = inputs['text_input'], inputs['text_mask'], inputs['text_token_type']
#         visual_embeds, visual_attention_mask, visual_token_type_ids = inputs['frame_input'], inputs['frame_mask'], inputs['frame_token_type']
# #         visual_embeds = self.project_linear(visual_embeds)
        
#         bert_embedding = self.bert(input_ids=input_ids,
#                             attention_mask=attention_mask,
#                             token_type_ids=token_type_ids,
#                             visual_embeds=visual_embeds,
#                             visual_attention_mask=visual_attention_mask,
#                             visual_token_type_ids=visual_token_type_ids,
#                             output_hidden_states=True).last_hidden_state
#         bert_embedding = torch.mean(bert_embedding, dim=1)
#         prediction = self.classifier(bert_embedding)
#         #prediction = self.ClassificationHead(bert_embedding)
#         if inference:
#             return torch.argmax(prediction, dim=1)
#         else:
#             return self.cal_loss(prediction, inputs['label'], self.CrossEntropy)
        
#     @staticmethod
#     def cal_loss(prediction, label, crit):
#         label = label.squeeze(dim=1)
#         loss = crit(prediction, label)
#         with torch.no_grad():
#             pred_label_id = torch.argmax(prediction, dim=1)
#             accuracy = (label == pred_label_id).float().sum() / label.shape[0]
#         return loss, accuracy, pred_label_id, label
    

class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, args):
        super().__init__()
        self.norm= nn.BatchNorm1d(args.bert_output_size)
        self.dense = nn.Linear(args.bert_output_size, 512)
        self.norm_1= nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.2)
        self.dense_1 = nn.Linear(512, 256)  
        self.norm_2= nn.BatchNorm1d(256)
        self.out_proj = nn.Linear(256, 200)

    def forward(self, features, **kwargs):
        x = self.norm(features)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.relu(self.norm_1(x))
        x = self.dropout(x)
        x = self.dense_1(x)
        x = torch.relu(self.norm_2(x))
        x = self.dropout(x)        
        x = self.out_proj(x)
        return x
    
class VlBertMultiModal(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.bert = VLBertModel.from_pretrained(args.bert_dir, cache_dir=args.bert_cache)
        self.classifier = nn.Linear(args.bert_output_size, len(CATEGORY_ID_LIST))
        if 'large' in args.bert_dir:
            args.bert_output_size = 1024
        self.project_linear = nn.Sequential(
                                nn.Linear(args.frame_embedding_size, args.bert_output_size),
                                #nn.ReLU(),
                                nn.LayerNorm(args.bert_output_size)
                                )
        self.FocalLoss = FocalLoss(num_class=len(CATEGORY_ID_LIST))
        
    def forward(self, inputs, inference=False):
        input_ids, attention_mask, token_type_ids = inputs['text_input'], inputs['text_mask'], inputs['text_token_type']
        visual_embeds, visual_attention_mask, visual_token_type_ids = inputs['frame_input'], inputs['frame_mask'], inputs['frame_token_type']
        visual_embeds = self.project_linear(visual_embeds)
        
        bert_embedding = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            visual_embeds=visual_embeds,
                            visual_attention_mask=visual_attention_mask,
                            visual_token_type_ids=visual_token_type_ids,
                            output_hidden_states=True).last_hidden_state
        bert_embedding = torch.mean(bert_embedding, dim=1)
        prediction = self.classifier(bert_embedding)

        if inference:
            return torch.argmax(prediction, dim=1)
        else:
            return self.cal_loss(prediction, inputs['label'], self.FocalLoss)

    @staticmethod
    def cal_loss(prediction, label, crit):
        label = label.squeeze(dim=1)
        loss = crit(prediction, label)
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label

class VisualBertNeXtVLADMultiModal(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.bert = Visual_BertModel.from_pretrained(args.bert_dir, cache_dir=args.bert_cache)
        
        if 'large' in args.bert_dir:
            args.bert_output_size = 1024
        self.nextvlad = NeXtVLAD(args.frame_embedding_size, args.vlad_cluster_size, output_size=args.vlad_hidden_size, dropout=args.dropout)
        self.project_linear = nn.Sequential(
                                nn.Linear(args.frame_embedding_size, args.bert_output_size),
                                nn.LayerNorm(args.bert_output_size)
                                )
        self.classifier = nn.Linear(args.bert_output_size + args.vlad_hidden_size, len(CATEGORY_ID_LIST))
        self.FocalLoss = FocalLoss(num_class=len(CATEGORY_ID_LIST))
        
    def forward(self, inputs, inference=False):
        input_ids, attention_mask, token_type_ids = inputs['text_input'], inputs['text_mask'], inputs['text_token_type']
        visual_embeds, visual_attention_mask, visual_token_type_ids = inputs['frame_input'], inputs['frame_mask'], inputs['frame_token_type']
        
        video_embedding = self.nextvlad(visual_embeds, visual_attention_mask)
        
        visual_embeds = self.project_linear(visual_embeds)
        
        bert_embedding = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            visual_embeds=visual_embeds,
                            visual_attention_mask=visual_attention_mask,
                            visual_token_type_ids=visual_token_type_ids,
                            output_hidden_states=True).last_hidden_state
        bert_embedding = torch.mean(bert_embedding, dim=1)
        
        final_embedding = torch.cat([video_embedding, bert_embedding], -1)
        prediction = self.classifier(final_embedding)

        if inference:
            return torch.argmax(prediction, dim=1)
        else:
            return self.cal_loss(prediction, inputs['label'], self.FocalLoss)

    @staticmethod
    def cal_loss(prediction, label, crit):
        label = label.squeeze(dim=1)
        loss = crit(prediction, label)
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label
    
class UniBertMultiModal(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.bert = UniBertModel.from_pretrained(args.bert_dir, cache_dir=args.bert_cache)
        self.classifier = nn.Linear(args.bert_output_size, len(CATEGORY_ID_LIST))
        self.FocalLoss = FocalLoss(num_class=len(CATEGORY_ID_LIST))
        
    def forward(self, inputs, inference=False):
        input_ids, attention_mask = inputs['text_input'], inputs['text_mask']
        visual_embeds, visual_attention_mask = inputs['frame_input'], inputs['frame_mask']
        bert_embedding = self.bert(video_feature=visual_embeds, video_mask=visual_attention_mask, text_input_ids=input_ids, text_mask=attention_mask)
        bert_embedding = torch.mean(bert_embedding, dim=1)
        prediction = self.classifier(bert_embedding)

        if inference:
            return torch.argmax(prediction, dim=1)
        else:
            return self.cal_loss(prediction, inputs['label'], self.FocalLoss)

    @staticmethod
    def cal_loss(prediction, label, crit):
        label = label.squeeze(dim=1)
        loss = crit(prediction, label)
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label
    
class UniBert(BertPreTrainedModel):
    def __init__(self, args):
        self.config = BertConfig.from_pretrained(args.bert_dir)
        super().__init__(self.config)
        

        self.embeddings = BertEmbeddings(self.config)
        self.video_fc = nn.Sequential(
                                nn.Linear(args.frame_embedding_size, args.bert_output_size),
                                nn.LayerNorm(args.bert_output_size)
                                )
        self.video_embeddings = BertEmbeddings(self.config)
        self.encoder = BertEncoder(self.config)
        self.classifier = nn.Linear(args.bert_output_size, len(CATEGORY_ID_LIST))
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    def forward(self, inputs, inference=False):        
        input_ids, attention_mask = inputs['text_input'], inputs['text_mask']
        visual_embeds, visual_attention_mask = inputs['frame_input'], inputs['frame_mask']
        text_emb = self.embeddings(input_ids)
        text_mask = inputs['text_mask']
        
        # text input is [CLS][SEP] t e x t [SEP]
        cls_emb = text_emb[:, 0:1, :]
        text_emb = text_emb[:, 1:, :]
        
        cls_mask = text_mask[:, 0:1]
        text_mask = text_mask[:, 1:]
        
        # reduce frame feature dimensions 
        video_feature = self.video_fc(visual_embeds)
        video_emb = self.video_embeddings(visual_embeds)
 
        # [CLS] Video [SEP] Text [SEP]
        embedding_output = torch.cat([cls_emb, video_emb, text_emb], 1)
        
        mask = torch.cat([cls_mask, visual_attention_mask, text_mask], 1)
        mask = mask[:, None, None, :]
        mask = (1.0 - mask) * -10000.0
        
        encoder_outputs = self.encoder(embedding_output, attention_mask=mask)['last_hidden_state']
        bert_embedding = torch.mean(encoder_outputs, dim=1)
        prediction = self.classifier(bert_embedding)

        if inference:
            return torch.argmax(prediction, dim=1)
        else:
            return self.cal_loss(prediction, inputs['label'], self.FocalLoss)

    @staticmethod
    def cal_loss(prediction, label, crit):
        label = label.squeeze(dim=1)
        loss = crit(prediction, label)
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label

class UniBert_1(BertPreTrainedModel):
    def __init__(self, args, config):
        super().__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.video_fc = nn.Sequential(
                                nn.Linear(args.frame_embedding_size, args.bert_output_size),
                                nn.LayerNorm(args.bert_output_size)
                                )
        self.video_embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.classifier = nn.Linear(args.bert_output_size, len(CATEGORY_ID_LIST))
        self.FocalLoss = FocalLoss(num_class=len(CATEGORY_ID_LIST))
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    def forward(self, inputs, inference=False):        
        input_ids, attention_mask = inputs['text_input'], inputs['text_mask']
        visual_embeds, visual_attention_mask = inputs['frame_input'], inputs['frame_mask']
        text_emb = self.embeddings(input_ids)
        text_mask = inputs['text_mask']
        
        # text input is [CLS][SEP] t e x t [SEP]
        cls_emb = text_emb[:, 0:1, :]
        text_emb = text_emb[:, 1:, :]
        
        cls_mask = text_mask[:, 0:1]
        text_mask = text_mask[:, 1:]
        
        # reduce frame feature dimensions 
        video_feature = self.video_fc(visual_embeds)
        video_emb = self.video_embeddings(inputs_embeds=visual_embeds)
 
        # [CLS] Video [SEP] Text [SEP]
        embedding_output = torch.cat([cls_emb, video_emb, text_emb], 1)
        
        mask = torch.cat([cls_mask, visual_attention_mask, text_mask], 1)
        mask = mask[:, None, None, :]
        mask = (1.0 - mask) * -10000.0
        
        encoder_outputs = self.encoder(embedding_output, attention_mask=mask)['last_hidden_state']
        bert_embedding = torch.mean(encoder_outputs, dim=1)
        prediction = self.classifier(bert_embedding)

        if inference:
            return torch.argmax(prediction, dim=1)
        else:
            return self.cal_loss(prediction, inputs['label'], self.FocalLoss)

    @staticmethod
    def cal_loss(prediction, label, crit):
        label = label.squeeze(dim=1)
        loss = crit(prediction, label)
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label
    
class VisualBertMultiModal_HMC(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.L12_table = [[0, 1, 2, 3, 4, 5, 6, 7, 8],
                    [9, 10, 11, 12, 13],
                    [14, 15, 16, 17, 18, 19, 20, 21, 22],
                    [23, 24, 25, 26, 27, 28],
                    [29, 30, 31, 32, 33, 34],
                    [35, 36, 37],
                    [38, 39, 40],
                    [41, 42, 43, 44, 45, 46],
                    [47, 48, 49, 50, 51, 52],
                    [53, 54, 55, 56, 57, 58, 59, 60],
                    [61, 62, 63, 64],
                    [65, 66, 67, 68, 69, 70],
                    [71, 72, 73, 74, 75, 76],
                    [77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88],
                    [89, 90, 91, 92],
                    [93, 94, 95, 96, 97, 98, 99, 100, 101, 102],
                    [103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113],
                    [114, 115, 116, 117, 118, 119, 120, 121, 122],
                    [123, 124, 125, 126, 127, 128, 129],
                    [130, 131, 132, 133, 134, 135],
                    [136, 137, 138, 139, 140,141,142,143,144,145,146,147,148,149,150,151],
                    [152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175],
                    [176,177, 178,179, 180,181,182,183,184, 185,186,187,188,189,190,191,192,193,194,195,196,197,198,199]]
        self.hmc_lambda = args.hmc_lambda
        self.hmc_beta = args.hmc_beta
        self.bert = Visual_BertModel.from_pretrained(args.bert_dir, cache_dir=args.bert_cache)
        self.classifier = nn.Linear(args.bert_output_size, len(CATEGORY_ID_LIST))
        self.hmc_classifier = HMC(args.bert_output_size, 23, 200, self.L12_table, 512, mask_value=-100)
        #self.hmc_loss = hmc_loss()
        if 'large' in args.bert_dir:
            args.bert_output_size = 1024
        self.project_linear = nn.Sequential(
                                nn.Linear(args.frame_embedding_size, args.bert_output_size),
                                nn.LayerNorm(args.bert_output_size)
                                )
        self.FocalLoss = FocalLoss(num_class=len(CATEGORY_ID_LIST))
        
    def forward(self, inputs, inference=False):
        input_ids, attention_mask, token_type_ids = inputs['text_input'], inputs['text_mask'], inputs['text_token_type']
        visual_embeds, visual_attention_mask, visual_token_type_ids = inputs['frame_input'], inputs['frame_mask'], inputs['frame_token_type']
        visual_embeds = self.project_linear(visual_embeds)
        
        bert_embedding = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            visual_embeds=visual_embeds,
                            visual_attention_mask=visual_attention_mask,
                            visual_token_type_ids=visual_token_type_ids,
                            output_hidden_states=True).last_hidden_state
        bert_embedding = torch.mean(bert_embedding, dim=1)
        L1, L2 = self.hmc_classifier(bert_embedding)
        #prediction = self.classifier(bert_embedding)

        if inference:
            L1_pred = torch.argmax(L1, dim=1)
            L2_pred = torch.argmax(L2, dim=1)
            return L1_pred, L2_pred
        else:
            loss = hmc_loss(L1, L2, inputs['l1_label'].squeeze(dim=1), inputs['l2_label'].squeeze(dim=1), self.hmc_lambda, self.hmc_beta)
            with torch.no_grad():
                l1_label = inputs['l1_label'].squeeze(dim=1)
                l2_label = inputs['l2_label'].squeeze(dim=1)
                pred_l1_label_id = torch.argmax(L1, dim=1)
                pred_l2_label_id = torch.argmax(L2, dim=1)
                l1_accuracy = (l1_label == pred_l1_label_id).float().sum() / l1_label.shape[0]
                l2_accuracy = (l2_label == pred_l2_label_id).float().sum() / l2_label.shape[0]
                
            return loss, l1_accuracy, l2_accuracy, l1_label, l2_label, pred_l1_label_id, pred_l2_label_id

#     @staticmethod
#     def cal_loss(prediction, label, crit):
#         label = label.squeeze(dim=1)
#         loss = crit(prediction, label)
#         with torch.no_grad():
#             pred_label_id = torch.argmax(prediction, dim=1)
#             accuracy = (label == pred_label_id).float().sum() / label.shape[0]
#         return loss, accuracy, pred_label_id, label
    

def hmc_loss(L1, L2, L1_gt, L2_gt, Lambda=None, Beta=None):
    """calculate hmc loss
    Args:
        L1: Tensor(batch*L1_labels_num) -> network output of level 1
        L2: Tensor(batch*L2_labels_num) -> network output of level 2
        L1_gt: Tensor(batch,) dtype=int -> ground truth of level 1
        L2_gt: Tensor(batch,) dtype=int -> ground truth of level 2
        Lambda: a float coefficient for L2 loss pernelty
        Beta: a float coefficient for L2-L1 loss pernelty
    """


    batch_num = L1.shape[0]
    Y1 = L1[torch.arange(batch_num), L1_gt]
    Y2 = L2[torch.arange(batch_num), L2_gt] + 1e-8
    #print(Y2)
    L1_loss = - Y1.log().mean()
    L2_loss = - Y2.log().mean()
    
    LH_loss = torch.max(Y2-Y1,torch.zeros_like(Y1)).mean()
    #print(L2_loss)
    return L1_loss + Lambda * L2_loss + Beta * LH_loss 
