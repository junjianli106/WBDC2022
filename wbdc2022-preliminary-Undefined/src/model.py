import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertEmbeddings, BertEncoder
from transformers.models.bert.modeling_bert import BertConfig, BertOnlyMLMHead
from masklm import MaskLM, MaskVideo, ShuffleVideo
from category_id_map import CATEGORY_ID_LIST
import os

class UniBertForMaskedLM(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = UniBert(config)
        self.cls = BertOnlyMLMHead(config)
        
    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    def forward(self, frame_input, frame_mask, text_input_ids, text_mask, gather_index=None, return_mlm=False):
        encoder_outputs = self.bert(frame_input, frame_mask, text_input_ids, text_mask)
        if return_mlm:
            return encoder_outputs, self.cls(encoder_outputs)[:, 1 + frame_input.size()[1]: , :]
        else:
            return encoder_outputs, None  
        
        
# 没有加载权重
class UniBert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.video_fc = nn.Sequential(
                                #nn.Linear(args.frame_embedding_size, args.bert_output_size),
                                nn.Linear(768, 768),
                                nn.LayerNorm(768)
                                )
        self.video_embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    def forward(self, visual_embeds, visual_attention_mask, input_ids, attention_mask, inference=False):        
        
        text_emb = self.embeddings(input_ids)
        text_mask = attention_mask
        
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
        
        return encoder_outputs
    
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
        self.decoder = nn.Linear(config.hidden_size, 1536, bias=False)
        self.bias = nn.Parameter(torch.zeros(1536))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states
    
class VisualOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = VisualLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores
    
class WCUniModel(nn.Module):
    def __init__(self, args, task=['mlm', 'mfm', 'itm', 'cls'], init_from_pretrain=True):
        super().__init__()
        uni_bert_cfg = BertConfig.from_pretrained(args.bert_dir)
        #uni_bert_cfg.num_hidden_layers = 1
        self.crit = nn.CrossEntropyLoss()
        self.classifier = nn.Linear(args.bert_output_size, len(CATEGORY_ID_LIST))
        self.newfc_hidden = torch.nn.Linear(uni_bert_cfg.hidden_size, 768)
        
        self.task = set(task)
        if 'cls' in task:
            #self.newfc_tag = torch.nn.Linear(cfg['HIDDEN_SIZE'], cfg['NUM_CLASSES'])
            self.newfc_cls = nn.Linear(args.bert_output_size, len(CATEGORY_ID_LIST))
        
        if 'mlm' in task:
            self.lm = MaskLM(tokenizer_path=args.bert_dir)
            self.num_class = len(CATEGORY_ID_LIST)
            self.vocab_size = uni_bert_cfg.vocab_size
        
        if 'mfm' in task:
            self.vm = MaskVideo()
            self.roberta_mvm_lm_header = VisualOnlyMLMHead(uni_bert_cfg) 
            
        if 'itm' in task:
            self.sv = ShuffleVideo()
            self.newfc_itm = torch.nn.Linear(uni_bert_cfg.hidden_size, 1) 

        if init_from_pretrain:
            #self.roberta = UniBertForMaskedLM(uni_bert_cfg).from_pretrained(args.bert_dir, config=uni_bert_cfg)
            self.roberta = UniBertForMaskedLM(uni_bert_cfg) #.from_pretrained(args.bert_dir, config=uni_bert_cfg)
            self.roberta = self.rename_dict(args, self.roberta)
        else:
            self.roberta = UniBertForMaskedLM(uni_bert_cfg)
    
    def forward(self, video_feature, video_mask, text_input_ids, text_mask, target=None, task=None, inference=False):
        loss, pred = 0, None
        
        if task is None:
            sample_task = self.task
        elif type(task) == str:
            sample_task = [task]
        elif type(task) == list:
            sample_task = task
        
        # perpare for pretrain task [mask and get task label]: {'mlm': mask title token} {'mfm': mask frame} {'itm': shuffle title-video}
        return_mlm = False
        if 'mlm' in sample_task:
            input_ids, lm_label = self.lm.torch_mask_tokens(text_input_ids.cpu())
            text_input_ids = input_ids.to(text_input_ids.device)
            lm_label = lm_label[:, 1:].to(text_input_ids.device)
            return_mlm = True
            
        if 'mfm' in sample_task:
            vm_input = video_feature
            input_feature, video_label = self.vm.torch_mask_frames(video_feature.cpu(), video_mask.cpu())
            video_feature = input_feature.to(video_feature.device)
            video_label = video_label.to(video_feature.device)
            
        if 'itm' in sample_task:
            input_feature, video_text_match_label = self.sv.torch_shuf_video(video_feature.cpu())
            video_feature = input_feature.to(video_feature.device)
            video_text_match_label = video_text_match_label.to(video_feature.device)
            
        # concat features
        features, lm_prediction_scores = self.roberta(video_feature, video_mask, text_input_ids, text_mask, return_mlm=return_mlm)
        features_mean = torch.mean(features, 1)
        embedding = self.newfc_hidden(features_mean)
        
        # compute pretrain task loss
        if 'mlm' in sample_task:
            pred = lm_prediction_scores.contiguous().view(-1, self.vocab_size)
            masked_lm_loss = nn.CrossEntropyLoss()(pred, lm_label.contiguous().view(-1))
            loss += masked_lm_loss  / len(sample_task)
            
        if 'mfm' in sample_task:
            vm_output = self.roberta_mvm_lm_header(features[:, 1:video_feature.size()[1] + 1, :])
            masked_vm_loss = self.calculate_mfm_loss(vm_output, vm_input, 
                                                     video_mask, video_label, normalize=False)
            loss += masked_vm_loss  / len(sample_task)
            
        if 'itm' in sample_task:
            pred = self.newfc_itm(features[:, 0, :])
            itm_loss = nn.BCEWithLogitsLoss()(pred.view(-1), video_text_match_label.view(-1))
            loss += itm_loss  / len(sample_task)
        
        if 'cls' in sample_task:
            prediction = self.classifier(features_mean)
            
            if inference:
                return prediction
                #return torch.argmax(prediction, dim=1)
            else:
                return self.cal_loss(prediction, target, self.crit)
        else:
            if 'itm' in sample_task:
                (loss, masked_lm_loss, itm_loss)
            else:
                return (loss, masked_lm_loss)
        
    def rename_dict(self, args, roberta):
        model_dict = roberta.state_dict()

        checkpoint_dict = {}
        model_loaded = torch.load(os.path.join(args.bert_dir, 'pytorch_model.bin'))['model_state_dict']
        for key, value in model_loaded.items():
            key = key.split('roberta.')[-1]
            checkpoint_dict[key] = value
        model_dict.update(checkpoint_dict)
        missing_keys, unexpected_keys = roberta.load_state_dict(model_dict, strict=False)
        print(f'missing keys:{missing_keys}, unexpected_keys:{unexpected_keys}')
        return roberta
    
    @staticmethod
    def cal_loss(prediction, label, crit):
        label = label.squeeze(dim=1)
        loss = crit(prediction, label)
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label