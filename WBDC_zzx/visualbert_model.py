import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, AutoModel, VisualBertConfig, VisualBertModel

from category_id_map import CATEGORY_ID_LIST


class MyVisualBert(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model = AutoModel.from_pretrained(args.bert_dir)
        print(type(self.model))
        config = self.model.config
        self.visual_proj = nn.Linear(768, config.hidden_size)
        # self.activate = nn.LeakyReLU()
        # self.activate = nn.Tanh()
        # self.activate = nn.ReLU()
        # self.activate = nn.GELU()
        self.args = args
        self.pool_dropout = nn.Dropout(p=args.pool_dropout)
        self.classifier = nn.Linear(config.hidden_size, len(CATEGORY_ID_LIST))
        # self.classifier = nn.Linear(2 * config.hidden_size, len(CATEGORY_ID_LIST))

        
    def forward(self, inputs, inference=False):
        
        input_ids = inputs['title_input']
        attention_mask = inputs['title_mask']
        visual_embeds = inputs['frame_input']
        visual_attention_mask = inputs['frame_mask']
        
        emb_layer = self.model.get_input_embeddings()
        word_emb = emb_layer(input_ids)
        img_emb = self.visual_proj(visual_embeds)
        # img_emb = self.activate(img_emb)
        inputs_embeds = torch.cat([word_emb, img_emb], dim=1)
        
        mask = torch.cat([attention_mask, visual_attention_mask], dim=1)
        # token_type_ids0 = torch.zeros(word_emb.shape[:-1], dtype=torch.long, device=word_emb.device)
        # token_type_ids1 = torch.ones(img_emb.shape[:-1], dtype=torch.long, device=word_emb.device)
        
        # token_type_ids = torch.cat([token_type_ids0, token_type_ids1], dim=-1)

        # output = self.model(input_ids=input_ids, attention_mask=attention_mask, visual_embeds=visual_embeds, visual_attention_mask=visual_attention_mask)
        output = self.model(inputs_embeds=inputs_embeds, attention_mask=mask)
        
        
        # word_emb = self.model.embeddings(input_ids=input_ids)
        # img_emb = self.visual_proj(visual_embeds)
        # # img_emb = self.activate(img_emb)
        # img_emb = self.model.embeddings(inputs_embeds=img_emb)
        # inputs_embeds = torch.cat([word_emb, img_emb], dim=1)
        # mask = torch.cat([attention_mask, visual_attention_mask], dim=1)
        # # print(inputs_embeds.size(), mask.size())
        # mask = mask[:, None, None, :]
        # mask = (1.0 - mask) * -10000.0
        # output = self.model.encoder(inputs_embeds, attention_mask=mask)
        # mask = torch.cat([attention_mask, visual_attention_mask], dim=1)
        
        if self.args.pool == 'cls':
            final_embedding = output.last_hidden_state[:, 0, :]
        
        elif self.args.pool == 'mean':
            last_hidden_state = output.last_hidden_state
            input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            final_embedding = sum_embeddings / sum_mask
            
        # mean_embedding = torch.mean(output.last_hidden_state, dim=1)
        # final_embedding = mean_embedding
        # max_embedding = torch.max(output.last_hidden_state, dim=1)[0]
        # final_embedding = torch.cat([mean_embedding, max_embedding], dim=-1)
        # mask_hids = output.last_hidden_state
        # mask_hids.masked_fill_((1-mask.unsqueeze(-1)).bool(), 0)
        # sum_hids = torch.sum(mask_hids, dim=1)
        # sum_attn = torch.sum(mask, dim=-1)
        # final_embedding = sum_hids / sum_attn.unsqueeze(-1)
        
        final_embedding = self.pool_dropout(final_embedding)
        prediction = self.classifier(final_embedding)

        if inference:
            # return torch.argmax(prediction, dim=1)
            return prediction
        else:
            return self.cal_loss(prediction, inputs['label'])

    @staticmethod
    def cal_loss(prediction, label):
        label = label.squeeze(dim=1)
        loss = F.cross_entropy(prediction, label)
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label


class VisualBert(nn.Module):
    def __init__(self, args):
        super().__init__()
        model = AutoModel.from_pretrained(args.bert_dir)
        config = model.config

        vconfig = VisualBertConfig(vocab_size=config.vocab_size, visual_embedding_dim=768, pad_token_id=config.pad_token_id, hidden_size=config.hidden_size, intermediate_size=config.intermediate_size,
                                   num_hidden_layers=config.num_hidden_layers, num_attention_heads=config.num_attention_heads)
        print(config, vconfig)
        # copy_attrs = ['hidden_size', 'intermediate_size', 'num_attention_heads', 'num_hidden_layers', ]
        assert vconfig.hidden_size == config.hidden_size

        vmodel = VisualBertModel(vconfig)
        missing_keys, unexpected_keys = vmodel.load_state_dict(
            model.state_dict(), strict=False)
        # 这样的话不会从文字的position embedding等 copy到图片，可以显式调用VisualBertEmbeddings里面的语句，不过感觉可能影响不大，暂时不做
        print(
            f'missing_keys:{missing_keys}, unexpected_keys:{unexpected_keys}')

        self.model = vmodel
        self.classifier = nn.Linear(config.hidden_size, len(CATEGORY_ID_LIST))
        
    def forward(self, inputs, inference=False):
        
        input_ids = inputs['title_input']
        attention_mask = inputs['title_mask']
        visual_embeds = inputs['frame_input']
        visual_attention_mask = inputs['frame_mask']
        
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, visual_embeds=visual_embeds, visual_attention_mask=visual_attention_mask)
        
        final_embedding = output.pooler_output
        
        prediction = self.classifier(final_embedding)

        if inference:
            return torch.argmax(prediction, dim=1)
        else:
            return self.cal_loss(prediction, inputs['label'])

    @staticmethod
    def cal_loss(prediction, label):
        label = label.squeeze(dim=1)
        loss = F.cross_entropy(prediction, label)
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label
