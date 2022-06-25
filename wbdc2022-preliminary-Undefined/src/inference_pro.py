import torch
from torch.utils.data import SequentialSampler, DataLoader

from config import parse_args
from data_helper import MultiModalDataset
from category_id_map import lv2id_to_category_id
from model import WCUniModel
from tqdm import tqdm
import os
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast as autocast
import json

def inference():
    args = parse_args()
    # 1. load data
    with open(args.test_annotation, 'r', encoding='utf8') as f:
        anns = json.load(f)
    dataset = MultiModalDataset(args, anns, args.test_zip_feats, test_mode=True)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
                            batch_size=args.test_batch_size,
                            sampler=sampler,
                            drop_last=False,
                            pin_memory=True,
                            num_workers=args.num_workers,
                            prefetch_factor=args.prefetch)

  # 2. load model
    models = []
    for i in range(10):
        model = WCUniModel(args, task=['cls'])
        checkpoint = torch.load(os.path.join(args.savedmodel_path, f'Unibert_f1_fold_{i}.bin'), map_location='cpu')
        print(checkpoint['mean_f1'])
        model.load_state_dict(checkpoint['model_state_dict'])
        if torch.cuda.is_available():
            model = torch.nn.parallel.DataParallel(model.cuda())
        model.eval()
        models.append(model)
    
    args.savedmodel_path = '/homeb/junjianli/competition/WBDC2022/wbdc2022_wc/save/Unibert/baseline_fgm1.5_ema_pretreain_10'
    for i in range(10):
        model = WCUniModel(args, task=['cls'])
        checkpoint = torch.load(os.path.join(args.savedmodel_path, f'Unibert_f1_fold_{i}.bin'), map_location='cpu')
        print(checkpoint['mean_f1'])
        model.load_state_dict(checkpoint['model_state_dict'])
        if torch.cuda.is_available():
            model = torch.nn.parallel.DataParallel(model.cuda())
        model.eval()
        models.append(model)
    
    
    # 3. inference
    predictions = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids, attention_mask = batch['text_input'], batch['text_mask']
            visual_embeds, visual_attention_mask = batch['frame_input'], batch['frame_mask']
            for idx, model in enumerate(models):
                if idx == 0:
                    prediction = model(visual_embeds, visual_attention_mask, input_ids, attention_mask, inference=True)
                else:
                    prediction += model(visual_embeds, visual_attention_mask, input_ids, attention_mask, inference=True)
            pred_label_id = torch.argmax(prediction, dim=1)
            predictions.extend(pred_label_id.cpu().numpy())

    # 4. dump results
    with open(args.test_output_csv, 'w') as f:
        for pred_label_id, ann in zip(predictions, dataset.anns):
            video_id = ann['id']
            category_id = lv2id_to_category_id(pred_label_id)
            f.write(f'{video_id},{category_id}\n')

if __name__ == '__main__':
    inference()
