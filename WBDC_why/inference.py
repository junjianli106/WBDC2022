import torch
import json
from tqdm import tqdm
from torch.utils.data import SequentialSampler, DataLoader
import pandas as pd

from config import parse_args
from data_helper import MultiModalDataset
from category_id_map import lv2id_to_category_id
from model import UniBertMultiModal
from model import DualBertMultiModal
from model import LXMERTMultiModal
from model import ALBEFMultiModal

import os

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
                            num_workers=args.num_workers)

    # 2. load model
    models = []
    for i in range(5):
        if args.model_type == "unibert":
            model = UniBertMultiModal(args)
        elif args.model_type == "albef":
            model = ALBEFMultiModal(args)
        elif args.model_type == "dualbert":
            model = DualBertMultiModal(args)
        elif args.model_type == "lxmert":
            model = LXMERTMultiModal(args)
        checkpoint = torch.load(f'{args.savedmodel_path}/{args.ckpt_file}_fold{i}.bin', map_location='cpu')
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
            for idx, model in enumerate(models):
                if idx == 0:
                    prediction = model(batch, inference=True)
                else:
                    prediction += model(batch, inference=True)
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
