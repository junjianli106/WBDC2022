import torch
from torch.utils.data import SequentialSampler, DataLoader

from config import parse_args
from data_helper import MultiModalDataset
from category_id_map import lv2id_to_category_id
from model import MultiModal
from visualbert_model import MyVisualBert
from tqdm import tqdm
import json

def inference():
    args = parse_args()
    # 1. load data
    dataset = MultiModalDataset(args, args.test_annotation, args.test_zip_feats, test_mode=True)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
                            batch_size=args.test_batch_size,
                            sampler=sampler,
                            drop_last=False,
                            pin_memory=True,
                            num_workers=args.num_workers,
                            prefetch_factor=args.prefetch)

    # 2. load model
    # model = MultiModal(args)
    model = MyVisualBert(args)
    checkpoint = torch.load(args.ckpt_file, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if torch.cuda.is_available():
        model = torch.nn.parallel.DataParallel(model.cuda())
    model.eval()

    # 3. inference
    predictions = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            prediction = model(batch, inference=True)
            if args.outraw:
                predictions.extend(prediction.tolist())
            else:
                pred_label_id = torch.argmax(prediction, dim=-1)
                predictions.extend(pred_label_id.cpu().numpy())

    # 4. dump results
    with open(args.test_output_csv, 'w') as f:
        for pred, ann in zip(predictions, dataset.anns):
            video_id = ann['id']
            if args.outraw:
                x = {'id': video_id, 'logit': pred}
                json.dump(x, f)
                f.write('\n')
            else:
                category_id = lv2id_to_category_id(pred)
                f.write(f'{video_id},{category_id}\n')


if __name__ == '__main__':
    inference()
