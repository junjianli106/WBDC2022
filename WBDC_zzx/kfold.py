import numpy as np

from sklearn.model_selection import StratifiedKFold
import json
import os

def kfold_split():
    path = './data/annotations/labeled.json'
    out_dir = './data/annotations/10fold_seed2022'
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    with open(path, 'r') as f:
        data = json.load(f)
    
    ys = []
    for d in data:
        ys.append(d['category_id'])
        
    skf = StratifiedKFold(n_splits=10, random_state=2022, shuffle=True)
    
    for k, (train_idx, val_idx) in enumerate(skf.split(data, ys)):
        print(len(train_idx), len(val_idx))
        train_data = np.array(data)[train_idx].tolist()
        val_data = np.array(data)[val_idx].tolist()
        train_path = f'{out_dir}/train{k}.json'
        val_path = f'{out_dir}/val{k}.json'
        with open(train_path, 'w') as outf:
            json.dump(train_data, outf, ensure_ascii=False, indent=1)
        
        with open(val_path, 'w') as outf:
            json.dump(val_data, outf, ensure_ascii=False, indent=1)
            

def kfold_predict():
    ckpt_dir = '/data/zhangzhexin/weixin_2022/challenge/save/chinese-roberta-wwm-ext/pretrainbs128_len128_epoch14/10fold_seed2022'
    k = 5
    # lists = [0, 1, 2, 5, 6, 7]
    lists = [3, 4, 8, 9]
    for i in lists:
        dir = os.path.join(ckpt_dir, str(i))
        bins = os.listdir(dir)
        bins.sort()
        bin = bins[-1]
        path = os.path.join(dir, bin)

        os.system(f'MKL_SERVICE_FORCE_INTEL=1 CUDA_VISIBLE_DEVICES=2 python inference.py --ckpt_file={path} \
    --bert_dir=hfl/chinese-roberta-wwm-ext --bert_seq_length=384 --test_output_csv=./result/10fold_result{i}.csv --pool=cls --outraw=1')

if __name__ == '__main__':
    
    kfold_predict()
    
