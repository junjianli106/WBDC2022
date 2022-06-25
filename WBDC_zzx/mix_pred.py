import json
import numpy as np
from tqdm import tqdm
from category_id_map import lv2id_to_category_id


def mix_pred(paths, outpath):
    fs = [open(path).readlines() for path in paths]
    print(len(fs[0]))
    
    with open(outpath, 'w') as outf:
        for t in tqdm(zip(*fs)):
            # print(len(t))
            id = None
            logits = []
            for d in t:
                obj = json.loads(d)
                id = obj['id']
                logits.append(obj['logit'])
            
            logits = np.array(logits)
            pred = logits.sum(axis=0).argmax()
            pred_id = lv2id_to_category_id(pred)
            outf.write(f'{id},{pred_id}\n')


if __name__ == '__main__':
    # paths = ['./result/result_seed12_raw.csv', './result/result_seed2022_raw.csv', './result/result_seed1314_raw.csv', './result/result_seed1026_raw.csv']
    paths = [f'./result/10fold_result{i}.csv' for i in range(10)]
    outpath = './result/result.csv'
    mix_pred(paths, outpath)
    