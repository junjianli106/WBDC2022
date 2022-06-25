from os import stat
from re import S
import torch
from tqdm import tqdm
from collections import OrderedDict


def mix_ckpts(paths, out_path):
    ckpts = []
    for path in tqdm(paths):
        ckpts.append(torch.load(path, map_location="cpu"))
    # print('sleep')
    # from time import sleep
    # sleep(100)
    l = len(paths)
    print(paths)
    print('finish load ckpts!')

    state_dict_name = 'model_state_dict'
    
    res = {}
    for k, v in ckpts[0].items():
        if k != state_dict_name:
            res[k] = v
    res[state_dict_name] = OrderedDict()
    assert type(ckpts[0][state_dict_name]) is OrderedDict
    types = set()  # torch.float32, torch.int64=>position_ids
    for k, v in ckpts[0][state_dict_name].items():
        types.add(v.dtype)
        if v.dtype == torch.int64:
            res[state_dict_name][k] = v
        else:
            temp = []
            for i in range(l):
                temp.append(ckpts[i][state_dict_name][k])
            res[state_dict_name][k] = torch.mean(torch.stack(temp, dim=0), dim=0)
    torch.save(res, out_path)

    print(types)
    print(f'finish save mixed ckpt to {out_path}!')


def get_ckpt(dir):
    import os
    name = os.listdir(dir)[0]
    return os.path.join(dir, name)


if __name__ == '__main__':
    paths = ['/data/zhangzhexin/weixin_2022/challenge/save/chinese-roberta-wwm-ext/pretrainbs128_len128_epoch14/lr6e-5_warmup0.1_bs32_len384_pooldrop0.1_fgm0.15_ema_seed1026/model_epoch_3.bin',
            '/data/zhangzhexin/weixin_2022/challenge/save/chinese-roberta-wwm-ext/pretrainbs128_len128_epoch14/lr6e-5_warmup0.1_bs32_len384_pooldrop0.1_fgm0.15_ema_seed1026/model_epoch_4.bin']
    outpath = '/data/zhangzhexin/weixin_2022/challenge/save/chinese-roberta-wwm-ext/pretrainbs128_len128_epoch14/lr6e-5_warmup0.1_bs32_len384_pooldrop0.1_fgm0.15_ema_seed1026/mix34.bin'
    mix_ckpts(paths, outpath)
