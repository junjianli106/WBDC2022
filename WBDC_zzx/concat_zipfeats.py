import zipfile
import os
from tqdm import tqdm

# paths = ['labeled.zip', 'test_a.zip', 'unlabeled.zip']
# for path in paths:
#     print(path)
#     with zipfile.ZipFile(f'./data/zip_feats/{path}') as zf:
#         zf.extractall('./data/zip_feats/all')

print('begin zip')
with zipfile.ZipFile(f'./data/zip_feats/all.zip', 'w', compression=zipfile.ZIP_DEFLATED) as outzf:
    for name in tqdm(os.listdir('./data/zip_feats/all')):
        # print(name)
        outzf.write(f'./data/zip_feats/all/{name}', arcname=name)