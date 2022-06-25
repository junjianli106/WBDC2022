import random
import json

def load_json(name):
    with open(name) as f:
        a = json.load(f)
    
    return a

random.seed(2022)
test = load_json('./data/annotations/test_a.json')
labeled = load_json('./data/annotations/labeled.json')

unlabeled = load_json('./data/annotations/unlabeled.json')

print(len(test), len(labeled), len(unlabeled))
res = test + labeled + unlabeled
print(len(res))
with open('./data/annotations/all.json', 'w') as outf:
    json.dump(res, outf, ensure_ascii=False, indent=2)