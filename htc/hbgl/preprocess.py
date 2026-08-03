#!/usr/bin/env python3
import sys
import tqdm
import json
from shutil import rmtree
from collections import defaultdict

task = sys.argv[1]
seq_len = int(sys.argv[2])

files=  ['../data/' + task + '/' + task + '_train.json',
            '../data/' + task + '/' + task + '_val.json',
            '../data/' + task + '/' + task + '_test.json']

label_dict = {}
hiera = defaultdict(set)
with open('../data/' + task + '/' + task + '.taxonomy') as f:
    label_dict['Root'] = -1
    for line in f.readlines():
        line = line.strip().split('\t')
        for i in line[1:]:
            if i not in label_dict:
                label_dict[i] = len(label_dict) - 1
            hiera[label_dict[line[0]]].add(label_dict[i])
    label_dict.pop('Root')



class_label_dict = {}
def loop_hiera(i, n):
    for j in list(hiera[i]):
        class_label_dict[j] = n
        if j in hiera:
            loop_hiera(j, n + 1)
loop_hiera(-1, 0)
print('end')

d = {}
label_sets = {}
for i in files:
    print(i)
    d[i] = [json.loads(f) for f in tqdm.tqdm(open(i))]

    for j in d[i]:
        for l in j['label']:
            label_sets[l] = 0
label_sets = label_sets.keys()

label_list = sorted([k for k in label_dict])
#label_list = sorted(list(label_sets))

print(len(label_list))
print(len(label_dict))
print(len(class_label_dict))
#assert len(label_list) == len(label_dict) == len(class_label_dict)
assert len(label_dict) == len(class_label_dict)

label_map = {label: f'[A_{i}]' for i, label in enumerate(label_list)}

import pickle
with open('../data/' + task + '/' + task + '_label_map.pkl', 'wb') as f:
    pickle.dump(label_map, f)


label_lens = []
def label_to_tgt(labels):
    global label_lens
    index = [class_label_dict[label_dict[i]] for i in labels]

    labels = [label_map[i] for i in labels]

    ms = sorted(list(zip(index, labels)), key=lambda x: x[0])
    labels = [i[1] for i in ms]

    label_lens.append(len(labels))
    return ' '.join(labels)

def label_to_tgt_list(labels):
    nyts = [[] for i in range(seq_len-1)]

    index = [class_label_dict[label_dict[i]] for i in labels]
    labels = [label_map[i] for i in labels]
    for i, l in zip(index, labels):
        nyts[i].append(l)
    # print(''.join([str(len(i)) for i in nyts]))
    return nyts


for file in d:
    assert '.json' in file
    print(file)
    if 'train' in file:
        with open(file.replace('.json', '_generated_tl.json'), 'w') as f:
            for l in tqdm.tqdm(d[file]):
                f.write(json.dumps({'src': l['token'],
                                    'tgt': label_to_tgt_list(l['label']) }) + '\n')

    else:
        with open(file.replace('.json', '_generated.json'), 'w') as f:
            for l in tqdm.tqdm(d[file]):
                f.write(json.dumps({'src': l['token'],
                                    'tgt': label_to_tgt(l['label']) }) + '\n')
