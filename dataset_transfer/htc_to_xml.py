import sys
import os

import json
import numpy as np
import tqdm as tqdm
    
def reverse_hierarchy(dataset):
    path = 'input/htc/' + dataset + '/' + dataset + '.taxonomy'
    hierarchy = {}

    with open(path, 'r') as file:
        for line in file:
            labels = line.strip().split('\t')

            parent = labels[0]
            children = labels[1:]
            if parent not in hierarchy:
                hierarchy[parent] = set()
            hierarchy[parent].update(children)

    precursor_map = {}
    for parent, children in hierarchy.items():
        for child in children:
            if child not in precursor_map:
                precursor_map[child] = set()
            precursor_map[child].add(parent)

    return precursor_map

def filter_labels(datapoint_labels, precursor_map):
    filtered_labels = set(datapoint_labels)
    for label in datapoint_labels:
        if label in precursor_map:
            precursors = precursor_map[label]
            for precursor in precursors:
                if precursor in filtered_labels:
                    filtered_labels.remove(precursor)
    return filtered_labels

def unique_labels(data, r_hiera):
    unique_labels = set()
    for line in tqdm.tqdm(data):
        data_point = json.loads(line)

        labels = data_point['label']
        labels = filter_labels(labels, r_hiera)
        for label in labels:
            unique_labels.add(label)

    return unique_labels

def htc_to_xml(data, id_map, r_hiera):
    labels_list = []
    raw_texts_list = []

    for line in tqdm.tqdm(data):
        data_point = json.loads(line)

        labels = data_point['label']
        labels = filter_labels(labels, r_hiera)
        labels = [str(id_map[label]) for label in labels if label in id_map]
        labels_text = ' '.join(labels)
        
        labels_list.append(labels_text)

        raw_texts_list.append(data_point['token'].replace('\n', ''))

    print(len(labels_list))
    print(len(raw_texts_list))

    return (raw_texts_list, labels_list)

def split_train_test(dataset, leaves_only):
    in_path = 'input/htc/' + dataset + '/' + dataset
    out_path = 'output/htc/' + dataset + '/' if not leaves_only else 'output/htc/' + dataset + '_leaves/'

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    data_train = []
    data_val = []
    data_test = []
    with open(in_path + '_train.json') as train, open(in_path + '_val.json') as val, open(in_path + '_test.json') as test:
        data_train = train.readlines()
        data_val = val.readlines()
        data_test = test.readlines()
    data_train += data_val

    r_hiera = reverse_hierarchy(dataset) if leaves_only else {}

    train_labels = unique_labels(data_train, r_hiera)
    test_labels = unique_labels(data_test, r_hiera)
    all_labels = train_labels | test_labels
    all_labels = sorted(all_labels, key=str.lower)
    print(f'label count: {len(all_labels)}')

    id_map = dict()
    for id, label in enumerate(all_labels):
        id_map[label] = id
    with open(out_path + 'id_map.json', 'w+') as f:
        json.dump(id_map, f, indent=4)

    train_raw_texts, train_labels = htc_to_xml(data_train, id_map, r_hiera)
    test_raw_texts, test_labels = htc_to_xml(data_test, id_map, r_hiera)

    with open(out_path + 'train_labels.txt', 'w+') as f:
        f.writelines('%s\n' % labels for labels in train_labels)
    with open(out_path + 'train_raw_texts.txt', 'w+') as f:
        f.writelines('%s\n' % text for text in train_raw_texts)

    with open(out_path + 'test_labels.txt', 'w+') as f:
        f.writelines('%s\n' % labels for labels in test_labels)
    with open(out_path + 'test_raw_texts.txt', 'w+') as f:
        f.writelines('%s\n' % text for text in test_raw_texts)

    return

if __name__ == '__main__':
    dataset = sys.argv[1]
    leaves_only = True if len(sys.argv) > 2 else False
    split_train_test(dataset, leaves_only)
    