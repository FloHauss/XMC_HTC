import sys
import os

import json
import numpy as np
import tqdm as tqdm
    
def count_layers(hierarchy, root):
    def dfs(node, depth):
        nonlocal max_depth
        if node in hierarchy:
            for child in hierarchy[node]:
                dfs(child, depth + 1)
        max_depth = max(max_depth, depth)

    max_depth = 0
    dfs(root, 0)
    return max_depth

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

    return hierarchy, precursor_map

def filter_labels(datapoint_labels, precursor_map):
    filtered_labels = set(datapoint_labels)
    for label in datapoint_labels:
        if label in precursor_map:
            precursors = precursor_map[label]
            for precursor in precursors:
                if precursor in filtered_labels:
                    filtered_labels.remove(precursor)
    return filtered_labels

def bfs_ordered_print(hierarchy, root, file_path):
    from collections import deque
    
    with open(file_path, 'w+') as f:
        queue = deque([root])
        
        while queue:
            node = queue.popleft()
            if node in hierarchy:
                children = sorted(hierarchy[node])  # Sorting to ensure consistent order
                
                #f.write(f'{node}: {children}\n')
                line = []
                line.append(node)
                line += children
                line = '\t'.join(line)
                f.write(f'{line}\n') 

                queue.extend(children)

def filter_hierarchy(dataset, hierarchy, valid_nodes):
    path = 'output/htc_lite/' + dataset + '_lite/' + dataset + '_lite.taxonomy'

    print('---')
    valid_nodes.add('Root')

    def recursive_filter(node):
        if node not in valid_nodes:
            if node in hierarchy:
                children = hierarchy[node]
                return [child for child in children if child in valid_nodes] + [
                    grandchild for child in children for grandchild in recursive_filter(child)
                ]
            else:
                return []
        else:
            if node in hierarchy:
                new_children = []
                for child in hierarchy[node]:
                    new_children.extend(recursive_filter(child))
                if new_children:
                    filtered_hierarchy[node] = set(new_children)
            return [node]

    # Initialize filtered hierarchy
    filtered_hierarchy = {}

    # Start the filtering process from the root node
    recursive_filter('Root')

    print('---')
    print(filtered_hierarchy)

    print(count_layers(filtered_hierarchy, 'Root'))
    
    bfs_ordered_print(filtered_hierarchy, 'Root', path)

    #with open(path, 'w+') as f: 
   #     for parent, children in filtered_hierarchy.items():
  #          line = []
    #        line.append(parent)
  #          line += children
  #          line = '\t'.join(line)

  #          f.write(f'{line}\n') 

def unique_labels(data):
    unique_labels = set()
    for line in tqdm.tqdm(data):
        data_point = json.loads(line)

        labels = data_point['label']
        for label in labels:
            unique_labels.add(label)

    return unique_labels

def htc_to_htc_lite(data, reverse_hierarchy):
    labels_list = []
    texts_list = []
    unique_labels = set()

    data_lite = []

    for line in tqdm.tqdm(data):
        data_point = json.loads(line)

        labels = data_point['label']
        leaves = filter_labels(labels, reverse_hierarchy)
        labels = [label for label in labels if label in leaves]
        unique_labels.update(labels)
        
        
        labels_list.append(labels)
        texts_list.append(data_point['token'].replace('\n', ''))

        data_lite.append({'token':data_point['token'], 'label':labels})

    print(len(labels_list))
    print(len(texts_list))
    print(len(data_lite))

    return (data_lite, unique_labels)

def split_train_test(dataset):
    in_path = 'input/htc/' + dataset + '/' + dataset
    out_path = 'output/htc_lite/' + dataset + '_lite/' 

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    out_path += dataset + '_lite'

    with open(in_path + '_train.json') as train, open(in_path + '_val.json') as val, open(in_path + '_test.json') as test:
        data_train = train.readlines()
        data_val = val.readlines()
        data_test = test.readlines()

    train_labels = unique_labels(data_train)
    val_labels = unique_labels(data_val)
    test_labels = unique_labels(data_test)
    all_labels = train_labels | val_labels | test_labels
    all_labels = sorted(all_labels, key=str.lower)
    print(f'label count: {len(all_labels)}')

    hiera, r_hiera = reverse_hierarchy(dataset)
    print(r_hiera)

    #all_labels = filter_labels(all_labels, r_hiera)
    #print(f'label count: {len(all_labels)}')
    #filter_hierarchy(dataset, hiera, all_labels)

    train, train_labels = htc_to_htc_lite(data_train, r_hiera)
    val, val_labels = htc_to_htc_lite(data_val, r_hiera)
    test, test_labels = htc_to_htc_lite(data_test, r_hiera)

    all_labels = train_labels | val_labels | test_labels
    print(len(all_labels))

    #all_labels = filter_labels(all_labels, r_hiera)
    #print(f'label count: {len(all_labels)}')
    filter_hierarchy(dataset, hiera, all_labels)

  #  print('---')
  #  print(set(all_labels).difference(kek))

    with open(out_path + '_train.json', 'w+') as f:
        for entry in train:
            json.dump(entry, f)
            f.write('\n')

    with open(out_path + '_val.json', 'w+') as f:
        for entry in val:
             json.dump(entry, f)
             f.write('\n')

    with open(out_path + '_test.json', 'w+') as f:
        for entry in test:
             json.dump(entry, f)
             f.write('\n')

    '''
    with open(out_path + '_train.json', 'w+') as f:
        for text, labels in zip(*train):
             entry = {'token':text, 'label':labels, 'doc_topic':[], 'doc_keyword':[]}
             json.dump(entry, f)
             f.write('\n')

    with open(out_path + '_val.json', 'w+') as f:
        for text, labels in zip(*val):
             entry = {'token':text, 'label':labels, 'doc_topic':[], 'doc_keyword':[]}
             json.dump(entry, f)
             f.write('\n')

    with open(out_path + '_test.json', 'w+') as f:
        for text, labels in zip(*test):
             entry = {'token':text, 'label':labels, 'doc_topic':[], 'doc_keyword':[]}
             json.dump(entry, f)
             f.write('\n')
    '''
    return

if __name__ == '__main__':
    dataset = sys.argv[1]
    split_train_test(dataset)
    