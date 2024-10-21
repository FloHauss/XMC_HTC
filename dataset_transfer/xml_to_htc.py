from collections import defaultdict
import sys
import os
import json

import numpy as np
from sklearn.model_selection import train_test_split

def hiera(dataset, label_map):
    in_path = 'input/xml/' + dataset + '/' + dataset + '.taxonomy'
    out_path = 'output/xml/' + dataset + '/' + dataset + '.taxonomy'
    hiera = defaultdict(set)

    with open(in_path) as fi, open(out_path, 'w+') as fo:
        for i, line in enumerate(fi.readlines()):
            line = line.strip().split('\t')
            line = [label_map[int(label)] if label.isdigit() else label for label in line]
            print(line)
            line = '\t'.join(line) + '\n'
            fo.write(line)

    with open(out_path) as f:
        for i, line in enumerate(f.readlines()):
            if i == 0:
                continue

            line = line.strip().split('\t')
            parent = line[0]
            hiera[parent] = hiera.get(parent, [])
            for child in line[1:]:
                hiera[parent].append(child)

    r_hiera = {}
    for i in hiera:
        for j in list(hiera[i]):
             r_hiera[j] = i

    return r_hiera

def process_in_chunks(train_path, val_path, chunk_size=1000, test_size=0.2, random_state=0, train_size=None, val_size=None):
    # Read the file line by line and process in chunks
    train_data = []
    val_data = []

    with open(train_path, 'r') as f:
        chunk = []
        for line in f:
            chunk.append(line)
            if len(chunk) == chunk_size:
                # Process the chunk
                chunk_data = np.array(chunk)
                chunk_train, chunk_val = train_test_split(chunk_data, test_size=test_size, random_state=random_state)
                train_data.extend(chunk_train)
                val_data.extend(chunk_val)
                chunk = []
        
        # Don't forget to process any remaining lines in the last chunk
        if chunk:
            chunk_data = np.array(chunk)
            chunk_train, chunk_val = train_test_split(chunk_data, test_size=test_size, random_state=random_state)
            train_data.extend(chunk_train)
            val_data.extend(chunk_val)
    
    # Shuffle the entire train and validation datasets
    np.random.shuffle(train_data)
    np.random.shuffle(val_data)

        # Reduce the size of the datasets if specified
    if train_size:
        train_data = train_data[:train_size]
    if val_size:
        val_data = val_data[:val_size]

    # Write the train and validation data to their respective files
    with open(train_path, 'w') as f:
        f.writelines(train_data)
    with open(val_path, 'w') as f:
        f.writelines(val_data)

def split_train_val(train_path, val_path, size):
    f = open(train_path, 'r')
    data = f.readlines()
    f.close()

    id = [i for i in range(size)]
    np_data = np.array(data)
    np.random.shuffle(id)
    np_data = np_data[id]
    train, val = train_test_split(np_data, test_size=0.2, random_state=0)

    train = list(train)
    val = list(val)
    
    f = open(train_path, 'w')
    f.writelines(train)
    f.close()
    f = open(val_path, 'w')
    f.writelines(val)
    f.close()

def txt_to_json(texts, labels, label_map, r_hiera, path):
    data = []
    data_size = 0   # amount of data
    with open(texts, encoding='utf8') as texts:
        for line in texts:
                data.append({'token':line.strip(), 'label':[], 'doc_topic':[], 'doc_keyword':[]})
                data_size += 1
    with open(labels) as labels:
        for i, line in enumerate(labels):
            label_list = [label_map[int(label)] for label in line.split()]

            extended_label_list = []
            for label in label_list:
                cur = label
                while cur:
                    if cur not in extended_label_list : extended_label_list.append(cur)
                    cur = r_hiera.get(cur, None)
                    
            data[i]['label'] = extended_label_list

    with open(path, 'w+') as out_file:
        for entry in data:
             json.dump(entry, out_file)
             out_file.write('\n')

    return data_size

def convert_to_json(dataset):
    in_path = 'input/xml/' + dataset + '/' + dataset
    out_path = 'output/xml/' + dataset + '/' + dataset

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    train_texts_path = in_path + '_train_texts.txt'
    test_texts_path = in_path + '_test_texts.txt'
    train_labels_path = in_path + '_train_labels.txt'
    test_labels_path = in_path + '_test_labels.txt'

    label_map_path = in_path + '_label_map.txt'

    label_map = {}
    with open(label_map_path, encoding='utf8') as f:
        for i, line in enumerate(f):
            label_map[i] = line.strip()

    train_path = out_path + '_train.json'
    val_path = out_path + '_val.json'
    test_path = out_path + '_test.json'

    r_hiera = hiera(dataset, label_map)
    data_size = txt_to_json(train_texts_path, train_labels_path, label_map, r_hiera, train_path)
    #split_train_val(train_path, val_path, data_size)
    process_in_chunks(train_path, val_path, train_size=30000, val_size=5000)
    txt_to_json(test_texts_path, test_labels_path, label_map, r_hiera, test_path)
    
if __name__ == '__main__':
    dataset = sys.argv[1]
    convert_to_json(dataset)