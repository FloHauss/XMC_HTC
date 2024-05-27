import os
import numpy as np
import scipy.sparse as sp

dataset="params.data_path"
num_labels="params.num_labels"
LF_data="params.LF_data"

def run(dataset, num_labels, LF_data):
    inv_prop = load_data(dataset, num_labels, LF_data)


def load_data(dataset, num_labels, LF_data):
     train_labels = make_csr_labels(num_labels, f'{dataset}/Y.trn.npz', LF_data) #Write train.npz for LF datasets
     inv_prop = get_inv_prop(dataset, train_labels)
     
     return inv_prop



def make_csr_labels(num_labels, file_name, LF_data):
    if os.path.exists(file_name):
        print(f"Loading {file_name}")
        Y = sp.load_npz(file_name)
    else:
        with open(os.path.splitext(file_name)[0]+'.txt') as fil:
            if LF_data:
                data = fil.readlines()[1:] 
            row_idx, col_idx = [], []
            for i, lab in enumerate(fil.readlines()):
                if LF_data:
                    l_list = [int(l) for l in lab.split()[0].split(',')]
                else:
                    l_list = [int(l) for l in lab.replace('\n', '').split(',')]
                col_idx.extend(l_list)
                row_idx.extend([i]*len(l_list))

            m = max(row_idx) + 1
            n = num_labels
            val_idx = [1]*len(row_idx)
            Y = sp.csr_matrix((val_idx, (row_idx, col_idx)), shape=(m, n))
            print(f"Created {file_name}")
            sp.save_npz(file_name, Y)
    return Y

def get_inv_prop(dataset, Y):
    if os.path.exists(os.path.join(dataset, 'inv_prop.npy')):
        inv_prop = np.load(os.path.join(dataset, 'inv_prop.npy'))
        return inv_prop

    print("Creating inv_prop file")
    
    A = {'Eurlex': 0.6, 'LF-Amazon-131K': 0.6, 'Amazon-670K': 0.6, 'Amazon-3M': 0.6, 'AmazonCat-13K': 0.55, 'Wiki-500K' : 0.5, 'Wiki10-31K' : 0.55}
    B = {'Eurlex': 2.6, 'LF-Amazon-131K': 2.6, 'Amazon-670K': 2.6, 'Amazon-3M': 2.6, 'AmazonCat-13K': 1.5, 'Wiki-500K': 0.4, 'Wiki10-31K': 1.5}

    d = dataset.split('/')[-1]
    a, b = A[d], B[d]
    
    num_samples = Y.shape[0]
    inv_prop = np.array(Y.sum(axis=0)).ravel()
    
    c = (np.log(num_samples) - 1) * np.power(b+1, a)
    inv_prop = 1 + c * np.power(inv_prop + b, -a)
    
    np.save(os.path.join(dataset, 'inv_prop.npy'), inv_prop)
    return inv_prop