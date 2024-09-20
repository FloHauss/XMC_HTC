from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix, save_npz
from tqdm import tqdm

# raw text
text = ["train_raw_texts.txt", "test_raw_texts.txt"]

with open(text[0], 'r') as file:
    lines = file.readlines()
    trn = [line.strip() for line in lines]

with open(text[1], 'r') as file:
    lines = file.readlines()
    tst = [line.strip() for line in lines]

vectorizer = TfidfVectorizer(stop_words='english', analyzer='word', max_features=None)  #change max features the way you need it

vectorizer.fit(trn)

X_trn = vectorizer.transform(trn)
X_tst = vectorizer.transform(tst)


csr_trn = csr_matrix(X_trn)
csr_tst = csr_matrix(X_tst)

save_npz(f"{text[0]}.npz", csr_trn)
save_npz(f"{text[1]}.npz", csr_tst)

# label
label = ["train_labels.txt", "test_labels.txt"]

with open(label[0], 'r') as file:
    lines = file.readlines()
    trn = [list(map(int, line.strip().split(','))) for line in lines]

with open(label[1], 'r') as file:
    lines = file.readlines()
    tst = [list(map(int, line.strip().split(','))) for line in lines]

def create_label_matrix(arr):
    num_rows = len(arr)
    num_cols = max(label for row in arr for label in row) + 1

    row_indices = []
    col_indices = []
    data = []
    
    for i, labels in enumerate(arr):
        for label in labels:
            row_indices.append(i)
            col_indices.append(label)
            data.append(1.0)

    label_matrix = csr_matrix((data, (row_indices, col_indices)), shape=(num_rows, num_cols))
    
    return label_matrix

trn_label = create_label_matrix(trn)
tst_label = create_label_matrix(tst)

save_npz(f"{label[0]}.npz", trn_label)
save_npz(f"{label[1]}.npz", tst_label)
