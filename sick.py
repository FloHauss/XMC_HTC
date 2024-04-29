
import re
from sklearn.cluster import KMeans
import numpy as np

with open('./train.txt', 'r', encoding="utf-8") as file:
    lines = file.readlines()
tfidf_values = [line.strip() for line in lines]
file.close()

samples = [x for x in range(0, 30938)]
value_list = [0] * 30938

for line in tfidf_values:

    line_split = line.split(" ")
    label_index = line_split[0].split(",")

    val_sum = 0
    for num in range(1, len(line_split)):
        val_string = re.split( "\:" , line_split[num])
        val_float = float(val_string[1])
        val_sum += val_float

        for index in label_index:
            value_list[int(index)] += val_sum
    
#print(value_list)
#print(len(value_list))

###########################################################################################
#Clustering:
print(value_list)

array = np.array(list(zip(samples,value_list)))
print(array)
kmeans = KMeans(n_clusters=6000, random_state=0, n_init="auto").fit_predict(array)
print(kmeans)