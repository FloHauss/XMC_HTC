import numpy as np , math, torch 
from sklearn.cluster import KMeans

NUM_LABELS = 2812281

LAYER1_BF = 1024
NUM_LAYER1_META_LABELS = LAYER1_BF 

LAYER2_BF = 8
NUM_LAYER2_META_LABELS = LAYER1_BF * LAYER2_BF

LAYER3_BF = 8
NUM_LAYER3_META_LABELS = LAYER1_BF * LAYER2_BF * LAYER2_BF
dataset = "Amazon-3M"


def splitArraykWays(array, k):
    part_size = len(array) // k
    rest = len(array) % k

    parts = []
    start = 0
    for i in range(k):
        end = start + part_size + (1 if i < rest else 0)
        parts.append(list(map(lambda x: x + (i,),array[start:end])))
        start = end

    return parts

def getAccumulatedTfIdfValue(line):
    result = 0 
    seperated_line = line.split(" ")
    del seperated_line[0]
    for item in seperated_line:
        result += float(item.split(":")[1])
    return round(result,2)



with open('./train.txt', 'r', encoding="utf-8") as file:
    lines = file.readlines()
corpus = [line.strip() for line in lines]
file.close()

with open('./Yf.txt', 'r', encoding="latin-1") as file:
    lines = file.readlines()
label_corpus = [line.strip() for line in lines]
file.close()

label_number_to_text_map = {}
for index, label in enumerate(label_corpus):
    label_number_to_text_map[index] = label.replace(" ","_")


value_list = [0] * NUM_LABELS

for line in corpus:
    tfidf_value = getAccumulatedTfIdfValue(line)
    relevant_lable_list = line.split(" ")[0].split(",")
    for relevant_label in relevant_lable_list:
        value_list[int(relevant_label)] += tfidf_value

enumerated_value_list = list(enumerate(value_list))

sorted_enumerated_value_list = sorted(enumerated_value_list, key=lambda x: x[1])

layer_1 = splitArraykWays(sorted_enumerated_value_list, LAYER1_BF)


layer2 = []
for array in layer_1:
    layer2.append(splitArraykWays(array, LAYER2_BF))


layer3 = []
for layer in layer2:
    for entry in layer:
        layer3.append(splitArraykWays(entry,LAYER3_BF))


layer4 = []
for entry in layer3:
    for subarray in entry:
        layer4.append(splitArraykWays(subarray,len(subarray)))

print(len(layer4))
print(layer4[65535])

with open(f"./{dataset}_Clustering.txt", "w", encoding="utf-8") as file:
    
    # Write root and children of root
    file.write("root ")
    for index in range(LAYER1_BF):
        file.write(f"Meta_Label_{index} ")
    file.write("\n")
    
    # Write Layer 1 Meta Labels and children of Layer 1 Meta Labels
    for index in range(NUM_LAYER1_META_LABELS):
        file.write(f"Meta_Label_{index} ")
        for index2 in range(LAYER2_BF):
            file.write(f"Meta_Label_{NUM_LAYER1_META_LABELS+index*LAYER2_BF+index2} ")
        file.write("\n")

    # Write Layer 2 Meta Labels and children of Layer 2 Meta Labels
    for index in range(NUM_LAYER2_META_LABELS):
        file.write(f"Meta_Label_{NUM_LAYER1_META_LABELS+index} ")
        for index2 in range(LAYER3_BF):
            file.write(f"Meta_Label_{NUM_LAYER1_META_LABELS + NUM_LAYER2_META_LABELS + index * LAYER3_BF + index2} ")
        file.write("\n")

    # Write Layer 3 Meta Labels and full resolution label children
    for index in range(NUM_LAYER3_META_LABELS):
        file.write(f"Meta_Label_{NUM_LAYER1_META_LABELS+NUM_LAYER2_META_LABELS+index} ")
        for entry in layer4[index]:
            file.write(f"{label_number_to_text_map[entry[0][0]]} ")
        file.write("\n")

"""
flattenedhierarchy = []
for layer in layer3:
    for entry in layer:
        flattenedhierarchy.append(entry[0])

sorted_flattened_hierarchy = sorted(flattenedhierarchy, key=lambda x: x[0])





with open("./wiki10-31k_clustering.txt", "w", encoding="utf-8") as file:
    for index, item in enumerate(sorted_flattened_hierarchy):
        item_with_labelname = (label_corpus[index],) + item[2:]
        file.write(str(item_with_labelname))
        file.write("\n")

#print(f" eintrag 1 : {layer3[0]}, eintrag 2: {layer3[1]}")
    



#log_value_list = getLogValueList(list = value_list, base = 10)

#print(list(zip(value_list,log_value_list)))

#array = np.array(log_value_list).reshape(-1,1)
#print(value_list)

#############################################################
####################CLUSTERING###############################

###Shallow###

N_CLUSTERS = round(NUM_LABELS/1000)

kmeans = KMeans(n_clusters=N_CLUSTERS)
kmeans.fit(array)
centroids = kmeans.cluster_centers_

#for index,item in enumerate(array):
 #   print(f"Item: {item.reshape(1,-1)},Index: {index}, Prediction: {kmeans.predict(item.reshape(1,-1))}")


with open("./shallow_clustering.txt", "w", encoding="utf-8") as file:
    for item in array:
        file.write(str(kmeans.predict(item.reshape(1,-1))[0]))
        file.write("\n")
        """
"""
NUM_CLUSTERS = 2
CLUSTER_SIZE = NUM_LABELS // NUM_CLUSTERS
X = torch.from_numpy(array)


choices, centers = kmeans_equal(X, num_clusters=NUM_CLUSTERS, cluster_size=CLUSTER_SIZE)
print(choices,centers)"""