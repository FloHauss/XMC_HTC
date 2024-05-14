
NUM_LABELS = 13330
LAYER1_BF = 256
NUM_LAYER1_META_LABELS = LAYER1_BF
LAYER2_BF = 8
NUM_LAYER2_META_LABELS = LAYER1_BF * LAYER2_BF
DATASET = "AmazonCat-13K"

def split_array_k_ways(array_to_split, k):
    part_size = len(array_to_split) // k
    rest = len(array_to_split) % k

    parts = []
    start = 0
    for i in range(k):
        end = start + part_size + (1 if i < rest else 0)
        parts.append(list(map(lambda x: x + (i,),array_to_split[start:end])))
        start = end

    return parts

def get_accumulated_tfidf_value(datapoint):
    result = 0
    seperated_line = datapoint.split(" ")
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
    tfidf_value = get_accumulated_tfidf_value(line)
    relevant_lable_list = line.split(" ")[0].split(",")
    for relevant_label in relevant_lable_list:
        value_list[int(relevant_label)] += tfidf_value

enumerated_value_list = list(enumerate(value_list))

sorted_enumerated_value_list = sorted(enumerated_value_list, key=lambda x: x[1])

layer_1 = split_array_k_ways(sorted_enumerated_value_list, LAYER1_BF)

layer2 = []
for array in layer_1:
    layer2.append(split_array_k_ways(array, LAYER2_BF))

layer3 = []
for layer in layer2:
    for entry in layer:
        layer3.append(split_array_k_ways(entry,len(entry)))

with open(f"./{DATASET}_Clustering.txt", "w", encoding="utf-8") as file:
    file.write("root ")
    for index in range(LAYER1_BF):
        file.write(f"Meta_Label_{index} ")
    file.write("\n")

    for index in range(NUM_LAYER1_META_LABELS):
        file.write(f"Meta_Label_{index} ")
        for index2 in range(LAYER2_BF):
            file.write(f"Meta_Label_{LAYER1_BF+index*LAYER2_BF+index2} ")
        file.write("\n")


    for index in range(NUM_LAYER2_META_LABELS):
        file.write(f"Meta_Label_{LAYER1_BF+index} ")
        for entry in layer3[index]:
            print(entry)
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
