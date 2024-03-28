from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

with open("./text_preprocessed.txt", "r", encoding="utf-8") as file:
    lines = file.readlines()
corpus= [line.strip() for line in lines]
file.close()


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
a, b = X.nonzero()

tuple_list = list(zip(a, b))

value_list = []

for tuple in tuple_list:
    value_list.append((tuple[0],X[tuple]))

tfidf_values = [tupel[1] for tupel in value_list]
minimun_value = min(tfidf_values)

value_list_normalized = []
for tupel in value_list:
    value_list_normalized.append((tupel[0],tupel[1]/minimun_value))

cnt = 0

with open("./train_without_labels.txt", "w", encoding="utf-8") as file:
    for element in value_list_normalized:
        if element[0] > cnt:
            file.write("\n")
            cnt += 1
        file.write(str(element[1]) + " ")

file.close()

