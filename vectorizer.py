from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

with open("./text_preprocessed.txt", "r", encoding="utf-8") as file:
    lines = file.readlines()
corpus= [line.strip() for line in lines]
file.close()

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus[0:10])
vectorizer.get_feature_names_out()
a, b = X.nonzero()


print(X[0, 3250])