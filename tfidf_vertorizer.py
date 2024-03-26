import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


with open('train_raw_texts.txt', 'r', encoding="utf-8") as file:
    lines = file.readlines()

corpus= [line.strip() for line in lines]
file.close()

text = corpus[0].lower()

stop_words = set(stopwords.words('english') + list(string.punctuation) + ["``","''","’"])

tokens = word_tokenize(text)
#print(tokens)

token_set = set(tokens)

filtered_token_set = []

for token in token_set:
    if not (token in stop_words) and not token.isdigit():
        filtered_token_set.append(token)

filtered_token_set.sort()

filtered_token_string = ""

for token in filtered_token_set:
    filtered_token_string += token + " "

print(filtered_token_string)


