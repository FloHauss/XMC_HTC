import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def hasNumbers(string):
    hasDigit = False
    for char in string:
        if(char.isdigit()):
            hasDigit = True
    return hasDigit

def listToString(list_of_strings):
    output_string = ""
    for string in list_of_strings:
        output_string += string + " "
    return output_string


with open('./train_raw_texts.txt', 'r', encoding="utf-8") as file:
    lines = file.readlines()
corpus= [line.strip() for line in lines]
file.close()

def preprocessing(text):
    text_split = text.lower().split(" ")

    filtered_text_split = []
    for word in text_split:
        if not("'" in word or "-" in word or "’" in word or hasNumbers(word)):
            filtered_text_split.append(word)

    filtered_text_string = listToString(filtered_text_split)

    stop_words = set(stopwords.words('english') + list(string.punctuation) + ["``","''","’"])
    tokens = word_tokenize(filtered_text_string)

    filtered_tokens = []
    for token in tokens:
        if not (token in stop_words):
            filtered_tokens.append(token)

    filtered_tokens_set = list(set(filtered_tokens))
    filtered_tokens_set.sort()

    preprocessed_string = listToString(filtered_tokens_set)

    return preprocessed_string

preprocessed_corpus = []
for text in corpus:
    preprocessed_corpus.append(preprocessing(text))

with open("./text_preprocessed.txt", "w", encoding="utf-8") as file:
    for element in preprocessed_corpus:
        file.write(element + "\n")
file.close()