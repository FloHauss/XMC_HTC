from sklearn.feature_extraction.text import TfidfVectorizer

with open("./text_preprocessed.txt", "r", encoding="utf-8") as file:
    lines = file.readlines()
corpus= [line.strip() for line in lines]

with open("./vocab.txt", "r", encoding="utf-8") as file:
    lines = file.readlines()
vocab_file = [line.strip() for line in lines]


with open("./Y.trn.txt", "r", encoding="utf-8") as file:
    lines = file.readlines()
labels_file = [line.strip() for line in lines]


labels_comma = []
for labels in labels_file:
    labels_comma.append(labels.replace(" ", ","))

vocab_set = []
for text in corpus:
    vocab = list(set(text.split()))
    vocab.sort()
    vocab_set.append(vocab)



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


with open("./train.txt", "w", encoding="utf-8") as file:
    
    for cnt in range(0,len(vocab_set)):

        amount_values = 0
        for value in value_list_normalized:
            if value[0] == cnt:
                amount_values += 1
        amount_vocab = len(vocab_set[cnt])
        max_iterations = min(amount_values, amount_vocab)

        cnt2 = 0
        values = [value[1] for value in value_list_normalized if value[0] == cnt]

        file.write(str(labels_comma[cnt]) + " ")
        while(cnt2 < max_iterations):
            file.write(str(vocab_file.index(vocab_set[cnt][cnt2]))+":"+ str(values[cnt2]) + " ")
            cnt2 += 1
        file.write("\n")


