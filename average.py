
with open('./train_labels.txt', 'r', encoding="utf-8") as file:
    lines = file.readlines()
train_labels= [line.strip() for line in lines]
file.close()


with open('./test_labels.txt', 'r', encoding="utf-8") as file:
    lines = file.readlines()
test_labels= [line.strip() for line in lines]
file.close()


average_train = 0
for line in train_labels:
    cnt = line.count(" ")
    average_train += (cnt +1)
train_av = average_train/len(train_labels)


average_test = 0
for line in test_labels:
    cnt = line.count(" ")
    average_test += (cnt +1)
test_av = average_test/len(test_labels)

print(f"Average labels per train point: {train_av}")
print(f"Average labels per test point: {test_av}")
