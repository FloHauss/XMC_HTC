
def getAccumulatedTfIdfValue(line):

    result = 0 

    seperated_line = line.split(" ")
    del seperated_line[0]
    for item in seperated_line:
        result += float(item.split(":")[1])
        print(f"Item : {item} , Result : {result}")
        
    print(result)
    return result 



with open('./train.txt', 'r', encoding="utf-8") as file:
    lines = file.readlines()
corpus = [line.strip() for line in lines]
file.close()

for line in corpus:
    relevant_lables = line.split(" ")[0]
    relevant_lable_list = 
    print(relevant_lables)
    #getAccumulatedTfIdfValue(line)