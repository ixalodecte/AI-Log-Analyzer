import os
import pandas as pd
from pylab import *

def save(filename, liste):
    with open(filename, 'w') as the_file:
        for line in liste:
            #if not line == "[]":
            l = " ".join(line.strip('][').split(', '))
                #if not l.isspace():
            the_file.write(l + "\n")


para = {
    "train_proportion": 0.05,
    "train_with_abnormal": False,
    "include_blank": False,
    "sequence_file": "../data/preprocess/sequence.csv",
    "train_file": "../data/train/train",
    "abnormal_file": "../data/train/abnormal",
    "normal_file": "../data/train/normal"
}

sequence_label = pd.read_csv(para["sequence_file"])
sequence = sequence_label["sequence"].values  # les sequences
label = sequence_label["label"].values        # les labels

blank = where(sequence == "[]")[0]
print("number of blank line : ", blank.shape[0])

# retire les lignes vide
sequence = delete(sequence, blank)
label = delete(label,blank)
separator = int(sequence.shape[0]*para["train_proportion"])
print(separator)
if not para["train_with_abnormal"]:
    normal = sequence[where(label == 0)]
    train = normal[:separator]
    test_normal = normal[separator:]
    test_abnormal = sequence[where(label == 1)]

    print("train :", train.shape[0], "sequences")
    print("\t", train.shape[0], "normal")
    print("\t 0 abnormal")

    print("test :", test_normal.shape[0] + test_abnormal.shape[0], "sequences")
    print("\t",test_normal.shape[0], "normal")
    print("\t",test_abnormal.shape[0], "abnormal")
else:
    train = sequence[:separator]
    test = sequence[separator:]

    test_label = label[separator:]
    train_label = label[:separator]
    #print(train)
    #print(where(test_label == 1))
    test_normal = test[(where(test_label == 0)[0])]
    test_abnormal = test[(where(test_label == 1)[0])]
    a=train[where(train_label == 0)]

    print("train :", train.shape[0], "sequences")
    print("\t",train[where(train_label == 0)].shape[0], "normal")
    print("\t",train[where(train_label == 1)].shape[0], "abnormal")

    print("test :", test.shape[0], "sequences")
    print("\t",test_normal.shape[0], "normal")
    print("\t",test_abnormal.shape[0], "abnormal")

if para["include_blank"]:train = append(train, ["[]"]*blank.shape[0])
print("train :", train.shape[0], "sequences")


save(para["train_file"], train)
save(para["abnormal_file"], test_abnormal)
save(para["normal_file"], test_normal)
