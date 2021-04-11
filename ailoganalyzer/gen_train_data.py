import os
import pandas as pd
from pylab import *
from sklearn.model_selection import train_test_split

def save(filename, liste):
    with open(filename, 'w') as the_file:
        for line in liste:
            #if not line == "[]":
            l = " ".join(line.strip('][').split(', '))
                #if not l.isspace():
            the_file.write(l + "\n")

def gen_train_test(test_prop, val_prop = 0.2, include_blank = False, train_with_abnormal = False):
    para = {
        "sequence_file": "../data/preprocess/sequence.csv",
        "train_file": "../data/train/train",
        "val_file" : "../data/train/normal",
        "abnormal_file": "../data/train/test_abnormal",
        "normal_file": "../data/train/test_normal"
    }
    sequence_label = pd.read_csv(para["sequence_file"])
    sequence = sequence_label["sequence"].values  # les sequences
    label = sequence_label["label"].values        # les labels

    blank = where(sequence == "[]")[0]
    print("number of blank line : ", blank.shape[0])

    # retire les lignes vide
    sequence = delete(sequence, blank)
    label = delete(label,blank)
    if not train_with_abnormal:
        normal = sequence[where(label == 0)]
        test_abnormal = sequence[where(label == 1)]
        train_set, test_normal = train_test_split(normal, test_size = test_prop, random_state = 42, shuffle = True)
        train, val = train_test_split(train_set, test_size = val_prop, random_state = 38, shuffle = True)

        print("train :", train.shape[0], "sequences")
        print("(dont", val.shape[0], "sequences pour la validation)")
        print("\t", train.shape[0], "normal")
        print("\t 0 abnormal")

        print("test :", test_normal.shape[0] + test_abnormal.shape[0], "sequences")
        print("\t",test_normal.shape[0], "normal")
        print("\t",test_abnormal.shape[0], "abnormal")
    else:
        train_set, test = train_test_split(sequence, test_size = test_prop, random_state = 42, shuffle = True)
        train_label, test_label = train_test_split(label, test_size = test_prop, random_state = 42, shuffle = True) # devrait etre ok mais a tester

        train, val = train_test_split(train_set, test_size = val_prop, random_state = 42, shuffle = True)

        #print(train)
        #print(where(test_label == 1))
        test_normal = test[(where(test_label == 0)[0])]
        test_abnormal = test[(where(test_label == 1)[0])]
        #a=train[where(train_label == 0)]

        print("train :", train.shape[0], "sequences")
        print("\t",train_set[where(train_label == 0)].shape[0], "normal")
        print("\t",train_set[where(train_label == 1)].shape[0], "abnormal")

        print("test :", test.shape[0], "sequences")
        print("\t",test_normal.shape[0], "normal")
        print("\t",test_abnormal.shape[0], "abnormal")

    if include_blank: train = append(train, ["[]"]*blank.shape[0])
    print("train :", train.shape[0], "sequences")


    save(para["train_file"], train)
    save(para["val_file"], val)
    save(para["abnormal_file"], test_abnormal)
    save(para["normal_file"], test_normal)
