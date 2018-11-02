# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 17:27:31 2018

@author: Gong
"""

import pandas as pd
import fasttext
import time

def load_file():
    file_path = "data\\"
    df = pd.read_csv(file_path + "all_news_tokenized.csv", encoding="utf-8")

    ratio = .7
    thr = int(len(df) * ratio)

    with open(file_path + "news_train.txt", 'w', encoding="utf-8") as f:
        for i in range(thr):
            f.write(df["content"][i] + "\t" + "__label__" + str(df["label"][i]) + "\n")

    with open(file_path + "news_test.txt", 'w', encoding="utf-8") as f:
        for i in range(thr, len(df)):
            f.write(df["content"][i] + "\t" + "__label__" + str(df["label"][i]) + "\n")

#load_file()
tos = time.clock()
classifer = fasttext.supervised("news_train.txt", "news_fasttext.model", label_prefix="__label__")
toe = time.clock()
print("Training time: %s Seconds" % (toe - tos))

tos = time.clock()
classifer = fasttext.load_model("news_fasttext.model.bin", label_prefix="__label__")
result = classifer.text("news_test.txt")
print(result.precision)
print(result.recall)
toe = time.clock()
print("Testing time: %s Seconds" % (toe - tos))