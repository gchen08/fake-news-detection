#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Created on 2018/10/28

@author: Gong
"""
import jieba
import pandas as pd


# segment
def token(content, stopwords):
    cut_result = jieba.cut(content.strip())
    out_str = ""
    for partition in cut_result:
        if partition != "\t" and partition not in stopwords:
            out_str += partition + " "
    return out_str


file_path = "data\\"
df = pd.read_csv(file_path + "all_news.csv", encoding="utf-8")

stopwords_list = []
with open("stopwords.txt", 'r') as f:
    for line in f.readlines():
        stopwords_list.append(line.strip())

contents = df["content"].tolist()
segments = []
i = 0
tot = len(df)
for line in contents:
    segments.append(token(line, stopwords_list))
    i += 1
    if i % 10000 == 0:
        print("progress: {:.2f}%".format(i * 100 / tot))
df["content"] = segments

# save result
df.to_csv(file_path + "all_news_tokenized.csv", encoding="utf-8", index=False)
print("Done.")
