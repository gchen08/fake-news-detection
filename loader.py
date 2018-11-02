#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on 2018/10/28

@author: Gong
"""

import pandas as pd
from sklearn.utils import shuffle

file_path = "data\\"

fake_news_file = "weibo_rumors.xlsx"
true_news_files = ["cnews.test.txt", "cnews.train.txt", "cnews.val.txt"]

df_in = pd.read_excel(file_path + fake_news_file, encoding="utf-8")
df_out = df_in[["分类", "微博内容"]][df_in["分类"] != 'x']
df_out["label"] = "false"
df_out.to_csv(file_path + "fake_news.csv", index=False, encoding="utf-8", header=["topic", "content", "label"])

df_out = pd.DataFrame()
for file in true_news_files:
    df_in = pd.read_table(file_path + file, encoding="utf-8", header=None)
    df_in["label"] = "true"
    df_out = df_out.append(df_in)
df_out.to_csv(file_path + "true_news.csv", index=False, encoding="utf-8", header=["topic", "content", "label"])

df = pd.DataFrame()
df_in = pd.read_csv(file_path + "true_news.csv", encoding="utf-8")
df = df.append(df_in)
df_in = pd.read_csv(file_path + "fake_news.csv", encoding="utf-8")
df = df.append(df_in)

origin = ["p", "c", "z", "s", "n"]
target = ["政治", "经济", "社会", "社会", "生活"]
df["topic"] = df["topic"].replace(origin, target)
origin = ["体育", "财经", "房产", "家居", "教育", "科技", "时尚", "时政", "游戏", "娱乐"]
target = ["社会", "经济", "经济", "生活", "生活", "社会", "生活", "政治", "生活", "生活"]
df["topic"] = df["topic"].replace(origin, target)

df = shuffle(df)
df.to_csv(file_path + "all_news.csv", index=False, encoding="utf-8", header=["topic", "content", "label"])
