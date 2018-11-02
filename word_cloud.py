# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 00:09:49 2018

@author: Gong
"""

import pandas as pd
from matplotlib import pyplot as plt
from wordcloud import WordCloud

file_path = "data\\"
df = pd.read_csv(file_path + "all_news_tokenized.csv", encoding="utf-8")

font = r"c:\windows\fonts\FZSTK.TTF"

df_true = df[df["label"] == True]
word_dict = {}
for content in df_true["content"]:
    assert "\n" not in content
    for word in content.split(" "):
        word_dict[word] = word_dict.get(word, 0) + 1
sorted_dict = sorted(word_dict, key=lambda x: word_dict[x], reverse=True)
out_str = []
for word in sorted_dict[:250]:
    if len(word) > 1:
        out_str.append(word)
word_cloud = WordCloud(font_path=font, background_color="white", width=1000, height=800).generate(" ".join(out_str))
word_cloud.to_file("true.png")
plt.imshow(word_cloud)
plt.axis("off")
plt.show()

df_false = df[df["label"] == False]
word_dict = {}
for content in df_false["content"]:
    assert "\n" not in content
    for word in content.split(" "):
        word_dict[word] = word_dict.get(word, 0) + 1
sorted_dict = sorted(word_dict, key=lambda x: word_dict[x], reverse=True)
out_str = []
for word in sorted_dict[:250]:
    if len(word) > 1:
        out_str.append(word)
word_cloud = WordCloud(font_path=font, background_color="white", width=1000, height=800).generate(" ".join(out_str))
word_cloud.to_file("false.png")
plt.imshow(word_cloud)
plt.axis("off")
plt.show()
