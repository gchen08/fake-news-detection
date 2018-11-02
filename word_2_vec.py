#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Created on 2018/10/28

@author: Gong
"""

import pandas as pd
from gensim.models import word2vec
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

file_path = "data\\"
df = pd.read_csv(file_path + "all_news_tokenized.csv", encoding="utf-8")
corpus_set = df["content"]
print("contents: {}".format(len(corpus_set)))

# tf-idf
vectorizer = CountVectorizer(min_df=1e-5)
tf_idf = TfidfTransformer().fit_transform(vectorizer.fit_transform(corpus_set))
words = vectorizer.get_feature_names()
print("words: {}".format(len(words)))
print("tf-idf: ({}, {})".format(tf_idf.shape[0], tf_idf.shape[1]))
df.to_csv("label.csv", columns=["label"], index=False)
sparse.save_npz("tf_idf.npz", tf_idf)
print("tf-idf saved")

# vector
with open(file_path + "content.txt", 'w', encoding="utf-8") as f_out:
    for content in df["content"]:
        f_out.write(content)
        f_out.write("\n")
corpus = word2vec.Text8Corpus(file_path + "content.txt")
model = word2vec.Word2Vec(corpus, min_count=5, size=64)
model.save("news_vec_model")
model.wv.save_word2vec_format("news_vec_model.bin", binary=True)
print("word2vec saved")