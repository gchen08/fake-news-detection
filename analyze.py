#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Created on 2018/10/31

@author: Gong
"""
from gensim.models import word2vec

model = word2vec.Word2Vec.load("news_vec_model")
print(model.similar_by_word("中国", topn=10))
