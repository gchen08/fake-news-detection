#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Created on 2018/10/28

@author: Gong
"""

import pandas as pd
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

label = pd.read_csv("label.csv")
data = sparse.load_npz("tf_idf.npz")

ratio = .7
thr = int(len(label) * ratio)

train_label, train_set = label[: thr].values, data[: thr]
test_label, test_set = label[thr:].values, data[thr:]

# LR
lr_model = LogisticRegression()
lr_model.fit(train_set, train_label)
y_pre = lr_model.predict(test_set)
print(classification_report(test_label, y_pre))

# RF
rf_model = RandomForestClassifier(n_estimators=100, random_state=1024)
rf_model.fit(train_set, train_label.ravel())
y_pre = rf_model.predict(test_set)
print(classification_report(test_label, y_pre))