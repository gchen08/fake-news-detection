#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Created on 2018/10/29

@author: Gong
"""

import numpy as np
import pandas as pd
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Input, concatenate
from keras.layers.core import Flatten, Reshape
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

NUM_WORDS = 200
EMBEDDING_DIM = 32

df = pd.read_csv("data\\all_news_tokenized.csv", encoding="utf-8").replace((True, False), ("true_news", "fake_news"))

# 划分训练集与测试集
ratio = .7
thr = int(len(df) * ratio)
train_data = df[: thr]
test_data = df[thr:]

# 标签数字化
creds = train_data.label.unique()
dic = {}
for i, cred in enumerate(creds):
    dic[cred] = i
labels = train_data.label.apply(lambda x: dic[x])

# 抽样构建验证集
val_data = train_data.sample(frac=0.2, random_state=200)
train_data = train_data.drop(val_data.index)

tokenizer = Tokenizer(num_words=NUM_WORDS)
tokenizer.fit_on_texts(train_data.content)
sequence_train = tokenizer.texts_to_sequences(train_data.content)
sequence_valid = tokenizer.texts_to_sequences(val_data.content)
word_index = tokenizer.word_index
print("Unique tokens: %s" % len(word_index))

X_train = sequence.pad_sequences(sequence_train)
X_val = sequence.pad_sequences(sequence_valid, maxlen=X_train.shape[1])
y_train = to_categorical(np.asanyarray(labels[train_data.index]))
y_val = to_categorical(np.asanyarray(labels[val_data.index]))
print("Train and val X tensor:", X_train.shape, X_val.shape)
print("Train and val label tensor:", y_train.shape, y_val.shape)

# embedding
vocabulary_size = min(len(word_index) + 1, NUM_WORDS)
embedding_layer = Embedding(vocabulary_size,
                            EMBEDDING_DIM)

# CNN

sequence_length = X_train.shape[1]
filter_size = [3, 4, 5]
num_filters = 100
drop = 0.5

inputs = Input(shape=(sequence_length,))
embedding = embedding_layer(inputs)
reshape = Reshape((sequence_length, EMBEDDING_DIM, 1))(embedding)

conv_0 = Conv2D(filters=num_filters,
                kernel_size=(filter_size[0], EMBEDDING_DIM),
                activation="relu",
                kernel_regularizer=regularizers.l2(0.01))(reshape)
conv_1 = Conv2D(filters=num_filters,
                kernel_size=(filter_size[1], EMBEDDING_DIM),
                activation="relu",
                kernel_regularizer=regularizers.l2(0.01))(reshape)
conv_2 = Conv2D(filters=num_filters,
                kernel_size=(filter_size[2], EMBEDDING_DIM),
                activation="relu",
                kernel_regularizer=regularizers.l2(0.01))(reshape)

maxpool_0 = MaxPooling2D((sequence_length - filter_size[0] + 1, 1), strides=(1, 1))(conv_0)
maxpool_1 = MaxPooling2D((sequence_length - filter_size[1] + 1, 1), strides=(1, 1))(conv_1)
maxpool_2 = MaxPooling2D((sequence_length - filter_size[2] + 1, 1), strides=(1, 1))(conv_2)

merged_tensor = concatenate([maxpool_0, maxpool_1, maxpool_2], axis=1)

flatten = Flatten()(merged_tensor)

dropout = Dropout(drop)(flatten)

output = Dense(units=2,
               activation="softmax",
               kernel_regularizer=regularizers.l2(0.01))(dropout)

model = Model(inputs, output)
adam = Adam(lr=1e-3)
model.compile(loss="categorical_crossentropy",
              optimizer=adam,
              metrics=["acc"])

# train
callbacks = [EarlyStopping(monitor='val_loss')]
batch_size = 128
epochs = 10
model.fit(X_train,
          y_train,
          validation_split=0.1,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_val, y_val),
          callbacks=callbacks)

# test
sequence_test = tokenizer.texts_to_sequences(test_data.content)
X_test = sequence.pad_sequences(sequence_test, maxlen=X_train.shape[1])
y_test = to_categorical(np.asanyarray(labels[test_data.index]))
score = model.evaluate(X_test, y_test)
print("test_loss: %f, accuracy: %f" % (score[0], score[1]))
