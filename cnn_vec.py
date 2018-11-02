# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 21:40:58 2018

@author: Gong
"""
import numpy as np
import pandas as pd
from gensim.models.keyedvectors import KeyedVectors
from keras import regularizers
from keras.layers import Embedding, Dense, Input, MaxPooling2D, Dropout, concatenate, Conv2D
from keras.layers.core import Reshape, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

MAX_LEN = 150
NUM_WORDS = 20000
EMBEDDING_DIM = 300


def read_file(file_path):
    df = pd.read_csv(file_path, encoding="utf-8")
    all_content = []
    all_label = []
    for _, row in df.iterrows():
        all_content.append(row["content"])
        all_label.append(1 if row["label"] is True else 0)
    return all_content, all_label


def preprocess(train_content, train_label, test_content, test_label):
    tokenizer = Tokenizer(num_words=NUM_WORDS)
    tokenizer.fit_on_sequences(train_content)
    x_train_seq = tokenizer.texts_to_sequences(train_content)
    x_test_seq = tokenizer.texts_to_sequences(test_content)
    x_train = sequence.pad_sequences(x_train_seq, maxlen=MAX_LEN)
    x_test = sequence.pad_sequences(x_test_seq, maxlen=MAX_LEN)
    y_train = np.array(train_label)
    y_test = np.array(test_label)
    return x_train, y_train, x_test, y_test, tokenizer.word_index


def text_cnn(x_train, word_index):
    word_vectors = KeyedVectors.load_word2vec_format("news_vec_model.bin", binary=True)
    vocabulary_size = min(len(word_index) + 1, NUM_WORDS)
    embedding_matrix = np.zeros((vocabulary_size, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= NUM_WORDS:
            continue
        try:
            embedding_vector = word_vectors[word]
            embedding_matrix[i] = embedding_vector
        except KeyError:
            embedding_matrix[i] = np.random.normal(0, np.sqrt(0.25), EMBEDDING_DIM)
    del word_vectors
    embedding_layer = Embedding(vocabulary_size,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                trainable=True)

    sequence_length = x_train.shape[1]
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
    # reshape = Reshape((3 * num_filters,))(flatten)

    dropout = Dropout(drop)(flatten)

    output = Dense(units=1,
                   activation='softmax',
                   kernel_regularizer=regularizers.l2(0.01))(dropout)

    model = Model(inputs, output)
    adam = Adam(lr=1e-3)
    model.compile(loss="binary_crossentropy",
                  optimizer=adam,
                  weighted_metrics=["accuracy"])
    return model


content, label = read_file("data\\all_news_tokenized.csv")
ratio = .7
thr = int(len(label) * ratio)
train_label, train_set = label[: thr], content[: thr]
test_label, test_set = label[thr:], content[thr:]
x_train, y_train, x_test, y_test, word_index = preprocess(train_set, train_label, test_set, test_label)
model = text_cnn(x_train, word_index)
batch_size = 64
epochs = 10
model.fit(x_train,
          y_train,
          batch_size=batch_size,
          epochs=epochs,
          shuffle=True)
score = model.evaluate(x_test, y_test)
print("test_loss: %f, accuracy: %f" % (score[0], score[1]))
with open("result\\cnn_vec.txt", "w") as f:
    f.write("test_loss: %f, accuracy: %f" % (score[0], score[1]))
f.close()
