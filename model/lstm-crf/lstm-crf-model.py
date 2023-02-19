import pandas as pd

import pickle
import operator
import re
import string
import pandas as pd
import keras.backend as k
import numpy as np
# import matplotlib.pyplot as plt
#
# from plot_keras_history import plot_history
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix
from keras_contrib.utils import save_load_utils

from keras import layers
from keras import optimizers

from keras.models import Model

from keras_contrib.layers import CRF
from keras_contrib import losses
from keras_contrib import metrics

# df1 = pd.read_csv('data/sample.txt')
# unique_values_list = df1['labels'].unique().tolist()
# print(unique_values_list)
# print(df1.shape)
# print(df1.dtypes)
# print(df1.isnull().values.any())
#
# df1 = df1.dropna()
# print(df1.isnull().values.any())
#
# print(len(df1))
# print('---')
#
# df1 = df1[~df1['words'].str.contains('\[')]
# df1 = df1[~df1['words'].str.contains('\]')]
# df1 = df1[~df1['words'].str.contains('\'')]
# df1 = df1[~df1['words'].str.contains('\(')]
# df1 = df1[~df1['words'].str.contains('\)')]
# df1 = df1[~df1['words'].str.contains('-')]
# df1 = df1[~df1['words'].str.contains(';')]
# df1 = df1[~df1['words'].str.contains(' ')]
# df1 = df1[~df1['words'].str.contains('“')]
# df1 = df1[~df1['words'].str.contains('”')]
# df1 = df1[~df1['words'].str.contains(',')]
# df1 = df1[~df1['words'].str.contains('’')]
# df1 = df1[~df1['words'].str.contains(':')]
# df1 = df1[~df1['words'].str.contains(' ')]

df1 = pd.read_csv('testing.csv')

sentence_id_list=[]
sentence_id_seq = 0
for word in df1['words'].tolist():
    if word == "." or word == "?" or word == "!":
        sentence_id_list.append(sentence_id_seq)
        sentence_id_seq += 1
    else:
        sentence_id_list.append(sentence_id_seq)

df1['sentence_id'] = sentence_id_list



all_words = list(set(df1["words"].values))
all_tags = list(set(df1["labels"].values))

word_counts = df1.groupby("sentence_id")["words"].agg(["count"])
word_counts = word_counts.rename(columns={"count": "Word count"})
word_counts.hist(bins=50, figsize=(8,6));

MAX_SENTENCE = word_counts.max()[0]
print("Longest sentence in the corpus contains {} words.".format(MAX_SENTENCE))

word2index = {word: idx + 2 for idx, word in enumerate(all_words)}
word2index["--UNKNOWN_WORD--"]=0
word2index["--PADDING--"]=1
index2word = {idx: word for word, idx in word2index.items()}

for k,v in sorted(word2index.items(), key=operator.itemgetter(1))[:10]:
    print(k,v)

test_word = "Scotland"

test_word_idx = word2index[test_word]
test_word_lookup = index2word[test_word_idx]

print("The index of the word {} is {}.".format(test_word, test_word_idx))
print("The word with index {} is {}.".format(test_word_idx, test_word_lookup))

tag2index = {tag: idx + 1 for idx, tag in enumerate(all_tags)}
tag2index["--PADDING--"] = 0

index2tag = {idx: word for word, idx in tag2index.items()}

def to_tuples(data):
    iterator = zip(df1["words"].values.tolist(),
                   df1["labels"].values.tolist())
    return [(word, tag) for word, tag in iterator]

sentences = df1.groupby("sentence_id").apply(to_tuples).tolist()
#
print(df1)





# X = [[word[0] for word in sentence] for sentence in sentences]
# y = [[word[2] for word in sentence] for sentence in sentences]
# print("X[0]:", X[0])
# print("y[0]:", y[0])
#
# X = [sentence + [word2index["--PADDING--"]] * (MAX_SENTENCE - len(sentence)) for sentence in X]
# y = [sentence + [tag2index["--PADDING--"]] * (MAX_SENTENCE - len(sentence)) for sentence in y]
# print("X[0]:", X[0])
# print("y[0]:", y[0])
#
# TAG_COUNT = len(tag2index)
# y = [ np.eye(TAG_COUNT)[sentence] for sentence in y]
# print("X[0]:", X[0])
# print("y[0]:", y[0])
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1234)
#
# print("Number of sentences in the training dataset: {}".format(len(X_train)))
# print("Number of sentences in the test dataset : {}".format(len(X_test)))
#
# X_train = np.array(X_train)
# X_test = np.array(X_test)
# y_train = np.array(y_train)
# y_test = np.array(y_test)
#
# WORD_COUNT = len(index2word)
# DENSE_EMBEDDING = 50
# LSTM_UNITS = 50
# LSTM_DROPOUT = 0.1
# DENSE_UNITS = 100
# BATCH_SIZE = 256
# MAX_EPOCHS = 5
#
# input_layer = layers.Input(shape=(MAX_SENTENCE,))
#
# model = layers.Embedding(WORD_COUNT, DENSE_EMBEDDING, embeddings_initializer="uniform", input_length=MAX_SENTENCE)(input_layer)
#
# model = layers.Bidirectional(layers.LSTM(LSTM_UNITS, recurrent_dropout=LSTM_DROPOUT, return_sequences=True))(model)
#
# model = layers.TimeDistributed(layers.Dense(DENSE_UNITS, activation="relu"))(model)
#
# crf_layer = CRF(units=TAG_COUNT)
# output_layer = crf_layer(model)
#
# ner_model = Model(input_layer, output_layer)
#
# loss = losses.crf_loss
# acc_metric = metrics.crf_accuracy
# opt = optimizers.Adam(lr=0.001)
#
# ner_model.compile(optimizer=opt, loss=loss, metrics=[acc_metric])
#
# ner_model.summary()
#
# history = ner_model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=MAX_EPOCHS, validation_split=0.1, verbose=2)
#
# y_pred = ner_model.predict(X_test)
#
# y_pred = np.argmax(y_pred, axis=2)
#
# y_test = np.argmax(y_test, axis=2)
#
