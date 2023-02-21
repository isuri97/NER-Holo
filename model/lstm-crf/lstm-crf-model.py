import pandas as pd

import pickle
import operator
import pandas as pd
import keras.backend as k
import numpy as np
import matplotlib.pyplot as plt
#
# from plot_keras_history import plot_history
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix
from keras_contrib.utils import save_load_utils

#
from keras.models import Model, Sequential
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras_contrib.layers import CRF
import keras as k

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

sentence_ids = list(set(sentence_id_list))
sentence_ids_train, sentence_ids_test = train_test_split(sentence_ids, test_size=0.3)
# df2 = df1[df1['sentence_id'] not in dropping_sentences]

# df_train, df_test = [x for _, x in df1.groupby(df1['sentence_id'] >= 400)]

df_train = df1[df1["sentence_id"].isin(sentence_ids_train)]
df_test = df1[df1["sentence_id"].isin(sentence_ids_test)]#train_test_split(df1, test_size=0.1)

print(f'training set size {len(df_train)}')
print(f'test set size {len(df_test)}')


# df_train

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

# test_word = "Scotland"
#
# test_word_idx = word2index[test_word]
# test_word_lookup = index2word[test_word_idx]
#
# print("The index of the word {} is {}.".format(test_word, test_word_idx))
# print("The word with index {} is {}.".format(test_word_idx, test_word_lookup))

tag2index = {tag: idx + 1 for idx, tag in enumerate(all_tags)}
tag2index["--PADDING--"] = 0

index2tag = {idx: word for word, idx in tag2index.items()}

# def to_tuples(data):
#     iterator = zip(df1["words"].values.tolist(),
#                    df1["labels"].values.tolist())
#     return [(word, tag) for word, tag in iterator]
#
# sentences = df1.groupby("sentence_id").apply(to_tuples).tolist()
#
# print(sentences[0])

# words = df1['words']
# lab = df1['labels']
# sentences = []
# lt = []
# sentence = ""
# lab_list=""
#
# for word, label in zip(words.to_list(), lab.to_list()):
#     sentence += word + ","
#     lab_list += label + ','
#     if word == "." or word == "?" or word == "!":
#         sentences.append(sentence.strip())
#         lt.append(lab_list)
#         sentence = ""
#         lab_list = ""

sentences = df1.groupby("sentence_id")[["words", "labels"]].agg(list)

# print(sentences.iloc[0])

X = sentences['words']
y = sentences['labels']

print(X[0])
print(y[0])


X = [[word2index[word] for word in sentence] for sentence in X]
y = [[tag2index[tag] for tag in sentence] for sentence in y]
print("X[0]:", X[0])
print("y[0]:", y[0])
#

#
X = [sentence + [word2index["--PADDING--"]] * (MAX_SENTENCE - len(sentence)) for sentence in X]
y = [sentence + [tag2index["--PADDING--"]] * (MAX_SENTENCE - len(sentence)) for sentence in y]
print("X[0]:", X[0])
print("y[0]:", y[0])
#
TAG_COUNT = len(tag2index)
y = [ np.eye(TAG_COUNT)[sentence] for sentence in y]
print("X[0]:", X[0])
print("y[0]:", y[0])
#
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1234)

print("Number of sentences in the training dataset: {}".format(len(X_train)))
print("Number of sentences in the test dataset : {}".format(len(X_test)))
#
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

WORD_COUNT = len(index2word)
DENSE_EMBEDDING = 50
LSTM_UNITS = 50
LSTM_DROPOUT = 0.1
DENSE_UNITS = 100
BATCH_SIZE = 128
MAX_EPOCHS = 30
#


from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras_contrib.layers import CRF
# Model definition
# input = Input(shape=(MAX_SENTENCE,))
# model = Embedding(input_dim=WORD_COUNT+2, output_dim=DENSE_EMBEDDING, # n_words + 2 (PAD & UNK)
#                   input_length=MAX_SENTENCE, mask_zero=True)(input)  # default: 20-dim embedding
# model = Bidirectional(LSTM(units=50, return_sequences=True,
#                            recurrent_dropout=0.1))(model)  # variational biLSTM
# model = TimeDistributed(Dense(50, activation="relu"))(model)  # a dense layer as suggested by neuralNer
# crf = CRF(WORD_COUNT+1)  # CRF layer, n_tags+1(PAD)
# out = crf(model)  # output
# model = Model(input, out)
# model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])
# model.summary()

#
# input_layer = layers.Input(shape=(MAX_SENTENCE,))
# model = layers.Embedding(WORD_COUNT, DENSE_EMBEDDING, embeddings_initializer="uniform", input_length=MAX_SENTENCE)(input_layer)
# model = layers.Bidirectional(layers.LSTM(LSTM_UNITS, recurrent_dropout=LSTM_DROPOUT, return_sequences=True))(model)
# model = layers.TimeDistributed(layers.Dense(DENSE_UNITS, activation="relu"))(model)
# crf_layer = CRF(units=TAG_COUNT)
# output_layer = crf_layer(model)
# ner_model = Model(input_layer, output_layer)
# loss = losses.crf_loss
# acc_metric = metrics.crf_accuracy
# opt = optimizers.Adam(lr=0.001)
# ner_model.compile(optimizer=opt, loss=loss, metrics=[acc_metric])
# ner_model.summary()


model = Sequential()
model.add(Embedding(input_dim=WORD_COUNT, output_dim=200, input_length=MAX_SENTENCE))
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(units=128, return_sequences=True, recurrent_dropout=0.1)))
model.add(TimeDistributed(Dense(DENSE_UNITS, activation="relu")))
crf_layer = CRF(units=TAG_COUNT)
model.add(crf_layer)
model.summary()

model.compile(optimizer='adam', loss=crf_layer.loss_function, metrics=[crf_layer.accuracy])


history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=MAX_EPOCHS, validation_split=0.1, verbose=2)

# plot_history(history.history)
#
# y_pred = model.predict(X_test)
#
# y_pred = np.argmax(y_pred, axis=2)
#
# y_test = np.argmax(y_test, axis=2)
# accuracy = (y_pred == y_test).mean()

# print("Accuracy: {:.4f}/".format(accuracy))
# #

pred_cat = model.predict(X_test)
pred = np.argmax(pred_cat, axis=-1)
y_te_true = np.argmax(y_test, -1)

from sklearn_crfsuite.metrics import flat_classification_report
# Convert the index to tag
pred_tag = [[index2tag[i] for i in row] for row in pred]
y_te_true_tag = [[index2tag[i] for i in row] for row in y_te_true]

report = flat_classification_report(y_pred=pred_tag, y_true=y_te_true_tag)
print(report)

with open('metrics.txt', 'w') as f:
   f.write(flat_classification_report(y_pred=pred_tag, y_true=y_te_true_tag, digit=4))


# Plot training & validation accuracy values
plt.plot(history.history['crf_viterbi_accuracy'])
plt.plot(history.history['val_crf_viterbi_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'])
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
