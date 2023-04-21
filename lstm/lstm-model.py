import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
from sklearn.metrics import classification_report
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

data=pd.read_csv('data/new/cleaned/dataset.csv')
print(data.head())


agg_func = lambda s: [(w, t) for w, t in zip(s["words"].values.tolist(),
                                                        s["labels"].values.tolist())]


agg_data=data.groupby(['sentence_id']).apply(agg_func).reset_index().rename(columns={0:'Sentence_Tag_Pair'})
# print(agg_data.head())


agg_data['Sentence']=agg_data['Sentence_Tag_Pair'].apply(lambda sentence:" ".join([s[0] for s in sentence]))
agg_data['Tag']=agg_data['Sentence_Tag_Pair'].apply(lambda sentence:" ".join([s[1] for s in sentence]))



agg_data['tokenised_sentences']=agg_data['Sentence'].apply(lambda x:x.split())
agg_data['tag_list']=agg_data['Tag'].apply(lambda x:x.split())
print(agg_data.head())


sentences_list=agg_data['Sentence'].tolist()
tags_list=agg_data['tag_list'].tolist()

print("Number of Sentences in the Data ",len(sentences_list))
print("Are number of Sentences and Tag list equal ",len(sentences_list)==len(tags_list))

tokeniser= tf.keras.preprocessing.text.Tokenizer(lower=False,filters='')

tokeniser.fit_on_texts(sentences_list)\

encoded_sentence=tokeniser.texts_to_sequences(sentences_list)
print("First Original Sentence ",sentences_list[0])
print("First Encoded Sentence ",encoded_sentence[0])
print("Is Length of Original Sentence Same as Encoded Sentence ",len(sentences_list[0].split())==len(encoded_sentence[0]))
print("Length of First Sentence ",len(encoded_sentence[0]))

tags=list(set(data['labels'].values))
print(tags)
num_tags=len(tags)
print("Number of Tags ",num_tags)

tags_map={tag:i for i,tag in enumerate(tags)}
print("Tags Map ",tags_map)

reverse_tag_map={v: k for k, v in tags_map.items()}

encoded_tags=[[tags_map[w] for w in tag] for tag in tags_list]
print("First Sentence ",sentences_list[0])
print('First Sentence Original Tags ',tags_list[0])
print("First Sentence Encoded Tags ",encoded_tags[0])
print("Is length of Original Tags and Encoded Tags same ",len(tags_list[0])==len(encoded_tags[0]))
print("Length of Tags for First Sentence ",len(encoded_tags[0]))

tags_map={tag:i for i,tag in enumerate(tags)}
print("Tags Map ",tags_map)
#
# reverse_tag_map={v: k for k, v in tags_map.items()}

max_sentence_length=max([len(s.split()) for s in sentences_list])
print(max_sentence_length)


max_len=128


padded_encoded_sentences=pad_sequences(maxlen=max_len,sequences=encoded_sentence,padding="post",value=0)
padded_encoded_tags=pad_sequences(maxlen=max_len,sequences=encoded_tags,padding="post",value=tags_map['O'])

target= [to_categorical(i,num_classes = num_tags) for i in  padded_encoded_tags]
print("Shape of Labels  after converting to Categorical for first sentence ",target[0].shape)

from sklearn.model_selection import train_test_split
X_train,X_val_test,y_train,y_val_test = train_test_split(padded_encoded_sentences,target,test_size = 0.3,random_state=42)
X_val,X_test,y_val,y_test = train_test_split(X_val_test,y_val_test,test_size = 0.1,random_state=42)
print("Input Train Data Shape ",X_train.shape)
print("Train Labels Length ",len(y_train))
print("Input Test Data Shape ",X_test.shape)
print("Test Labels Length ",len(y_test))

print("Input Validation Data Shape ",X_val.shape)
print("Validation Labels Length ",len(y_val))


print("Shape of First Sentence -Train",X_train[0].shape)
print("Shape of First Sentence Label  -Train",y_train[0].shape)

from tensorflow.keras import Model,Input
from tensorflow.keras.layers import LSTM,Embedding,Dense
from tensorflow.keras.layers import TimeDistributed, SpatialDropout1D,Bidirectional

embedding_dim=128
vocab_size=len(tokeniser.word_index)+1
lstm_units=128
max_len=128

input_word = Input(shape = (max_len,))
model = Embedding(input_dim = vocab_size+1,output_dim = embedding_dim,input_length = max_len)(input_word)

model = LSTM(units=embedding_dim,return_sequences=True)(model)
out = TimeDistributed(Dense(num_tags,activation = 'softmax'))(model)
model = Model(input_word,out)
model.summary()

model.compile(optimizer = 'adam',loss = 'categorical_crossentropy',metrics = ['accuracy'])

history = model.fit(X_train,np.array(y_train),validation_data=(X_val,np.array(y_val)),batch_size = 32,epochs = 1)

preds=model.predict(X_test) ## Predict using model on Test Data


def evaluatePredictions(test_data, preds, actual_preds):
    print("Shape of Test Data Array", test_data.shape)
    y_actual = np.argmax(np.array(actual_preds), axis=2)
    y_pred = np.argmax(preds, axis=2)
    num_test_data = test_data.shape[0]
    print("Number of Test Data Points ", num_test_data)
    data = pd.DataFrame()
    df_list = []
    for i in range(num_test_data):
        test_str = list(test_data[i])
        df = pd.DataFrame()
        df['test_tokens'] = test_str
        df['tokens'] = df['test_tokens'].apply(lambda x: tokeniser.index_word[x] if x != 0 else '<PAD>')
        df['actual_target_index'] = list(y_actual[i])
        df['pred_target_index'] = list(y_pred[i])
        df['actual_target_tag'] = df['actual_target_index'].apply(lambda x: reverse_tag_map[x])
        df['pred_target_tag'] = df['pred_target_index'].apply(lambda x: reverse_tag_map[x])
        df['id'] = i + 1
        df_list.append(df)
    data = pd.concat(df_list)
    pred_data = data[data['tokens'] != '<PAD>']
    accuracy = pred_data[pred_data['actual_target_tag'] == pred_data['pred_target_tag']].shape[0] / pred_data.shape[0]

    return pred_data, accuracy

pred_data,accuracy=evaluatePredictions(X_test,preds,y_test)

y_pred=pred_data['pred_target_tag'].tolist()
y_actual=pred_data['actual_target_tag'].tolist()

print(classification_report(y_actual,y_pred,digits=4))