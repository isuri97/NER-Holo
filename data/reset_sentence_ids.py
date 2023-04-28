import pandas as pd
import csv


df1 = pd.read_csv('new/cleaned/gold4.csv', sep='\t',quoting=csv.QUOTE_NONE, encoding='utf-8')
df1

sentence_id_list = []

sentence_id_seq = 81356
for word in df1['words'].tolist():
    if word == "." or word == "?" or word == "!":
        sentence_id_list.append(sentence_id_seq)
        sentence_id_seq += 1
    else:
        sentence_id_list.append(sentence_id_seq)

df1['sentence_id'] = sentence_id_list
# df1 = pd.DataFrame({'document_id': df1['sentence_id'], 'words': df1['words'], 'labels': df1['labels']})

df1.to_csv('train51.csv',index=False,sep=',')
# print('a')




