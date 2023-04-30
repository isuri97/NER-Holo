import csv
import sys

import pandas as pd

from sklearn import metrics

df_test = pd.read_csv('data/new/cleaned/gold.csv', sep='\t', quoting=csv.QUOTE_NONE, encoding='utf-8')
pred = pd.read_csv('data/new/cleaned/reg1.csv', quoting=csv.QUOTE_NONE, encoding='utf-8')



# df_test = df_test.dropna(subset=['sentence_id'])
# df_test = df_test.dropna(subset=['words'])
df_test = df_test.dropna(subset=['labels'])
#
# pred = pred.dropna(subset=['sentence_id'])
# pred = pred.dropna(subset=['words'])
pred = pred.dropna(subset=['labels'])
print(len(df_test))
print(len(pred))
df_test=df_test.loc[df_test['doc_id']!='0']
df_test=df_test.loc[df_test['doc_id']!='1']
df_test=df_test.loc[df_test['doc_id']!='3']
pred=pred.loc[pred['document_id']!=0]
pred=pred.loc[pred['document_id']!=1]
pred=pred.loc[pred['document_id']!=3]
print(len(df_test))
print(len(pred))



with open('out_new.txt', 'w') as f:
    f.write(metrics.classification_report(df_test['labels'], pred['labels'], digits=4))