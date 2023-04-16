import pandas as pd

df1 = pd.read_csv('data/testing.csv')
df1 = pd.DataFrame({'document_id': df1['sentence_id'], 'words': df1['words'], 'labels': df1['labels']})

sentence_id_list = []

sentence_id_seq = 0
for word in df1['words'].tolist():
    if word == "." or word == "?" or word == "!":
        sentence_id_list.append(sentence_id_seq)
        sentence_id_seq += 1
    else:
        sentence_id_list.append(sentence_id_seq)

df1['sentence_id'] = sentence_id_list

# print('a')

