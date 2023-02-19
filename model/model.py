import pandas as pd
import torch.cuda
from sklearn import metrics
from sklearn.model_selection import train_test_split
import argparse
from collections import Counter

from simpletransformers.ner import NERModel, NERArgs

from contextlib import redirect_stdout

parser = argparse.ArgumentParser(
    description='''evaluates multiple models  ''')
parser.add_argument('--model_name', required=False, help='model name', default="bert")
parser.add_argument('--model_type', required=False, help='model type', default="bert-base-cased")
parser.add_argument('--cuda_device', required=False, help='cuda device', default=0)
parser.add_argument('--train', required=False, help='train file', default='data/sample.txt')

arguments = parser.parse_args()

df1 = pd.read_csv('data/new/testing.csv')
df1 = pd.DataFrame({'sentence_id': df1['sentence_id'], 'words': df1['words'], 'labels': df1['labels']})

# df_train, df_test = [x for _, x in df1.groupby(df1['sentence_id'] >= 400)]

df_train = df1
df_test = df1

print(f'training set size {len(df_train)}')
print(f'test set size {len(df_test)}')

# concatenate words till . and add comma
words = df_test['words']
sentence_ids = df_test['sentence_id']

df_test = df_test.astype({'labels': 'string'})

sentences = []
ids = []
sentence = ""
for word, s_id in zip(words.to_list(), sentence_ids.to_list()):
    sentence += word + " "
    if word == "." or word == "?" or word == "!":
        sentences.append(sentence.strip())
        ids.append(s_id)
        sentence = ""

# parity check
total_count = 0
for sentence in sentences:
    total_count += len(sentence.split(' '))

print(f'parity number is {total_count} and actual number is {len(words)}')

model_args = {
    'train_batch_size': 64,
    'eval_batch_size': 8,
    'overwrite_output_dir': True,
    'num_train_epochs': 1,
    'use_multiprocessing': False,
    'use_multiprocessing_for_evaluation': False,
}

MODEL_NAME = arguments.model_name
MODEL_TYPE = arguments.model_type
cuda_device = int(arguments.cuda_device)
# MODEL_TYPE, MODEL_NAME,
model = NERModel(
    MODEL_TYPE, MODEL_NAME,
    use_cuda=torch.cuda.is_available(),
    cuda_device=cuda_device,
    args=model_args,
    labels=['O', 'B-DATE', 'B-PERSON', 'B-GPE', 'B-ORG', 'I-ORG', 'B-CARDINAL', 'B-LANGUAGE',
            'B-EVENT', 'I-DATE', 'B-NORP', 'B-TIME', 'I-TIME', 'I-GPE', 'B-ORDINAL', 'I-PERSON', 'B-MILITARY',
            'I-MILITARY', 'I-NORP', 'B-CAMP', 'I-EVENT', 'I-CARDINAL', 'B-LAW', 'I-LAW', 'B-QUANTITY', 'B-RIVER',
            'I-RIVER', 'B-PERCENT', 'I-PERCENT', 'B-WORK_OF_ART', 'I-QUANTITY', 'B-FAC', 'I-FAC', 'I-WORK_OF_ART',
            'B-MONEY', 'I-MONEY', 'B-STREET', 'I-STREET', 'B-LOC', 'B-GHETTO', 'B-SEA-OCEAN', 'I-SEA-OCEAN',
            'B-PRODUCT', 'I-CAMP', 'I-LOC', 'I-PRODUCT', 'I-GHETTO', 'B-SPOUSAL', 'I-SPOUSAL', 'B-SHIP', 'I-SHIP',
            'B-FOREST', 'I-FOREST', 'B-GROUP', 'I-GROUP', 'B-MOUNTAIN', 'I-MOUNTAIN']
)

df_train, df_eval = train_test_split(df_train, test_size=0.2, random_state=777)
# Train the model
model.train_model(df_train, eval_df=df_eval)

predictions, outputs = model.predict(sentences)


ll = []
key_list = []

for i in predictions:
    for h in i:
        for v in h.values():
            ll.append(v)
for i in predictions:
    for h in i:
        for v in h.keys():
            key_list.append(v)

new1 = pd.DataFrame()
new1['w'] = sentences
new1['w'].to_csv('pred_sent-list.csv')

print('------')
print(len(ll))
print('======')
# print(key_list)

w_list = words.to_list()
new2 = pd.DataFrame()
new2['w'] = sentences
new2['w'].to_csv('word-sent.csv')

df_test['predictions'] = ll

y_true = df_test['labels']
y_pred = df_test['predictions']

# print(metrics.confusion_matrix(y_true, y_pred))
with open('metrics.txt', 'w') as f:
    f.write(metrics.classification_report(y_true, y_pred, digits=7))
