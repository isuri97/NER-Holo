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

df1 = pd.read_csv('data/new/cleaned/dataset.csv')
df1 = pd.DataFrame({'document_id': df1['document_id'], 'words': df1['words'], 'labels': df1['labels']})

sentence_id_list = []

sentence_id_seq = 0

dropping_sentences = []

for word in df1['words'].tolist():
    if word == "." or word == "?" or word == "!":
        sentence_id_list.append(sentence_id_seq)
        sentence_id_seq += 1
        word_count = 0
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

# concatenate words till . and add comma
words = df_test['words']
sentence_ids = df_test['sentence_id']

#df_test = df_test.astype({'labels': 'string'})

sentences = []
ids = []
sentence = ""
word_count = 0
for word, s_id in zip(words.to_list(), sentence_ids.to_list()):
    word_count += 1
    sentence += word + " "
    if word == "." or word == "?" or word == "!":
        sentences.append(sentence.strip())
        ids.append(s_id)
        word_count = 0
        sentence = ""

model_args = NERArgs()
model_args.train_batch_size = 64
model_args.eval_batch_size = 64
model_args.overwrite_output_dir = True
model_args.num_train_epochs = 1
model_args.use_multiprocessing = False
model_args.use_multiprocessing_for_evaluation = False
model_args.classification_report = True
model_args.wandb_project="holo-ner"
model_args.labels_list = ['O', 'B-DATE', 'B-PERSON', 'B-GPE', 'B-ORG', 'I-ORG', 'B-CARDINAL', 'B-LANGUAGE',
                          'B-EVENT', 'I-DATE', 'B-NORP', 'B-TIME', 'I-TIME', 'I-GPE', 'B-ORDINAL', 'I-PERSON',
                          'B-MILITARY',
                          'I-MILITARY', 'I-NORP', 'B-CAMP', 'I-EVENT', 'I-CARDINAL', 'B-LAW', 'I-LAW', 'B-QUANTITY',
                          'B-RIVER',
                          'I-RIVER', 'B-PERCENT', 'I-PERCENT', 'B-WORK_OF_ART', 'I-QUANTITY', 'B-FAC', 'I-FAC',
                          'I-WORK_OF_ART',
                          'B-MONEY', 'I-MONEY', 'B-STREET', 'I-STREET', 'B-LOC', 'B-GHETTO', 'B-SEA-OCEAN',
                          'I-SEA-OCEAN',
                          'B-PRODUCT', 'I-CAMP', 'I-LOC', 'I-PRODUCT', 'I-GHETTO', 'B-SPOUSAL', 'I-SPOUSAL', 'B-SHIP',
                          'I-SHIP',
                          'B-FOREST', 'I-FOREST', 'B-GROUP', 'I-GROUP', 'B-MOUNTAIN', 'I-MOUNTAIN']

MODEL_NAME = arguments.model_name
MODEL_TYPE = arguments.model_type
cuda_device = int(arguments.cuda_device)
# MODEL_TYPE, MODEL_NAME,
model = NERModel(
    MODEL_TYPE, MODEL_NAME,
    use_cuda=torch.cuda.is_available(),
    cuda_device=cuda_device,
    args=model_args,
)

# df_train, df_eval = train_test_split(df_train, test_size=0.2, random_state=777)
# Train the model


model.train_model(df_train)
model.save_model()

# predictions, outputs = model.predict(sentences)

print(len(df_test))
results, outputs, preds_list, truths, preds = model.eval_model(df_test)
print(results)
preds_list = [tag for s in preds_list for tag in s]
ll = []
key_list = []

# for i in predictions:
#     for h in i:
#         for v in h.values():
#             ll.append(v)
# for i in predictions:
#     for h in i:
#         for v in h.keys():
#             key_list.append(v)

#df_test["predictions"] = preds_list

# pred_stats = pd.DataFrame()
# pred_stats['words'] = key_list
# pred_stats['tags'] = ll
#
# df1.to_csv('dataset.csv', index=False)
# pred_stats.to_csv('prediction_stats.csv', index=False)
#
# df_test['predictions'] = ll
#df_test.to_csv("predictions.csv", index=False)
#y_true = df_test['labels']
#y_pred = df_test['predictions']

#print(metrics.confusion_matrix(y_true, y_pred))
#with open('metrics.txt', 'w') as f:
#    f.write(metrics.classification_report(y_true, y_pred, digits=7))

print(metrics.classification_report(truths,preds,digits=4))