import pandas as pd
import torch.cuda
from sklearn import metrics
from sklearn.model_selection import train_test_split
import argparse
from collections import Counter

from simpletransformers.ner import NERModel, NERArgs
import csv

from contextlib import redirect_stdout

parser = argparse.ArgumentParser(
    description='''evaluates multiple models  ''')
parser.add_argument('--model_name', required=False, help='model name', default="bert")
parser.add_argument('--model_type', required=False, help='model type', default="bert-base-cased")
parser.add_argument('--cuda_device', required=False, help='cuda device', default=0)
parser.add_argument('--train', required=False, help='train file', default='data/sample.txt')

arguments = parser.parse_args()

df_train= pd.read_csv('data/new/cleaned/together.csv', sep='\t', error_bad_lines=False,quoting=csv.QUOTE_NONE, encoding='utf-8')
df_test = pd.read_csv('data/new/cleaned/gold.csv', sep = '\t')
df_test.dropna(subset=['labels'],inplace=True)



# df1 = pd.DataFrame({'document_id': df_train['document_id'], 'words': df_test['words'], 'labels': df_train['labels']})
#
# sentence_id_list = []
#
# sentence_id_seq = 0
#
# dropping_sentences = []
#
# for word in df1['words'].tolist():
#     if word == "." or word == "?" or word == "!":
#         sentence_id_list.append(sentence_id_seq)
#         sentence_id_seq += 1
#         word_count = 0
#     else:
#         sentence_id_list.append(sentence_id_seq)
#
# df1['sentence_id'] = sentence_id_list

# sentence_ids = list(set(sentence_id_list))

# sentence_ids_train, sentence_ids_test = train_test_split(sentence_ids, test_size=0.3)
# df2 = df1[df1['sentence_id'] not in dropping_sentences]

# df_train, df_test = [x for _, x in df1.groupby(df1['sentence_id'] >= 400)]

# df_train = df1[df1["sentence_id"].isin(sentence_ids_train)]
# df_test = df1[df1["sentence_id"].isin(sentence_ids_test)]  # train_test_split(df1, test_size=0.1)



print(f'training set size {len(df_train)}')
print(f'test set size {len(df_test)}')

# concatenate words till . and add comma
# words = df_test['words']
# sentence_ids = df_test['sentence_id']

# df_test = df_test.astype({'labels': 'string'})

# sentences = []
# ids = []
# sentence = ""
# word_count = 0
# for word, s_id in zip(words.to_list(), sentence_ids.to_list()):
#     word_count += 1
#     sentence += word + " "
#     if word == "." or word == "?" or word == "!":
#         sentences.append(sentence.strip())
#         ids.append(s_id)
#         word_count = 0
#         sentence = ""

model_args = NERArgs()
model_args.train_batch_size = 64
model_args.eval_batch_size = 64
model_args.overwrite_output_dir = True
model_args.num_train_epochs = 1
model_args.use_multiprocessing = False
model_args.use_multiprocessing_for_evaluation = False
model_args.classification_report = True
# model_args.wandb_project="holo-ner"
model_args.labels_list = ['O', 'B-DATE', 'B-PERSON', 'B-GPE', 'B-ORG', 'I-ORG', 'B-LANGUAGE',
                          'B-EVENT', 'I-DATE',  'B-TIME', 'I-TIME', 'I-GPE','I-PERSON',
                          'B-MILITARY','I-MILITARY','B-CAMP', 'I-EVENT', 'I-CARDINAL', 'B-LAW', 'I-LAW',
                          'B-RIVER','I-RIVER','I-QUANTITY', 'B-STREET', 'I-STREET', 'B-LOC', 'B-GHETTO', 'B-SEA-OCEAN',
                          'I-SEA-OCEAN','I-CAMP', 'I-LOC',  'I-GHETTO', 'B-SPOUSAL', 'I-SPOUSAL', 'B-SHIP',
                          'I-SHIP', 'B-FOREST', 'I-FOREST', 'B-GROUP', 'I-GROUP', 'B-MOUNTAIN', 'I-MOUNTAIN']

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
# print(results)
preds_list = [tag for s in preds_list for tag in s]
ll = []
key_list = []

# df_test['original_test_set'] = truths
# df_test['predicted_set'] = preds

# take the label and count is it match with

labels = ['B-SHIP', 'I-SHIP', 'B-GHETTO', 'I-GHETTO', 'B-STREET', 'I-STREET', 'B-MILITARY', 'I-MILITARY', 'B-DATE',
          'I-DATE', 'B-PERSON', 'I-PERSON',
          'B-GPE', 'I-GPE', 'B-TIME', 'I-TIME', 'B-EVENT', 'I-EVENT', 'B-ORG', 'I-ORG', 'B-TIME', 'I-TIME']

# for i in predictions:
#     for h in i:
#         for v in h.values():
#             ll.append(v)
# for i in predictions:
#     for h in i:
#         for v in h.keys():
#             key_list.append(v)

# df_test["predictions"] = preds_list

# pred_stats = pd.DataFrame()
# pred_stats['words'] = key_list
# pred_stats['tags'] = ll
#
# df1.to_csv('dataset.csv', index=False)
# pred_stats.to_csv('prediction_stats.csv', index=False)
#
# df_test['predictions'] = ll
# df_test.to_csv("predictions.csv", index=False)
# y_true = df_test['labels']
# y_pred = df_test['predictions']

# print(metrics.confusion_matrix(y_true, y_pred))
# with open('metrics.txt', 'w') as f:
#    f.write(metrics.classification_report(y_true, y_pred, digits=7))

# ct = len(truths)
# pt = len(preds)
#
# print(ct)
# print(preds)

print(metrics.classification_report(truths, preds, digits=4))

# new_df = pd.DataFrame({'truthset': truths, 'predset': preds})
#
# print(new_df)
#
#
#
# truth_set = new_df['truthset']
# predicted_set = new_df['predset']

#
# tr_set = set()
# confusion_dict = {}  # {org:{org:count,per:count}}
#
# for t, p in zip(truth_set, predicted_set):
#     if tr_set.__contains__(t):
#         values_dict = confusion_dict[t]
#     else:
#         values_dict = dict()
#         confusion_dict[t] = values_dict
#     tr_set.add(t)
#
#     if values_dict.keys().__contains__(p):
#         values_dict[p] = values_dict[p] + 1
#     else:
#         values_dict[p] = 1
#
#
# print(confusion_dict)
#
# lst = ['O','B-CAMP', 'I-CAMP', 'B-SHIP', 'I-SHIP','B-GHETTO', 'I-GHETTO', 'B-PERSON', 'I-PERSON', 'B-STREET', 'I-STREET', 'B-DATE', 'I-DATE',\
# 'B-GPE', 'I-GPE', 'B-TIME', 'I-TIME', 'B-EVENT', 'I-EVENT','B-MILITARY', 'I-MILITARY', 'B-ORG', 'I-ORG' ]
#
# final_list = []
# for tag in lst:
#     preds_dict = confusion_dict[tag]
#     print(preds_dict)
#     new_list = []
#     for i in lst:
#         if preds_dict.keys().__contains__(i):
#             count = preds_dict[i]
#         else:
#             count = 0
#         new_list.append(count)
#     final_list.append(new_list)
#
# print(final_list)
#
#
#
