import pandas as pd
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
# parser.add_argument('--test', required=False, help='test file', default='data/SOLD_test.tsv')
# parser.add_argument('--lang', required=False, help='language', default="sin")

arguments = parser.parse_args()

df1 = pd.read_csv('data/testing.csv')

# df_train, df_test = [x for _, x in df1.groupby(df1['sentence_id'] >= 480)]
# print(len(df_train))

df_train = df1
df_test = df1

print(len(df_train))
print(len(df_test))

# concatenate words till . and add comma
words = df_test['words']
sentence_ids = df_test['sentence_id']

df_test

df_test = df_test.astype({'labels': 'string'})
print(df_test.dtypes)

sentences = []
ids = []
sentence = ""
for word, s_id in zip(words.to_list(), sentence_ids.to_list()):
    sentence += word + " "
    if word == "." or word == "?" or word == "!":
        sentences.append(sentence.strip())
        ids.append(s_id)
        sentence = ""

# sentences = " ".join(words).split(".")
# sentences = [sentence.strip() + "." for sentence in sentences if sentence.strip()]

# sentences=sentences[0:5]sentence
sentences

# using the train test split function
# X_train, y_train,= train_test_split(df1,test_size=0.1)
# df_train['labels'] = encode(df_train["labels"])
# df_test['labels'] = encode(df_test["labels"])

train_df = pd.DataFrame(df_train, columns=["sentence_id", "words", "labels"])
eval_df = pd.DataFrame(df_test, columns=["sentence_id", "words", "labels"])

model_args = {
    'train_batch_size': 32,
    'eval_batch_size': 8,
    'overwrite_output_dir':True,
    'num_train_epochs': 1,

}
# model_args.labels_list = ['O', 'B-DATE', 'B-PERSON', 'B-GPE', 'B-ORG', 'I-ORG', 'B-CARDINAL', 'B-LANGUAGE',
#                           'B-EVENT', 'I-DATE', 'B-NORP', 'B-TIME', 'I-TIME', 'I-GPE', 'B-ORDINAL', 'I-PERSON', 'B-MILITARY',
#                           'I-MILITARY', 'I-NORP', 'B-CAMP', 'I-EVENT', 'I-CARDINAL', 'B-LAW', 'I-LAW', 'B-QUANTITY', 'B-RIVER',
#                           'I-RIVER', 'B-PERCENT', 'I-PERCENT', 'B-WORK_OF_ART', 'I-QUANTITY', 'B-FAC', 'I-FAC', 'I-WORK_OF_ART',
#                           'B-MONEY', 'I-MONEY', 'B-STREET', 'I-STREET', 'B-LOC', 'B-GHETTO', 'B-SEA-OCEAN', 'I-SEA-OCEAN',
#                           'B-PRODUCT', 'I-CAMP', 'I-LOC', 'I-PRODUCT', 'I-GHETTO', 'B-SPOUSAL', 'I-SPOUSAL', 'B-SHIP', 'I-SHIP',
#                           'B-FOREST', 'I-FOREST', 'B-GROUP', 'I-GROUP', 'B-MOUNTAIN', 'I-MOUNTAIN']


# models = {'bert': 'bert-base-uncased', 'roberta': 'roberta-base', 'xlnet': 'xlnet-base-cased'}
#
# # Create a NERModel
# for model_type, model_name in models.items():
#     model = NERModel(model_type, model_name,  args=model_args)
#     model.train_model(train_df)
#     # Evaluate the model
#     result, model_outputs, predictions = model.eval_model(eval_df)
#     print('#####')
#     print(result)

MODEL_NAME = arguments.model_name
MODEL_TYPE = arguments.model_type
cuda_device = int(arguments.cuda_device)
# MODEL_TYPE, MODEL_NAME,
model = NERModel(
    MODEL_TYPE, MODEL_NAME ,
    use_cuda=cuda_device,
    args=model_args,
    labels=['O', 'B-DATE', 'B-PERSON', 'B-GPE', 'B-ORG', 'I-ORG', 'B-CARDINAL', 'B-LANGUAGE',
            'B-EVENT', 'I-DATE', 'B-NORP', 'B-TIME', 'I-TIME', 'I-GPE', 'B-ORDINAL', 'I-PERSON', 'B-MILITARY',
            'I-MILITARY', 'I-NORP', 'B-CAMP', 'I-EVENT', 'I-CARDINAL', 'B-LAW', 'I-LAW', 'B-QUANTITY', 'B-RIVER',
            'I-RIVER', 'B-PERCENT', 'I-PERCENT', 'B-WORK_OF_ART', 'I-QUANTITY', 'B-FAC', 'I-FAC', 'I-WORK_OF_ART',
            'B-MONEY', 'I-MONEY', 'B-STREET', 'I-STREET', 'B-LOC', 'B-GHETTO', 'B-SEA-OCEAN', 'I-SEA-OCEAN',
            'B-PRODUCT', 'I-CAMP', 'I-LOC', 'I-PRODUCT', 'I-GHETTO', 'B-SPOUSAL', 'I-SPOUSAL', 'B-SHIP', 'I-SHIP',
            'B-FOREST', 'I-FOREST', 'B-GROUP', 'I-GROUP', 'B-MOUNTAIN', 'I-MOUNTAIN']
)
# Train the model
model.train_model(train_df)
# model.save_model(output_dir='outputs/saved_model/')

# Evaluate the model
# result, model_outputs, predictions = model.eval_model(eval_df)
# print('#####')
# print(result)

predictions, outputs = model.predict(sentences)

# predictions = []
# with open('filex.txt', 'a') as f:
#     for h, id in zip(sentences, ids):
#         sentence_lst = []
#         sentence_lst.append(h)
#         preds, outputs = model.predict(sentence_lst)
#         predictions.append(preds)
#         # if len(df_test.groupby('sentence_id').get_group(id)) != len(predictions):
#         if len(h.split(' ')) != len(preds[0]):
#             f.write(h)
#             print(h)
# print(len(preds))
#
# print(len(eval_df['labels']))
# pred_list = []



# for pred in predictions:
#     for tag in pred:
#         pred_list.append(tag.values())
#
#
# lab=eval_df['labels'].tolist()
# for i in range(0,len(pred_list)):
#     if pred_list[i]!=lab[i]:
#         print(f'not matched index {i}, preadval = {predictions[i]}, gold val = {lab[i]}')


ll = []
key_list = []
# for i in predictions:
#     for lst in i[0]:
#         # for (k,v) in lst:
#         ll.append(lst.values())
#         # ll += list(lst.values())
#         # key_list += list(lst.keys())
#         key_list.append(lst.keys())

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
# new2['w'].to_csv('word-list.csv')


# for word in words:
#     w_list.append(word)


missing_words = set(w_list) - set(key_list)
print(missing_words)

result = list((Counter(w_list) - Counter(key_list)).elements())
print(result)
#
# missing_indexes_and_words = []
# for i, word in enumerate(words):
#     if word not in key_list:
#         missing_indexes_and_words.append((i, word))
#
# print(len(missing_indexes_and_words))
# new = pd.DataFrame()
# new['sub'] = missing_indexes_and_words
#
# new['sub'].to_csv('te.csv')
#
# eval_df['labels'] = decode(eval_df['labels'])
# eval_df['predictions'] = decode(eval_df['predictions'])

# print_information(eval_df,'predictions','labels')

eval_df['predictions'] = ll

y_true = eval_df['labels']
y_pred = eval_df['predictions']

# print(metrics.confusion_matrix(y_true, y_pred))
print(metrics.classification_report(y_true, y_pred, digits=7))

# print_information_multi_class(eval_df,'predictions','labels')
# sentences = ["Some arbitary sentence", "Simple Transformers sentence"]
# predictions, raw_outputs = model.predict(sentences)
# #
# # Predictions on arbitary text strings
# sentences = ["Some arbitary sentence", "Simple Transformers sentence"]
# predictions, raw_outputs = model.predict(sentences)
#
# print(predictions)
#
# # More detailed preditctions
# for n, (preds, outs) in enumerate(zip(predictions, raw_outputs)):
#     print("\n___________________________")
#     print("Sentence: ", sentences[n])
#     for pred, out in zip(preds, outs):
#         key = list(pred.keys())[0]
#         new_out = out[key]
#         preds = list(softmax(np.mean(new_out, axis=0)))
#         print(key, pred[key], preds[np.argmax(preds)], preds)
