import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.model_selection import train_test_split
import argparse

from simpletransformers.ner import NERModel, NERArgs

parser = argparse.ArgumentParser(
    description='''evaluates multiple models  ''')
parser.add_argument('--model_name', required=False, help='model name', default="bert")
parser.add_argument('--model_type', required=False, help='model type', default="bert-base-uncased")
parser.add_argument('--cuda_device', required=False, help='cuda device', default=0)
parser.add_argument('--train', required=False, help='train file', default='data/sample.txt')
# parser.add_argument('--test', required=False, help='test file', default='data/SOLD_test.tsv')
# parser.add_argument('--lang', required=False, help='language', default="sin")

arguments = parser.parse_args()

# # Creating train_df  and eval_df for demonstration
# train_data = [
#     [0, "Simple", "B-MISC"],
#     [0, "Transformers", "I-MISC"],
#     [0, "started", "O"],
#     [0, "with", "O"],
#     [0, "text", "O"],
#     [0, "classification", "B-MISC"],
#     [1, "Simple", "B-MISC"],
#     [1, "Transformers", "I-MISC"],
#     [1, "can", "O"],
#     [1, "now", "O"],
#     [1, "perform", "O"],
#     [1, "NER", "B-MISC"],
# ]
# train_df = pd.DataFrame(train_data, columns=["sentence_id", "words", "labels"])
#
# eval_data = [
#     [0, "Simple", "B-MISC"],
#     [0, "Transformers", "I-MISC"],
#     [0, "was", "O"],
#     [0, "built", "O"],
#     [0, "for", "O"],
#     [0, "text", "O"],
#     [0, "classification", "B-MISC"],
#     [1, "Simple", "B-MISC"],
#     [1, "Transformers", "I-MISC"],
#     [1, "then", "O"],
#     [1, "expanded", "O"],
#     [1, "to", "O"],
#     [1, "perform", "O"],
#     [1, "NER", "B-MISC"],
# ]
# eval_df = pd.DataFrame(eval_data, columns=["sentence_id", "words", "labels"])

df1 = pd.read_csv('data/sample.txt')
unique_values_list = df1['labels'].unique().tolist()
print(unique_values_list)
print(df1.shape)
print(df1.dtypes)
print(df1.isnull().values.any())

df1 = df1.dropna()
print(df1.isnull().values.any())


# using the train test split function
X_train, y_train,= train_test_split(df1,random_state=777,test_size=0.1)


train_df = pd.DataFrame(X_train, columns=["sentence_id", "words", "labels"])
eval_df = pd.DataFrame(y_train, columns=["sentence_id", "words", "labels"])


model_args = NERArgs()
model_args.labels_list = ['O', 'B-DATE', 'B-PERSON', 'B-GPE', 'B-ORG', 'I-ORG', 'B-CARDINAL', 'B-LANGUAGE',
                          'B-EVENT', 'I-DATE', 'B-NORP', 'B-TIME', 'I-TIME', 'I-GPE', 'B-ORDINAL', 'I-PERSON', 'B-MILITARY',
                          'I-MILITARY', 'I-NORP', 'B-CAMP', 'I-EVENT', 'I-CARDINAL', 'B-LAW', 'I-LAW', 'B-QUANTITY', 'B-RIVER',
                          'I-RIVER', 'B-PERCENT', 'I-PERCENT', 'B-WORK_OF_ART', 'I-QUANTITY', 'B-FAC', 'I-FAC', 'I-WORK_OF_ART',
                          'B-MONEY', 'I-MONEY', 'B-STREET', 'I-STREET', 'B-LOC', 'B-GHETTO', 'B-SEA-OCEAN', 'I-SEA-OCEAN',
                          'B-PRODUCT', 'I-CAMP', 'I-LOC', 'I-PRODUCT', 'I-GHETTO', 'B-SPOUSAL', 'I-SPOUSAL', 'B-SHIP', 'I-SHIP',
                          'B-FOREST', 'I-FOREST', 'B-GROUP', 'I-GROUP', 'B-MOUNTAIN', 'I-MOUNTAIN']

model_args.overwrite_output_dir = True
model_args.reprocess_input_data = True

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

model = NERModel(
    MODEL_TYPE, MODEL_NAME,
    args=model_args,
)
# Train the model
model.train_model(train_df)

# Evaluate the model
result, model_outputs, predictions = model.eval_model(eval_df)
print('#####')
print(result)


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