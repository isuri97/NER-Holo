import pandas as pd

df = pd.read_csv('data/new/cleaned/dataset.csv')
words = df['words']
tags = df['labels']

sentences = []
tags_list = []
ids = []
sentence = ""
tag_sequence = ""
for word, tag in zip(words.to_list(), tags.to_list()):
    sentence += word + " "
    tag_sequence += str(tag) + " "
    if word == "." or word == "?" or word == "!":
        sentences.append(sentence.strip())
        tags_list.append(tag_sequence.strip())
        sentence = ""
        tag_sequence = ""

dataset = pd.DataFrame()

dataset['text'] = sentences
dataset['labels'] = tags_list
df = dataset

# print(dataset)

labels = [i.split() for i in df['labels'].values.tolist()]

# Check how many labels are there in the dataset
unique_labels = set()

for lb in labels:
    [unique_labels.add(i) for i in lb if i not in unique_labels]

print(unique_labels)

labels_to_ids = {k: v for v, k in enumerate(sorted(unique_labels))}
ids_to_labels = {v: k for v, k in enumerate(sorted(unique_labels))}
print(labels_to_ids)
