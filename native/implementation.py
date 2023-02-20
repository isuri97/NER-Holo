import pandas as pd

df = pd.read_csv('data/new/cleaned/dataset.csv')
words = df['words']

sentences = []
ids = []
sentence = ""
for word, s_id in zip(words.to_list()):
    sentence += word + " "
    if word == "." or word == "?" or word == "!":
        sentences.append(sentence.strip())
        ids.append(s_id)
        sentence = ""
