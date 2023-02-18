import pandas as pd


df1 = pd.read_csv('data/testing.csv')
print(f'initial length {len(df1)}')
unique_values_list = df1['labels'].unique().tolist()

df1 = df1.dropna()

df1 = df1[~df1['words'].str.contains('\[')]
df1 = df1[~df1['words'].str.contains('\]')]
df1 = df1[~df1['words'].str.contains('\'')]
df1 = df1[~df1['words'].str.contains('\(')]
df1 = df1[~df1['words'].str.contains('\)')]
df1 = df1[~df1['words'].str.contains('-')]
df1 = df1[~df1['words'].str.contains(';')]
df1 = df1[~df1['words'].str.contains(' ')]
df1 = df1[~df1['words'].str.contains('“')]
df1 = df1[~df1['words'].str.contains('"')]
df1 = df1[~df1['words'].str.contains('”')]
df1 = df1[~df1['words'].str.contains(',')]
df1 = df1[~df1['words'].str.contains('’')]
df1 = df1[~df1['words'].str.contains(':')]
df1 = df1[~df1['words'].str.contains(' ')]

print(f'final length {len(df1)}')

df1.to_csv('data/new/testing.csv')

