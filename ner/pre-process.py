import pandas as pd


df1 = pd.read_csv('data/sample.txt')
unique_values_list = df1['labels'].unique().tolist()
print(unique_values_list)
print(df1.shape)
print(df1.dtypes)
print(df1.isnull().values.any())

df1 = df1.dropna()
print(df1.isnull().values.any())

print(len(df1))
print('---')

df1 = df1[~df1['words'].str.contains('\[')]
df1 = df1[~df1['words'].str.contains('\]')]
df1 = df1[~df1['words'].str.contains('\'')]
df1 = df1[~df1['words'].str.contains('\(')]
df1 = df1[~df1['words'].str.contains('\)')]
df1 = df1[~df1['words'].str.contains('-')]
df1 = df1[~df1['words'].str.contains(';')]
df1 = df1[~df1['words'].str.contains(' ')]
df1 = df1[~df1['words'].str.contains('“')]
df1 = df1[~df1['words'].str.contains('”')]
df1 = df1[~df1['words'].str.contains(',')]
df1 = df1[~df1['words'].str.contains('’')]
df1 = df1[~df1['words'].str.contains(':')]
df1 = df1[~df1['words'].str.contains(' ')]

df1.to_csv('testing.csv')
# df_train, df_test = [x for _, x in df1.groupby(df1['sentence_id'] <= 190)]
# print(len(df_train))
