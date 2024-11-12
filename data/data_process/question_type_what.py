import pyarrow.parquet as pq
import pandas as pd

#train#

df = pd.read_parquet('/home/jyli/characteristic_identify/data/source_data/squad/plain_text/train-00000-of-00001.parquet', columns=['question'])
#df_sample = df.sample(n=200, random_state=42)
df_filter = df[df['question'].str.startswith(('What is', 'What was', 'What type of', 'What sort of', 'What do', 'What does', 'What did', 'Which', 'What kind', 'Whichever'), na=False)].copy()
df_filter = df_filter[~df_filter['question'].str.contains(',')]
df_filter = df_filter.drop_duplicates(subset=['question'])
df_filter['label'] = 1
df_filter.to_json('/home/jyli/characteristic_identify/data/our_data/question_type/train/squad_what_train.json',orient='records',lines=True)
print(df_filter.head(5))
print(df_filter.columns.tolist())


#validate#

df = pd.read_parquet('/home/jyli/characteristic_identify/data/source_data/squad/plain_text/validation-00000-of-00001.parquet', columns=['question'])
#df_sample = df.sample(n=100, random_state=42)
df_filter = df[df['question'].str.startswith(('What is', 'What was', 'What type of', 'What sort of', 'What do', 'What does', 'What did', 'Which', 'What kind', 'Whichever'), na=False)].copy()
df_filter = df_filter[~df_filter['question'].str.contains(',')]
df_filter = df_filter.drop_duplicates(subset=['question'])
df_filter['label'] = 1
df_filter.to_json('/home/jyli/characteristic_identify/data/our_data/question_type/validate/squad_what_validate.json',orient='records',lines=True)
print(df_filter.head(5))
print(df_filter.columns.tolist())