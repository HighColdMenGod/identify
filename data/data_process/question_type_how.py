import pyarrow.parquet as pq
import pandas as pd

#train#

df = pd.read_parquet('/home/jyli/characteristic_identify/data/source_data/squad/plain_text/train-00000-of-00001.parquet', columns=['question'])
#df_sample = df.sample(n=200, random_state=42)
df_filter = df[df['question'].str.startswith(('How is', 'How are', 'How was', 'How were', 'How did', 'How does', 'How do', 'How has','How had', 'How have', 'How must', 'How can', 'How could', 'How might', 'How may'), na=False)].copy()
df_filter = df_filter.drop_duplicates(subset=['question'])
df_filter['label'] = 5
df_filter.to_json('/home/jyli/characteristic_identify/data/our_data/question_type/train/squad_how_train.json',orient='records',lines=True)
print(df_filter.head(5))
print(df_filter.columns.tolist())


#validate#

df = pd.read_parquet('/home/jyli/characteristic_identify/data/source_data/squad/plain_text/validation-00000-of-00001.parquet', columns=['question'])
#df_sample = df.sample(n=100, random_state=42)
df_filter = df[df['question'].str.startswith(('How is', 'How are', 'How was', 'How were', 'How did', 'How does', 'How do', 'How has','How had', 'How have', 'How must', 'How can', 'How could', 'How might', 'How may'), na=False)].copy()
df_filter = df_filter.drop_duplicates(subset=['question'])
df_filter['label'] = 5
df_filter.to_json('/home/jyli/characteristic_identify/data/our_data/question_type/validate/squad_how_validate.json',orient='records',lines=True)
print(df_filter.head(5))
print(df_filter.columns.tolist())