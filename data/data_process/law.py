import pandas as pd
import json
import random

# 读取 Parquet 文件
parquet_file_path = '/home/jyli/characteristic_identify/data/source_data/field/LEDGAR/data/train-00000-of-00001.parquet'  # 替换为你的 Parquet 文件路径
df = pd.read_parquet(parquet_file_path)

# 筛选出 context 字段长度在 10 到 20 的项
filtered_df = df[df['text'].apply(lambda x: 100 <= len(x) <= 140)]

# 随机选取 200 条数据
sampled_df = filtered_df.sample(n=200, random_state=42)

# 添加 label 字段值为 0
sampled_df['label'] = 2

sampled_df = sampled_df.rename(columns={'text': 'context'})

# 将数据保存为 JSON 文件
output_data = sampled_df[['context', 'label']].to_dict(orient='records')  
json_file_path = '/home/jyli/characteristic_identify/data/our_data/field/train2.json'
with open(json_file_path, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=4)

print(f"保存成功！文件路径：{json_file_path}")
