import pandas as pd
import json

# 读取 Parquet 文件
parquet_file_path = '/home/jyli/characteristic_identify/data/source_data/field/agriculture/data/train-00000-of-00001.parquet'  
df = pd.read_parquet(parquet_file_path)

# 筛选出 context 字段长度在 100 到 140 的项
filtered_df = df[df['answers'].apply(lambda x: 100 <= len(x) <= 140)]

# 随机选取 300 条数据
sampled_df = filtered_df.sample(n=300, random_state=42)

# 添加 label 字段值为 0
sampled_df['label'] = 3

# 重命名 'text' 列为 'context'
sampled_df = sampled_df.rename(columns={'answers': 'context'})

# 将数据分为前200项和后100项
first_200 = sampled_df.head(200)
last_100 = sampled_df.tail(100)

# 将前200项保存为第一个 JSON 文件
first_200_output_data = first_200[['context', 'label']].to_dict(orient='records')
first_json_file_path = '/home/jyli/characteristic_identify/data/our_data/field/train3.json'
with open(first_json_file_path, 'w', encoding='utf-8') as f:
    json.dump(first_200_output_data, f, ensure_ascii=False, indent=4)

# 将后100项保存为第二个 JSON 文件
last_100_output_data = last_100[['context', 'label']].to_dict(orient='records')
second_json_file_path = '/home/jyli/characteristic_identify/data/our_data/field/validate3.json'
with open(second_json_file_path, 'w', encoding='utf-8') as f:
    json.dump(last_100_output_data, f, ensure_ascii=False, indent=4)

print(f"保存成功！前200项文件路径：{first_json_file_path}")
print(f"保存成功！后100项文件路径：{second_json_file_path}")
