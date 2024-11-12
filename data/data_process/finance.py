
import json
import random

# 读取包含数据的 JSON 文件
with open('/home/jyli/characteristic_identify/data/source_data/field/ConvFinQA-finance/test_private.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 提取符合条件的字符串（长度在 70 到 100 之间）
filtered_texts = []
for entry in data:
    if 'pre_text' in entry:  # 确保每个 entry 中包含 'pre_text' 键
        filtered_texts.extend([text for text in entry['pre_text'] if 100 <= len(text) <= 140])

# 检查是否有足够的数据来随机抽样
if len(filtered_texts) < 200:
    print(f"符合条件的文本数量不足 200，实际数量为 {len(filtered_texts)}。请调整筛选条件或减少输出数量。")
else:
    # 随机选取 200 条数据
    sampled_texts = random.sample(filtered_texts, 100)

    # 创建新的数据结构，并添加 label 字段
    output_data = [{'context': text, 'label': 1} for text in sampled_texts]

    # 保存到新的 JSON 文件
    output_file_path = '/home/jyli/characteristic_identify/data/our_data/field/validate1.json'
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    print(f"保存成功！文件路径：{output_file_path}")
