import json
import random
from collections import defaultdict

# 读取包含数据的 JSON 文件
with open('/home/jyli/characteristic_identify/data/source_data/intention/medical_intention/KUAKE-QIC/KUAKE-QIC/KUAKE-QIC_dev.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

label_map = {
    '病情诊断': 0,
    '治疗方案': 1,
    '就医建议': 2,
    '疾病表述': 3,
    '注意事项': 4,
}

# 提取符合条件的字符串（长度在 10 到 20 之间）
filtered_texts = []

for entry in data:
    if 'query' in entry and 'label' in entry and entry['label'] in label_map:  
        text = entry['query']
        label = entry['label']
        # 检查字符串的长度是否在 10 到 20 个字符之间
        if 8 <= len(text) <= 18:
            filtered_texts.append((text, label))

# 初始化字典用于按标签存储筛选后的数据
label_data = defaultdict(list)

# 遍历筛选后的数据，将其按标签存储
for text, label in filtered_texts:
    label_data[label].append((text, label))

# 按每个标签随机选择 200 条数据
final_texts = []
for label, texts in label_data.items():
    if len(texts) >= 90:
        sampled_texts = random.sample(texts, 90)
    else:
        print(f"标签 '{label}' 的数据不足 90 条，实际数量为 {len(texts)}，将全部使用。")
        sampled_texts = texts
    final_texts.extend(sampled_texts)

#打乱顺序
random.shuffle(final_texts)

# 输出符合条件的文本数量以及每个 label 的统计信息
print(f"最终选择的文本数量: {len(final_texts)}")
for label in label_map.keys():
    count = sum(1 for _, lbl in final_texts if lbl == label)
    print(f"标签 '{label}' 的数量: {count}")

# 保存到新的 JSON 文件
output_data = [{'context': text, 'label': label_map.get(label)} for text, label in final_texts]
output_file_path = '/home/jyli/characteristic_identify/data/our_data/intention/validate.json'

with open(output_file_path, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=4)

print(f"保存成功！文件路径：{output_file_path}")
