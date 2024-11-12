import json
import matplotlib.pyplot as plt
from collections import Counter

# 文件路径
file_path = '/home/jyli/characteristic_identify/data/source_data/intention/medical_intention/KUAKE-QIC/KUAKE-QIC/KUAKE-QIC_train.json'

# 读取 JSON 文件
with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 统计 text 字段的长度
text_lengths = [len(entry['query']) for entry in data if 'query' in entry]

# 输出一些统计信息
print(f"文本的最大长度: {max(text_lengths)}")
print(f"文本的最小长度: {min(text_lengths)}")
print(f"文本的平均长度: {sum(text_lengths) / len(text_lengths):.2f}")

# 绘制长度分布的直方图
plt.figure(figsize=(10, 6))
plt.hist(text_lengths, bins=30, color='skyblue', edgecolor='black')
plt.xlabel('Text Length')
plt.ylabel('Frequency')
plt.title('Distribution of Text Lengths in JSON File')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
