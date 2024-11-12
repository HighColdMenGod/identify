import json
import random

# 输入的8个JSON文件路径
input_files = [
    '/home/jyli/characteristic_identify/data/our_data/language/en.json',
    '/home/jyli/characteristic_identify/data/our_data/language/zh.json',
    '/home/jyli/characteristic_identify/data/our_data/language/de.json', 
    '/home/jyli/characteristic_identify/data/our_data/language/fr.json', 
    '/home/jyli/characteristic_identify/data/our_data/language/it.json', 
    '/home/jyli/characteristic_identify/data/our_data/language/pt.json', 
    '/home/jyli/characteristic_identify/data/our_data/language/sp.json', 
    '/home/jyli/characteristic_identify/data/our_data/language/th.json'
]

# 读取并合并所有JSON文件的数据
data = []
for file in input_files:
    with open(file, 'r', encoding='utf-8') as f:
        data.extend(json.load(f))  # 将每个文件的数据合并到data列表中

# 打乱数据顺序，确保训练集和验证集的随机性


# 计算分割位置
train_size = int(len(data) * 0.75)  # 训练集占 75%
validate_size = len(data) - train_size  # 验证集占 25%

# 划分数据集
train_data = data[:train_size]
validate_data = data[train_size:]

random.shuffle(validate_data)

# 保存训练集和验证集到 JSON 文件
with open('/home/jyli/characteristic_identify/data/our_data/language/train.json', 'w', encoding='utf-8') as train_file:
    json.dump(train_data, train_file, ensure_ascii=False, indent=4)

with open('/home/jyli/characteristic_identify/data/our_data/language/validate.json', 'w', encoding='utf-8') as validate_file:
    json.dump(validate_data, validate_file, ensure_ascii=False, indent=4)

print("训练集和验证集已成功保存！")
