import json

# 输入和输出文件路径
input_txt_file = '/home/jyli/characteristic_identify/data/our_data/language/txt_data/zh.txt'  # 假设你的txt文件名为 data.txt
output_json_file = '/home/jyli/characteristic_identify/data/our_data/language/zh.json'  # 输出的JSON文件名

# 初始化一个空列表，存放所有的JSON对象
data = []

# 逐行读取文本文件
with open(input_txt_file, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()  # 去除每行文本的前后空白字符
        if line:  # 如果行不为空
            # 构建每个条目的JSON对象
            entry = {
                'context': line,
                'label': 1
            }
            data.append(entry)



# 将数据写入到JSON文件
with open(output_json_file, 'w', encoding='utf-8') as out_f:
    json.dump(data, out_f, ensure_ascii=False, indent=4)

print(f"数据已保存到 {output_json_file}")
