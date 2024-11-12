import json

# 定义输入和输出文件名
input_file = "/home/jyli/characteristic_identify/data/source_data/sentence_type/swda/train.txt"
output_file = "/home/jyli/characteristic_identify/data/our_data/sentence_type/train.json"

# 存储所有符合条件的对话记录
data = []

# 打开输入文件并逐行读取
with open(input_file, "r", encoding="utf-8") as file:
    for line in file:
        # 去除行末的换行符
        line = line.strip()
        # 按制表符分割每行数据
        parts = line.split("\t")
        
        # 确保每行至少包含4部分数据
        if len(parts) >= 4:
            # 获取 s_type 和 text
            s_type = int(parts[1])  # s_type 是 parts[1]
            text = parts[3]         # text 是 parts[3]
            word_count = len(text.split())  # 计算 text 中的单词数
            
            # 判断条件：保留指定 s_type 且 word_count 在 6 到 15 之间
            if s_type in [2, 4, 13, 20] and 6 <= word_count <= 15:
                # 转换 s_type 的值
                if s_type in [2, 4]:
                    s_type = 0
                elif s_type in [13, 20]:
                    s_type = 1
                
                # 添加符合条件的记录到 data 列表中
                data.append({
                    "s_type": s_type,
                    "text": text
                })

# 将数据写入到 JSON 文件中
with open(output_file, "w", encoding="utf-8") as json_file:
    json.dump(data, json_file, ensure_ascii=False, indent=4)

print(f"转换完成！符合条件的数据已保存到 {output_file}")
