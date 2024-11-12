import requests
import random
import json
from hashlib import md5

# Set your own appid/appkey.
appid = '20241107002196902'
appkey = '9rPI8BYR13j2rI08KTMw'

# 输入文件路径
input_file = '/home/jyli/characteristic_identify/data/source_data/language/alpaca_data.json'  # 输入文件路径
output_file = '/home/jyli/characteristic_identify/data/our_data/language/language_train.json'  # 输出文件路径
output_txt_file = '/home/jyli/characteristic_identify/data/our_data/language/temp.txt'

languages = {
    'zh': 1,  # 中文
    # 'de': 2,     # 德语
    # 'fra': 3,     # 法语
    # 'it': 4,     # 意大利语
    # 'pt': 5,     # 葡萄牙语
    # 'jp': 6,     # 日语
    # 'spa': 7,     # 西班牙语
    # 'th': 8      # 泰语
}

endpoint = 'http://api.fanyi.baidu.com'
path = '/api/trans/vip/translate'
url = endpoint + path

# query = 'Hello World! This is 1st paragraph.\nThis is 2nd paragraph.'

# Generate salt and sign
def make_md5(s, encoding='utf-8'):
    return md5(s.encode(encoding)).hexdigest()

def translate_baidu(text, from_lang='en', to_lang='zh'):
    salt = random.randint(32768, 65536)
    sign = make_md5(appid + text + str(salt) + appkey)

    # Build request
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    payload = {'appid': appid, 'q': text, 'from': from_lang, 'to': to_lang, 'salt': salt, 'sign': sign}

    # Send request
    r = requests.post(url, params=payload, headers=headers)

    result = r.json()
    print(result['trans_result'][0]['dst'])
    return result['trans_result'][0]['dst']

# with open(output_file, 'w', encoding='utf-8') as output_f:
#     output_f.write('[')  # 文件开始
    
#     first_item = True  # 用于控制逗号格式
#     # 打开并读取输入的 JSON 文件
#     with open(input_file, 'r', encoding='utf-8') as input_f:
#         data = json.load(input_f)
        
#         for item in data[:200]:  # 只处理前200条数据
#             # 拼接 instruction, input 和 output 为 sentence
#             sentence = f"{item['instruction']} {item['input']} {item['output']}"
#             sentence = sentence.replace("\n", "")
            
#             # 保存原始英文
#             original_item = {
#                 'sentence': sentence,
#                 'label': 0
#             }
#             # 写入原始英文到文件
#             # print(f"sentence{sentence}")
#             if not first_item:
#                 output_f.write(',\n')
#             first_item = False
#             json.dump(original_item, output_f, ensure_ascii=False, indent=4)
        
#     output_f.write('\n]')  # 文件结束

with open(output_file, 'r') as en_file:
    data = json.load(en_file)

with open(output_txt_file, 'w', encoding='utf-8') as output_txt:
    for item in data:
        # 提取 sentence 字段
        sentence = item.get('sentence', '')
        
        # 写入 sentence 并添加换行符
        output_txt.write(sentence + '\n')

