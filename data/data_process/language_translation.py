import json
import hashlib
import random
import requests
from hashlib import md5

# 百度翻译 API 的 app_id 和 secret_key
app_id = '20241107002196902'  # 在百度翻译开放平台注册并获取
secret_key = '9rPI8BYR13j2rI08KTMw'

# 输入文件路径
input_file = '/home/jyli/characteristic_identify/data/source_data/language/alpaca_data.json'  # 输入文件路径
output_file = '/home/jyli/characteristic_identify/data/our_data/language/language_train.json'  # 输出文件路径

# 语言和对应的 label 字典
languages = {
    'zh': 1,  # 中文
    'de': 2,     # 德语
    'fra': 3,     # 法语
    'it': 4,     # 意大利语
    'pt': 5,     # 葡萄牙语
    'hi': 6,     # 印地语
    'spa': 7,     # 西班牙语
    'th': 8      # 泰语
}

def make_md5(s, encoding='utf-8'):
    return md5(s.encode(encoding)).hexdigest()

# 百度翻译 API 请求函数
def translate_baidu(text, from_lang='en', to_lang='zh'):
    salt = random.randint(32768, 65536)
    sign = make_md5(app_id + text + str(salt) + secret_key)
    
    url = "http://api.fanyi.baidu.com/api/trans/vip/translate"
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    params = {
        'q': text,
        'from': from_lang,
        'to': to_lang,
        'appid': app_id,
        'salt': salt,
        'sign': sign
    }
    
    response = requests.post(url, params=params, headers=headers)
    result = response.json()
    
    if 'trans_result' in result:
        return result['trans_result'][0]['dst']
    else:
        return None

# 准备输出的 JSON 文件
with open(output_file, 'w', encoding='utf-8') as output_f:
    output_f.write('[')  # 文件开始
    
    first_item = True  # 用于控制逗号格式
    
    # 打开并读取输入的 JSON 文件
    with open(input_file, 'r', encoding='utf-8') as input_f:
        data = json.load(input_f)
        
        for item in data[50:200]:  # 只处理前200条数据
            # 拼接 instruction, input 和 output 为 sentence
            sentence = f"{item['instruction']} {item['input']} {item['output']}"
            
            # 保存原始英文
            original_item = {
                'sentence': sentence,
                'label': 0
            }
            
            # 写入原始英文到文件
            if not first_item:
                output_f.write(',\n')
            first_item = False
            json.dump(original_item, output_f, ensure_ascii=False, indent=4)
            
            # 翻译并保存其他语言
            for lang, label in languages.items():
                # 翻译 sentence
                translated_sentence = translate_baidu(sentence, from_lang='en', to_lang=lang)
                
                if translated_sentence:
                    # 创建翻译后的字典
                    translated_item = {
                        'sentence': translated_sentence,
                        'label': label
                    }
                    
                    # 写入翻译后的内容
                    output_f.write(',\n')
                    json.dump(translated_item, output_f, ensure_ascii=False, indent=4)
    
    output_f.write('\n]')  # 文件结束

print(f"Translated data has been saved to {output_file}")
