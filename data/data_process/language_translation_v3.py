import os
import requests
import json
import time
import random
import hashlib
import urllib.parse

class Translation():
    """
    该类实现了从TXT文件逐行翻译文本并保存翻译结果的功能
    """

    def __init__(self, appid, appkey):
        self.appid = appid
        self.appkey = appkey

    def generate_sign(self, query, salt):
        """
        生成签名，百度翻译API要求使用md5对参数进行加密
        """
        sign = self.appid + query + str(salt) + self.appkey
        return hashlib.md5(sign.encode('utf-8')).hexdigest()

    def translate(self, query):
        """
        调用百度翻译API进行翻译
        """
        url = "http://api.fanyi.baidu.com/api/trans/vip/translate"
        salt = random.randint(32768, 65536)  # 随机生成salt
        sign = self.generate_sign(query, salt)

        params = {
            'q': query,
            'from': 'en',  # 源语言：英语
            'to': 'zh',    # 目标语言：中文
            'appid': self.appid,
            'salt': salt,
            'sign': sign
        }
        try:
            response = requests.get(url, params=params)
            result = response.json()
            if "trans_result" in result:
                return result["trans_result"][0]["dst"]
            else:
                return "翻译失败"
        except Exception as e:
            return f"请求失败: {str(e)}"

    def process(self, in_file, out_file):
        """
        处理输入文件并将翻译结果保存到输出文件
        :param in_file: 输入的txt文件路径
        :param out_file: 输出的txt文件路径
        """
        with open(in_file, "r", encoding="utf-8") as infile, open(out_file, "w", encoding="utf-8") as outfile:
            num = 0
            for line in infile:
                line = line.strip()
                if line:  # 确保不为空行
                    num += 1
                    translation = self.translate(line)
                    # 输出翻译结果
                    if translation=="翻译失败":
                        print(f"第{num}次失败")
                    outfile.write(f"{translation}\n")
                    print(f"{num}. 原文: {line} -> 翻译: {translation}")

if __name__ == '__main__':
    appid = '20241107002196902'
    appkey = '9rPI8BYR13j2rI08KTMw'
    
    t = Translation(appid, appkey)
    in_file = "/home/jyli/characteristic_identify/data/our_data/language/en.txt"
    out_file = "/home/jyli/characteristic_identify/data/our_data/language/zh.txt"
    t.process(in_file=in_file, out_file=out_file)