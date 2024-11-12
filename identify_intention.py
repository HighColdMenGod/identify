import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
from tqdm import tqdm
import json
import numpy as np
import sklearn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import copy
from matplotlib import font_manager
from matplotlib.animation import FuncAnimation

class LinearNet(nn.Module):
    def __init__(self):
        super(LinearNet, self).__init__()
        self.fc1 = nn.Linear(4096, 4096)        #取出最后一层
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4096, 5)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


train =open("/home/jyli/characteristic_identify/data/our_data/intention/train.json","r",encoding='utf-8')
validate =open("/home/jyli/characteristic_identify/data/our_data/intention/validate.json","r",encoding='utf-8')


# llama_template='Please classify the following text into one of the following domains: Biomedicine, Finance, Law, Agriculture.\n\nText: {}\n\nPlease output the corresponding domain label:'
vicuna_template = '<s> USER:{}\n ASSISTANT:'
tokenizer = AutoTokenizer.from_pretrained("/home/jyli/models/vicuna-7b-v1.5", use_fast=False, trust_remote_code=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
print(device)
model = AutoModelForCausalLM.from_pretrained("/home/jyli/models/vicuna-7b-v1.5", device_map="auto",trust_remote_code=True,torch_dtype=torch.float16)
model.generation_config = GenerationConfig.from_pretrained("/home/jyli/models/vicuna-7b-v1.5")
# state_dict = model.state_dict()
# model.load_state_dict(state_dict)

#data =[]
data=torch.empty(0, 4096)
label=torch.empty(0, 1, dtype=torch.long)
testdata=torch.empty(0, 4096)
testlabel=torch.empty(0, 1, dtype=torch.long)
# datalist = json.load(p)

train_data = json.load(train)             #训练集
validate_data = json.load(validate)       #验证集

bingqingzhenduan = []
zhiliaofangan = []
jiuyijianyi = []
jibingbiaoshu = []
zhuyishixiang = []


label_map = {
    '病情诊断': 0,
    '治疗方案': 1,
    '就医建议': 2,
    '疾病表述': 3,
    '注意事项': 4,
}

for item in train_data:
    inp=item["context"]
    lab=item["label"]
    prompt=vicuna_template.format(inp)
    # print(f"prompt:{prompt}")
    inputs=tokenizer(prompt,add_special_tokens=False, return_tensors="pt")
    #print(inputs)
    inputs = inputs.to(device)
    pred = model(**inputs,output_hidden_states=True)
    #print(type(pred))
    # print("*"*50)
    middle =[]
    #print(pred.hidden_states) 
    for i in pred.hidden_states:
        # print(i.shape)
        middle.append(i)

    feature = middle[-1][-1][-1, :].data.cpu().numpy()
    if(lab==0):
        bingqingzhenduan.append(feature)
    elif(lab==1):
        zhiliaofangan.append(feature)
    elif(lab==2):
        jiuyijianyi.append(feature)
    elif(lab==3):
        jibingbiaoshu.append(feature)
    elif(lab==4):
        zhuyishixiang.append(feature)
    data = torch.cat((data, middle[-1][-1][-1,:].detach().cpu().unsqueeze(0)), dim=0)
    label = torch.cat((label, torch.tensor([[lab]], dtype=torch.long)), dim=0)
    
# print(f"label:{label}")


for item in validate_data:
    inp=item["context"]
    lab=item["label"]
    prompt=vicuna_template.format(inp)
    inputs=tokenizer(prompt,add_special_tokens=False, return_tensors="pt")
    #print(inputs)
    inputs = inputs.to(device)
    pred = model.model(**inputs,output_hidden_states=True)
    #print(type(pred))
    # print("*"*50)
    middle =[]
    #print(pred.hidden_states) 
    for i in pred.hidden_states:
        #print(i.shape)
        middle.append(i)

    testdata = torch.cat((testdata, middle[-1][-1][-1,:].detach().cpu().unsqueeze(0)), dim=0)       #middle[-1]:获取最后一层；[-1]:获取最后一个样本; [-1,:]:获取最后一个token的隐藏状态特征
    testlabel = torch.cat((testlabel, torch.tensor([[lab]], dtype=torch.long)), dim=0)


model_mlp = LinearNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_mlp.parameters(), lr=0.0001)
x_train_tensor = data.clone().detach()
y_train_tensor = label.clone().detach().squeeze()
    # 早停策略参数
patience = 5
best_loss = float('inf')
counter = 0
best_model = None
model_path="/home/jyli/characteristic_identify/mlp/intention"
    # 训练模型
num_epochs = 200
for epoch in range(num_epochs):
    model_mlp.train()
    optimizer.zero_grad()
    outputs = model_mlp(x_train_tensor)
    # print(f'outputs shape: {outputs.shape}, y_train_tensor shape: {y_train_tensor.shape}')

    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
        
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

         # 早停策略
    if loss.item() < best_loss:
        best_loss = loss.item()
        counter = 0
        best_model = copy.deepcopy(model_mlp.state_dict())  # 保存最佳模型
        torch.save(best_model, model_path+"_task_intention")

    else:
        counter += 1
        if counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')      
            break

print("*"*100)

# best_model = torch.load(model_path+"_task45")
# best_model.eval()
best_model = LinearNet().to(device)  # 初始化模型结构
best_model.load_state_dict(torch.load(model_path+"_task_intention"))  # 加载权重
best_model.eval()  # 设置模型为评估模式

y_test_tensor = testlabel.clone().detach().squeeze()
print(f"y_test_tensor shape: {y_test_tensor.shape}")
res=[]                    

with torch.no_grad():
    outputs = best_model(testdata.to(device))
    print(f"outputs shape: {outputs.shape}")

    predicted = torch.argmax(outputs, dim=1)        #获取类别索引
    print(f"predicted shape: {predicted.shape}")
    # print(f"predicted value: {predicted}")

    accuracy = accuracy_score(y_test_tensor.cpu(), predicted.cpu())  
    print("*"*100)
    print(f'Overall Accuracy: {accuracy:.4f}')

    res.append(accuracy)

font_path = "/home/jyli/fonts/simhei.ttf"  # 替换为你下载字体的路径
font_manager.fontManager.addfont(font_path)
plt.rcParams['font.family'] = ['simhei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


bingqingzhenduan = np.array(bingqingzhenduan)
zhiliaofangan = np.array(zhiliaofangan)
jiuyijianyi = np.array(jiuyijianyi)
jibingbiaoshu = np.array(jibingbiaoshu)
zhuyishixiang = np.array(zhuyishixiang)

all_data = np.concatenate([bingqingzhenduan, zhiliaofangan, jiuyijianyi, jibingbiaoshu, zhuyishixiang], axis=0)

tsne = TSNE(n_components=3, init='pca', random_state=42)
data_3d = tsne.fit_transform(all_data)

num_bingqingzhenduan = len(bingqingzhenduan)
num_zhiliaofangan = len(zhiliaofangan)
num_jiuyijianyi = len(jiuyijianyi)
num_jibingbiaoshu = len(jibingbiaoshu)
num_zhuyishixiang = len(zhuyishixiang)

bingqingzhenduan_3d = data_3d[:num_bingqingzhenduan]
zhiliaofangan_3d = data_3d[num_bingqingzhenduan:num_bingqingzhenduan + num_zhiliaofangan]
jiuyijianyi_3d = data_3d[num_bingqingzhenduan + num_zhiliaofangan:num_bingqingzhenduan + num_zhiliaofangan + num_jiuyijianyi]
jibingbiaoshu_3d = data_3d[num_bingqingzhenduan + num_zhiliaofangan + num_jiuyijianyi:num_bingqingzhenduan + num_zhiliaofangan + num_jiuyijianyi + num_jibingbiaoshu]
zhuyishixiang_3d = data_3d[num_bingqingzhenduan + num_zhiliaofangan + num_jiuyijianyi + num_jibingbiaoshu:]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(bingqingzhenduan_3d[:, 0], bingqingzhenduan_3d[:, 1], bingqingzhenduan_3d[:, 2], c='r', label='病情诊断', marker='o')
ax.scatter(zhiliaofangan_3d[:, 0], zhiliaofangan_3d[:, 1], zhiliaofangan_3d[:, 2], c='b', label='治疗方案', marker='^')
ax.scatter(jiuyijianyi_3d[:, 0], jiuyijianyi_3d[:, 1], jiuyijianyi_3d[:, 2], c='g', label='就医建议', marker='s')
ax.scatter(jibingbiaoshu_3d[:, 0], jibingbiaoshu_3d[:, 1], jibingbiaoshu_3d[:, 2], c='y', label='疾病表述', marker='*')
ax.scatter(zhuyishixiang_3d[:, 0], zhuyishixiang_3d[:, 1], zhuyishixiang_3d[:, 2], c='m', label='注意事项', marker='x')

ax.set_title('medical intention', fontsize=15)
# ax.set_xlabel('维度 1', fontsize=12)
# ax.set_ylabel('维度 2', fontsize=12)
# ax.set_zlabel('维度 3', fontsize=12)

ax.legend(loc='best')

# 定义旋转函数
def rotate(angle):
    ax.view_init(azim=angle)

# 创建动画
ani = FuncAnimation(fig, rotate, frames=np.arange(0, 360, 1), interval=50)

ani.save('/home/jyli/characteristic_identify/png/intention/rotation_animation.gif', writer='pillow')

# plt.savefig('/home/jyli/characteristic_identify/png/intention/feature_no_prompt.png')
plt.show()
