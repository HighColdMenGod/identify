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


class LinearNet(nn.Module):
    def __init__(self):
        super(LinearNet, self).__init__()
        self.fc1 = nn.Linear(4096, 4096)        #取出最后一层
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4096, 4)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        # x = torch.sigmoid(x)
        return x


train =open("/home/jyli/characteristic_identify/data/our_data/field/field_train.json","r",encoding='utf-8')
validate =open("/home/jyli/characteristic_identify/data/our_data/field/field_validate.json","r",encoding='utf-8')


# llama_template='Please classify the following text into one of the following domains: Biomedicine, Finance, Law, Agriculture.\n\nText: {}\n\nPlease output the corresponding domain label:'
vicuna_template = '<s> USER:{}\n ASSISTANT:'
tokenizer = AutoTokenizer.from_pretrained("/home/jyli/models/vicuna-7b-v1.5", use_fast=False, trust_remote_code=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
print(device)
model = AutoModelForCausalLM.from_pretrained("/home/jyli/models/vicuna-7b-v1.5", device_map="auto",trust_remote_code=True,torch_dtype=torch.float16)
model.generation_config = GenerationConfig.from_pretrained("/home/jyli/models/vicuna-7b-v1.5")
state_dict = model.state_dict()
model.load_state_dict(state_dict)

#data =[]
data=torch.empty(0, 4096)
label=torch.empty(0, 1, dtype=torch.long)
testdata=torch.empty(0, 4096)
testlabel=torch.empty(0, 1, dtype=torch.long)
# datalist = json.load(p)

train_data = json.load(train)             #训练集
validate_data = json.load(validate)       #验证集

medical = []
finance = []
law = []
agriculture = []

label_map = {
    'medical': 0,
    'finance': 1,
    'law': 2,
    'agricuture': 3,
}

for item in train_data:
    inp=item["context"]
    lab=item["label"]
    prompt=vicuna_template.format(inp)
    # print(f"prompt:{prompt}")
    inputs=tokenizer(prompt,add_special_tokens=False, return_tensors="pt")
    #print(inputs)
    inputs = inputs.to(device)
    pred = model.model(**inputs,output_hidden_states=True)
    #print(type(pred))
    # print("*"*50)
    middle =[]
    #print(pred.hidden_states) 
    for i in pred.hidden_states:
        # print(i.shape)
        middle.append(i)

    feature = middle[-1][-1][-1, :].data.cpu().numpy()
    if(lab==0):
        medical.append(feature)
    elif(lab==1):
        finance.append(feature)
    elif(lab==2):
        law.append(feature)
    elif(lab==3):
        agriculture.append(feature)
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
model_path="/home/jyli/characteristic_identify/mlp/field"
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
        torch.save(best_model, model_path+"_task_field")

    else:
        counter += 1
        if counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')      
            break

print("*"*100)

# best_model = torch.load(model_path+"_task45")
# best_model.eval()
best_model = LinearNet().to(device)  # 初始化模型结构
best_model.load_state_dict(torch.load(model_path+"_task_field"))  # 加载权重
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

medical = np.array(medical)
finance = np.array(finance)
law = np.array(law)
agriculture = np.array(agriculture)

all_data = np.concatenate([medical, finance, law, agriculture], axis=0)
tsne = TSNE(n_components=3, init='pca', random_state=42)
data_3d = tsne.fit_transform(all_data)

num_medical = len(medical)
num_finance = len(finance)
num_law = len(law)
num_agriculture = len(agriculture)

medical_3d = data_3d[:num_medical]
finance_3d = data_3d[num_medical:num_medical + num_finance]
law_3d = data_3d[num_medical + num_finance:num_medical + num_finance + num_law]
agriculture_3d = data_3d[num_medical + num_finance + num_law:]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(medical_3d[:, 0], medical_3d[:, 1], medical_3d[:, 2], c='r', label='Medical', marker='o')
ax.scatter(finance_3d[:, 0], finance_3d[:, 1], finance_3d[:, 2], c='b', label='Finance', marker='^')
ax.scatter(law_3d[:, 0], law_3d[:, 1], law_3d[:, 2], c='g', label='Law', marker='s')
ax.scatter(agriculture_3d[:, 0], agriculture_3d[:, 1], agriculture_3d[:, 2], c='y', label='Agriculture', marker='*')

ax.set_title('domain', fontsize=15)
# ax.set_xlabel('Component 1', fontsize=12)
# ax.set_ylabel('Component 2', fontsize=12)
# ax.set_zlabel('Component 3', fontsize=12)

ax.legend()

plt.savefig('/home/jyli/characteristic_identify/png/field/feature_no_prompt.png')
plt.show()
