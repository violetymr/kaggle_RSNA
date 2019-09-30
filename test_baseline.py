#!/usr/bin/env python
# coding: utf-8

# In[93]:

#from apex import amp
#import matplotlib.pyplot as plt
import csv
import os
import pandas as pd
import glob
import numpy as np

import cv2
import torch

from torch.utils.data import Dataset
from albumentations import Compose, ShiftScaleRotate, Resize
from albumentations.pytorch import ToTensor

from efficientnet_pytorch import EfficientNet

import torch.optim as optim
from tqdm import tqdm_notebook as tqdm

dir_csv = '../input'
dir_train_img = '../input/stage_1_train_images'
dir_test_img = '../input/stage_1_test_images'

# Parameters
n_classes = 6
n_epochs = 2
batch_size = 128


# # # 读取csv 构建新的csv
#
# # In[48]:
# #为什么test都是0.5？
#
# #为什么test只要一个label?以及开始时train为什么都是0？
# #os.path.join  vs  csv.reader??合并文档目录的
# #csv  vs  dataset image 什么区别
# #CSV文件最早用在简单的数据库里，由于其格式简单，并具备很强的开放性，所以起初被扫图家用作自己图集的标记。CSV文件是个纯文本文件，每一行表示一张图片的许多属性。你在收一套图集时，只要能找到它的CSV文件，用专用的软件校验后，你对该图集的状况就可以了如指掌。 每行相当于一条记录，是用“，”分割字段的纯文本数据库文件。
# #https://www.runoob.com/python/python-os-path.html
# #train = csv.reader(open('../input/stage_1_train.csv','r+'))
# train = pd.read_csv(os.path.join(dir_csv, 'stage_1_train.csv'))
# #ID,Label
# #ID_63eb1e259_epidural,0
#
# #str.split
# #https://blog.csdn.net/weixin_40161254/article/details/95321858
# train[['ID', 'Image', 'Diagnosis']] = train['ID'].str.split('_', expand=True) #str.split pandas一列分多列的函数#expand添加原来的label列
# train = train[['Image', 'Diagnosis', 'Label']] #把id去掉了
# train.drop_duplicates(inplace=True) #(inplace=True)是直接对原dataFrame进行操作
#
# #pandas pivot table
# ##https://blog.csdn.net/qq_36495431/article/details/81123240
# train = train.pivot(index='Image', columns='Diagnosis', values='Label').reset_index()#行索引和列索引
# train['Image'] = 'ID_' + train['Image']
# train.head()
#
# # In[49]:
#
#
# #函数功能：匹配所有的符合条件的文件，并将其以list的形式返回
# png = glob.glob(os.path.join(dir_train_img, '*.png'))
# #print(png) 返回列表 ['../input/rsna-train-stage-1-images-png-224x/ID_000039fa0.png']
# png = [os.path.basename(png)[:-4] for png in png]  #[:-4] 从开始到倒数第4个保留
# #print(png) ['ID_000039fa0']
# png = np.array(png)
# #png 里面是图片目前只有5个 csv ; train[]现在有所有的train数据的csv
# #isin()接受一个列表，判断该列中元素是否在列表中。
# #isin判断是不是在 所以就把train csv变成了五个了
# train = train[train['Image'].isin(png)]
# train.to_csv('train.csv', index=False)
#
#
# # In[50]:
#
#
# test = pd.read_csv(os.path.join(dir_csv, 'stage_1_sample_submission.csv'))
#
#
# # In[51]:
#
#
# ## Also prepare the test data
# test[['ID','Image','Diagnosis']] = test['ID'].str.split('_', expand=True)
# test['Image'] = 'ID_' + test['Image']
# test = test[['Image', 'Label']]
# test.drop_duplicates(inplace=True)
#
# test.to_csv('test.csv', index=False)
# #





# # 构建dataset

# In[52]:


# Functions--construct dataset
#torch.utils.data.Dataset
#有__getitem__(self, index)函数来根据索引序号获取图片和标签, 有__len__(self)函数来获取数据集的长
class IntracranialDataset(Dataset):

    def __init__(self, csv_file, path, labels, transform=None):
        
        self.path = path
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.labels = labels

    def __len__(self):
        
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.path, self.data.loc[idx, 'Image'] + '.png')#用csv里idx img.png 读train集里的图片
        img = cv2.imread(img_name) #imread(path,name) 
        
        if self.transform:       
            
            augmented = self.transform(image=img)
            img = augmented['image']   
            
        if self.labels:
            
            labels = torch.tensor(
                self.data.loc[idx, ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']])
            return {'image': img, 'labels': labels}    
        
        else:      
            
            return {'image': img}
    


# Data loaders
#transform using the library ----albumentation
transform_train = Compose([
    ShiftScaleRotate(),
    ToTensor()
])
transform_test= Compose([
    ToTensor()
])

#construct dataset
train_dataset = IntracranialDataset(
    csv_file='train.csv', path=dir_train_img, transform=transform_train, labels=True)

test_dataset = IntracranialDataset(
    csv_file='test.csv', path=dir_test_img, transform=transform_test, labels=False)




# In[60]:


data_loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
data_loader_test = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
print(data_loader_train)

# In[78]:


#batch = next(iter(data_loader_train))
#batch['image'][0].numpy().shape#tensor->numpy.array (3, 224, 224) 3channels 224*224pixels
#np.transpose(batch['image'][0].numpy(), (1,2,0)).shape#(224, 224, 3)
#fig, axs = plt.subplots(1, 2, figsize=(15,5))
#axs[0].imshow(np.transpose(batch['image'][0].numpy(), (1,2,0))[:,:,0])## 显示图片的第一个通道


# In[61]:


#batch = next(iter(data_loader_train))
#fig, axs = plt.subplots(1, 2, figsize=(15,5))

#for i in np.arange(2):
    
#    axs[i].imshow(np.transpose(batch['image'][i].numpy(), (1,2,0))[:,:,0], cmap=plt.cm.bone)


# In[62]:


# Plot test example
#train train png->csv 
#test csv+png load image
#batch = next(iter(data_loader_test))
#fig, axs = plt.subplots(1, 2, figsize=(15,5))

#for i in np.arange(2):
    
#    axs[i].imshow(np.transpose(batch['image'][i].numpy(), (1,2,0))[:,:,0], cmap=plt.cm.bone)


# # Model

# In[91]:




# Model

device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")
model = EfficientNet.from_pretrained('efficientnet-b0') 
model._fc = torch.nn.Linear(1280, n_classes)

model.to(device)

criterion = torch.nn.BCEWithLogitsLoss()
plist = [{'params': model.parameters(), 'lr': 2e-5}]
optimizer = optim.Adam(plist, lr=2e-5)

#model, optimizer = amp.initialize(model, optimizer, opt_level="O1")


# # training

# In[96]:


for epoch in range(n_epochs):
    
    print('Epoch {}/{}'.format(epoch, n_epochs - 1))
    print('-' * 10)

    model.train()    
    tr_loss = 0

   # tk0 = tqdm(data_loader_train, desc="Iteration")

    for step, batch in enumerate(data_loader_train):

        #print(batch[0])
        inputs = batch["image"]
        labels = batch["labels"]

        inputs = inputs.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.float)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

 #       with amp.scale_loss(loss, optimizer) as scaled_loss:
 #           scaled_loss.backward()
        loss.backward()
        tr_loss += loss.item()

        optimizer.step()
        optimizer.zero_grad()

    epoch_loss = tr_loss / len(data_loader_train)
    print('Training Loss: {:.4f}'.format(epoch_loss))


# # testing
print("testing")
# In[97]:


for param in model.parameters():
    param.requires_grad = False

model.eval()

test_pred = np.zeros((len(test_dataset) * n_classes, 1))
#print("enum",tqdm(data_loader_test))

for i, x_batch in enumerate(data_loader_test):
    
    x_batch = x_batch["image"]
    x_batch = x_batch.to(device, dtype=torch.float)
    
    with torch.no_grad():
        
        pred = model(x_batch)
        
        test_pred[(i * batch_size * n_classes):((i + 1) * batch_size * n_classes)] = torch.sigmoid(
            pred).detach().cpu().reshape((len(x_batch) * n_classes, 1))


# # submission

# In[ ]:

submission =  pd.read_csv(os.path.join(dir_csv, 'stage_1_sample_submission.csv'))
submission = pd.concat([submission.drop(columns=['Label']), pd.DataFrame(test_pred)], axis=1)
submission.columns = ['ID', 'Label']
submission.to_csv('submission.csv', index=False)
submission.head()

