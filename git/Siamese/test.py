import Siamese
import torch
import torch.nn as nn
import cv2
import torch.nn.functional as F
import glob
import random
from PIL import Image
import numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
from torch.autograd import Variable
from train_data import SiameseNetwork
import siamese_window
import myres50
print("test")
test_dir='../data/test/'
batch_size = 1
model = myres50.resnet50(num_classes=50)
if torch.cuda.is_available():
    model=model.cuda()
    print("gpu")

state_dict = torch.load('../weights/35.pth')
model.load_state_dict(state_dict)

test_transforms = transforms.Compose([
    # 亮度 对比度 饱和度
    # transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
    # 随机旋转（-60，60）
    # transforms.RandomRotation(6),
    # 将图像转换为tensor并且归一化至【0-1】
    transforms.ToTensor(),
    transforms.Normalize((.5, .5, .5), (.2, .2, .2))
])


# 数据加载
folder_dataset=dset.ImageFolder(root=test_dir)

siamese_dataset = siamese_window.net_dataset(img_folder=test_dir,imageFolderDataset=folder_dataset,
                              transform=test_transforms,
                              should_invert=False)

print('train_data')
test_data=DataLoader(siamese_dataset, batch_size=batch_size, shuffle=False)

len_test = len(test_data)
print(len_test)
f = open("../data/data.txt",'w')
def test():
    right_nums = 0
    all_nums = 0
    for i,data in enumerate(test_data,0):
        # print(len(train_data))
        #print(data)
        # label用来表示是否一样
        f = open("../data/data.txt", 'a+')
        img0,img1,label=data
        img0, img1, label=Variable(img0).cuda(),Variable(img1).cuda(),Variable(label).cuda()
        out1=model(img0)
        out2=model(img1)
        euclidean_distance = F.pairwise_distance(out1, out2)
        # label_int = label.int()
        # dis_f = euclidean_distance.float()
        print("label:",label[0][0].item(),"dis:",euclidean_distance[0].item())
        if label == 0 and euclidean_distance < 1.1:
            right_nums +=1
        if label == 1 and euclidean_distance >=1.1:
            right_nums +=1
        f.write(str(int(label[0][0].item()))+","+str(euclidean_distance[0].item())+'\n')
        all_nums+=1
        f.close()
    print(all_nums,right_nums)
    print("准确率:",right_nums/len_test)
    f.close()

test()


