'''
time:2019/12/04
将featureout数据保存
'''

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
import myres50
import read_mat_file
import matplotlib.pyplot as plt
import os

test_dir='../data/test/'
window_dict = read_mat_file.get_car_windows()

test_transforms = transforms.Compose([
    # 将图像转换为tensor并且归一化至【0-1】
    transforms.ToTensor(),
    transforms.Normalize((.5, .5, .5), (.2, .2, .2))
])

model = myres50.resnet50(num_classes=50)
if torch.cuda.is_available():
    model=model.cuda()
    print("gpu")
print(model)
state_dict = torch.load('../weights/35.pth')
model.load_state_dict(state_dict)

class net_dataset(Dataset):
    def __init__(self, img_folder, imageFolderDataset, transform=None, should_invert=True):
        self.transform = transform
        self.should_invert = should_invert
        self.img_folder = img_folder
        self.imageFolderDataset = imageFolderDataset

    def __getitem__(self, item):
        self.img_folder_list = glob.glob(self.img_folder + '*')
        label = random.randint(0, len(self.img_folder_list) - 1)
        img0_path = self.img_folder_list[label]
        # 第一张图片的类别
        img0_class = img0_path.split('/')[-1]
        # print(img0_class)
        # print(img0_path)
        img_list = glob.glob(img0_path + '/*')
        # 获得初始图片
        img0_way = img_list[random.randint(0, len(img_list) - 1)]

        img0_name = img0_way.split('/')[-1]
        img0_rect = window_dict[img0_name]
        img0 = Image.open(img0_way)

        window0 = img0.crop(img0_rect)

        img0 = window0.resize((400, 200))


        # plt.imshow(img0)
        # plt.show()

        if self.transform is not None:
            img0 = self.transform(img0)


        return img0,  label

    def __len__(self):
        print("data-len", len(self.imageFolderDataset.imgs))
        return len(self.imageFolderDataset.imgs)



#这里数据集，每一张图片对应一个标签，为了在检索时区分，不要让自己跟自己进行检索
def produce_feature():
    doc_dir = []
    dirs = os.listdir(test_dir)
    img_label = 0
    full_batch_output = torch.cuda.FloatTensor()
    full_batch_label = torch.cuda.LongTensor()
    with torch.no_grad():
        for dir in dirs:
            img_pat = os.path.join(test_dir,dir)
            img_paths = os.listdir(img_pat)
            for img_name in img_paths:
                # print(img_name)
                img_label+=1
                print(img_label)
                img_path = os.path.join(img_pat,img_name)

                img0_rect = window_dict[img_name]
                img0 = Image.open(img_path)
                window0 = img0.crop(img0_rect)
                img0 = window0.resize((400, 200))
                img0 = test_transforms(img0)
                img0 = img0.unsqueeze(0)
                img0 = Variable(img0).cuda()
                outputs = model(img0)
                full_batch_output = torch.cat((full_batch_output, outputs), 0)
                img_label = torch.tensor(img_label).view(1).cuda()
                # print(torch.tensor(img_label).view(1))
                full_batch_label = torch.cat((full_batch_label, img_label), 0)
    torch.save(full_batch_output, './test_binary')
    torch.save(full_batch_label, './test_label')


if __name__== "__main__":
    produce_feature()