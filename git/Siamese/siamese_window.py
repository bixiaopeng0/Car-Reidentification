'''
对车窗进行siamese网络匹配
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


num_epoch=5000
train_dir='../data/train/'
batch_size=4
ln = 0.0001

flag = True
model = myres50.resnet50(num_classes=50)
# state_dict = torch.load('../weights/46.pth')
# model.load_state_dict(state_dict)
if flag == True:
    model_dict = model.state_dict()
    state_dict = torch.load('resnet50-19c8e357.pth')
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    pretrained_dict.pop('fc.weight')
    pretrained_dict.pop('fc.bias')
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)



if torch.cuda.is_available():
    model=model.cuda()
    print("gpu")

#提取车窗图片

window_dict = read_mat_file.get_car_windows()
print (window_dict)
def get_car_window(car_img):

    return window_img


# 损失函数
class ContrastiveLoss(torch.nn.Module):

    #margin设定的阈值
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):

        #不相同；label为1
        # F.cosine_similarity
        # euclidean_distance = F.pairwise_distance(output1, output2,keepdim = True)
        # print("cos", F.cosine_similarity(output1, output2)*10,"pair  ",F.pairwise_distance(output1, output2,keepdim = True))
        euclidean_distance = F.pairwise_distance(output1, output2,keepdim = True)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2)
                                      +(label) * torch.pow(torch.clamp(self.margin
                                                                       - euclidean_distance, min=0.0), 2))

        return loss_contrastive


train_transforms = transforms.Compose([
    # 亮度 对比度 饱和度
    transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
    # 随机旋转（-60，60）
    transforms.RandomRotation(6),
    # 将图像转换为tensor并且归一化至【0-1】
    transforms.ToTensor(),
    transforms.Normalize((.5, .5, .5), (.2, .2, .2))
])

# 数据加载
class net_dataset(Dataset):
        def __init__(self,img_folder,imageFolderDataset,transform=None,should_invert=True):
            self.transform = transform
            self.should_invert = should_invert
            self.img_folder=img_folder
            self.imageFolderDataset=imageFolderDataset

        def __getitem__(self, item):
            self.img_folder_list=glob.glob(self.img_folder+'*')
            img0_path=self.img_folder_list[random.randint(0,len(self.img_folder_list)-1)]
            # 第一张图片的类别
            img0_class=img0_path.split('/')[-1]
            # print(img0_class)
            # print(img0_path)
            img_list=glob.glob(img0_path+'/*')
            # 获得初始图片
            img0_way=img_list[random.randint(0,len(img_list)-1)]
            # 判断是否需要同类
            should_get_same_img=random.randint(0,1)
            # if should_get_same_img:
            #     should_get_same_img = random.randint(0, 1)
            if should_get_same_img:
                # 如果相同就在源目录再找一张
                img1_way=img_list[random.randint(0,len(img_list)-1)]
                img1_class=img0_class
            else:
                # 如果不同就在别的目录照一张
                while True:
                    img1_path=self.img_folder_list[random.randint(0,len(self.img_folder_list)-1)]
                    img1_class=img1_path.split('/')[-1]
                    if img1_class != img0_class:
                        img_list=glob.glob(img1_path+'/*')
                        img1_way=img_list[random.randint(0,len(img_list)-1)]
                        break

            img0_name = img0_way.split('/')[-1]
            img0_rect = window_dict[img0_name]
            img1_name = img1_way.split('/')[-1]
            img1_rect = window_dict[img1_name]
            #



            img0=Image.open(img0_way)
            img1=Image.open(img1_way)
            window0 = img0.crop(img0_rect)
            window1 = img1.crop(img1_rect)
            img0=window0.resize((400,200))
            img1=window1.resize((400,200))

            # plt.imshow(img0)
            # plt.show()




            if self.transform is not None:
                img0 = self.transform(img0)
                img1 = self.transform(img1)
            # train_transforms = transforms.Compose([
            #     #随机大小，随机长宽比，最后将图片resize成设定好的大小
            #     transforms.RandomResizedCrop(224),
            #     # 水平翻转
            #     transforms.RandomHorizontalFlip(),
            #     transforms.ToTensor(),
            #     transforms.Normalize((.5, .5, .5), (.5, .5, .5))
            # ])
            # img0=img0.resize((224,224))
            # img1=img1.resize((224,224))

            # if self.transform is not None:
            #     img0 = train_transforms(img0)
            #     img1 = train_transforms(img1)
            #print(img0.size)

            #label=1 if img1_class == img0_class else 0

            return img0, img1, torch.from_numpy(np.array([int(img1_class != img0_class)], dtype=np.float32))

        def __len__(self):
            print("data-len",len(self.imageFolderDataset.imgs))
            return len(self.imageFolderDataset.imgs)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = ln * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        print("learning rate",lr)



def train():
    print("data_floder")
    #folder_dataset 包含训练的根目录还有目录下图片的数量
    folder_dataset=dset.ImageFolder(root=train_dir)
    print("folder_dataset",folder_dataset)
    print("siamese_dataset")
    siamese_dataset = net_dataset(img_folder=train_dir,imageFolderDataset=folder_dataset,
                                  transform=train_transforms,
                                  should_invert=False)

    print('train_data')
    train_data=DataLoader(siamese_dataset, batch_size=batch_size, shuffle=False)
    print(len(train_data))


    criterion=ContrastiveLoss()
    optimizer=optim.Adam(model.parameters(),lr=0.0001)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    #print(train_data)
    loss_all=[]
    correct_nums_i = 0
    for epoch in range(num_epoch):
        # adjust_learning_rate(optimizer, epoch)
        correct_nums = 0

        for i,data in enumerate(train_data,0):

            #print(data)
            # label用来表示是否一样
            img0,img1,label=data
            img0, img1, label=Variable(img0).cuda(),Variable(img1).cuda(),Variable(label).cuda()
            optimizer.zero_grad()
            out1=model(img0)
            out2=model(img1)
            loss=criterion(out1,out2,label)
            loss.backward()
            optimizer.step()
            loss_all.append(loss.item())



            euclidean_distance = F.pairwise_distance(out1, out2)

            predict_label = euclidean_distance > 1.3
            label = label.view(4)
            label_uint8 = torch.tensor(label, dtype=torch.uint8)
            correct_nums += (predict_label.cpu()==label_uint8).sum().item()
            correct_nums_i += (predict_label.cpu()==label_uint8).sum().item()



            # print(i)
            if i %60==0:

                print("epoch:",epoch)
                print(sum(loss_all[:])/60)
                print("predict_batch_size precison", correct_nums_i / 240)
                correct_nums_i = 0
                f = open("loss.txt","a+")
                f.write(str(sum(loss_all[:])/60)+'\n')
                f.close()
                loss_all=[]
            # 事实上I到不了50000
            # if i % 50000==0:
            #     torch.save(net.state_dict(), 'model/double' + str(i) + '.pkl')
        f = open("loss.txt", "a+")
        f.write("precison  "+str(correct_nums/len(train_data)) + '\n')
        f.close()
        print("precion:",correct_nums/len(train_data))
        if epoch%1==0:
            torch.save(model.state_dict(),'../weights/'+str(epoch)+'.pth')


if __name__== "__main__":
    train()













