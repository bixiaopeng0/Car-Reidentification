import os
import argparse

import numpy as np
from scipy.spatial.distance import hamming, cdist
# from net import AlexNetPlusLatent
import resnet50_hash
from timeit import time

import torch
import torch.nn as nn

from torchvision import datasets, models, transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler

parser = argparse.ArgumentParser(description='Deep Hashing evaluate mAP')
parser.add_argument('--pretrained', type=int, default=0, metavar='pretrained_model',
                    help='loading pretrained model(default = None)')
parser.add_argument('--bits', type=int, default=48, metavar='bts',
                    help='binary bits')
args = parser.parse_args()

batch_size = 16

def load_data():
    train_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        # transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((.4, .4, .4), (.08, .08, .08))
    ])
    val_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        # transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((.4, .4, .4), (.08, .08, .08))
    ])
    train_dir = './mydataset/train'
    train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=False)

    val_dir = './mydataset/test'
    val_datasets = datasets.ImageFolder(val_dir, transform=val_transforms)
    val_dataloader = torch.utils.data.DataLoader(val_datasets, batch_size=batch_size, shuffle=False)
    # transform_train = transforms.Compose(
    #     [transforms.Scale(227),
    #      transforms.ToTensor(),
    #      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    # transform_test = transforms.Compose(
    #     [transforms.Scale(227),
    #      transforms.ToTensor(),
    #      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    # trainset = datasets.CIFAR10(root='./data', train=True, download=True,
    #                             transform=transform_train)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
    #                                           shuffle=False, num_workers=2)
    #
    # testset = datasets.CIFAR10(root='./data', train=False, download=True,
    #                            transform=transform_test)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=100,
    #                                          shuffle=False, num_workers=2)
    return train_dataloader, val_dataloader

def binary_output(dataloader):
    net = resnet50_hash.resnet50(False,10)
    net.load_state_dict(torch.load('./weight/40xxxxx.pth'))
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        net.cuda()
    full_batch_output = torch.cuda.FloatTensor()
    full_batch_label = torch.cuda.LongTensor()
    with torch.no_grad():
        net.eval()
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs,_= net(inputs)
            #torch cat 在第0维拼接
            full_batch_output = torch.cat((full_batch_output, outputs.data), 0)
            full_batch_label = torch.cat((full_batch_label, targets.data), 0)
    #torch round 四舍五入  在这里相当于阈值为0.5
    return torch.round(full_batch_output), full_batch_label,full_batch_output


#
def precision(trn_binary, trn_label, tst_binary, tst_label):
    trn_binary = trn_binary.cpu().numpy()
    trn_binary = np.asarray(trn_binary, np.int32)
    trn_label = trn_label.cpu().numpy()
    tst_binary = tst_binary.cpu().numpy()
    tst_binary = np.asarray(tst_binary, np.int32)
    tst_label = tst_label.cpu().numpy()
    #分别为查询图片数量  训练集图片数量
    query_times = tst_binary.shape[0]
    trainset_len = train_binary.shape[0]
    # print(query_times,trainset_len)
    AP = np.zeros(query_times)
    Ns = np.arange(1, trainset_len + 1)
    total_time_start = time.time()
    for i in range(query_times):
        print('Query ', i+1)
        query_label = tst_label[i]
        query_binary = tst_binary[i,:]
        #一次查询整个数据库
        query_result = np.count_nonzero(query_binary != trn_binary, axis=1)    #don't need to divide binary length
        sort_indices = np.argsort(query_result)
        #np.equal 判断是否相等返回值是布尔值   astype(int)强制转换为int类型
        #buufer_yse 维度9000 是否与检索图片为一类
        buffer_yes= np.equal(query_label, trn_label[sort_indices]).astype(int)
        print(buffer_yes,buffer_yes.shape)
        #cumsum 累加之前的元素  np.cumsum([1,1,1]) -> [1,2,3]
        #前N个数据里含有和查询图片同类的占比
        P = np.cumsum(buffer_yes) / Ns
        print("p",P)
        AP[i] = np.sum(P * buffer_yes) /sum(buffer_yes)
    map = np.mean(AP)
    print(map)
    print('total query time = ', time.time() - total_time_start)



if os.path.exists('./result/train_binary') and os.path.exists('./result/train_label') and \
   os.path.exists('./result/test_binary') and os.path.exists('./result/test_label') and args.pretrained == 0:
    train_binary = torch.load('./result/train_binary')
    train_label = torch.load('./result/train_label')
    test_binary = torch.load('./result/test_binary')
    test_label = torch.load('./result/test_label')

else:
    trainloader, testloader = load_data()
    train_binary, train_label,train_data = binary_output(trainloader)
    test_binary, test_label,test_data = binary_output(testloader)
    if not os.path.isdir('result'):
        os.mkdir('result')
    torch.save(train_binary, './result/train_binary')
    torch.save(train_label, './result/train_label')
    torch.save(test_binary, './result/test_binary')
    torch.save(test_label, './result/test_label')
    torch.save(train_data,'./result/train_data')


precision(train_binary, train_label, test_binary, test_label)
