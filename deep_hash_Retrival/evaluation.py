import os
import argparse

import numpy as np


from timeit import time

import torch
import torch.nn as nn

from torchvision import datasets, models, transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler
import resnet50_hash

batch_size = 16
class_nums = 10
parser = argparse.ArgumentParser(description='Deep Hashing evaluate mAP')

parser.add_argument('--bits', type=int, default=48, metavar='bts',
                    help='binary bits')
parser.add_argument('--path', type=str, default='model', metavar='P',
                    help='path directory')
args = parser.parse_args()

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
    return train_dataloader, val_dataloader

def binary_output(dataloader):
    net = resnet50_hash.resnet50(False,10)
    net.load_state_dict(torch.load('./weight/40xxxxx.pth'))
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        net.cuda()
    full_batch_output = torch.cuda.FloatTensor()
    full_batch_label = torch.cuda.LongTensor()
    #eval 测试模式 训练模式和测试模式要切换
    net.eval()
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs= net(inputs)
        full_batch_output = torch.cat((full_batch_output, outputs.data), 0)
        full_batch_label = torch.cat((full_batch_label, targets.data), 0)
        print(torch.round(full_batch_output), full_batch_label)
    return torch.round(full_batch_output), full_batch_label

def precision(trn_binary, trn_label, tst_binary, tst_label):
    trn_binary = trn_binary.cpu().numpy()
    trn_binary = np.asarray(trn_binary, np.int32)
    trn_label = trn_label.cpu().numpy()
    tst_binary = tst_binary.cpu().numpy()
    tst_binary = np.asarray(tst_binary, np.int32)
    tst_label = tst_label.cpu().numpy()
    #classess 类别数
    classes = np.max(tst_label) + 1
    for i in range(classes):
        if i == 0:
            tst_sample_binary = tst_binary[np.random.RandomState(seed=i).permutation(np.where(tst_label==i)[0])[:100]]
            tst_sample_label = np.array([i]).repeat(100)
            continue
        else:
            #数组拼接
            tst_sample_binary = np.concatenate([tst_sample_binary, tst_binary[np.random.RandomState(seed=i).permutation(np.where(tst_label==i)[0])[:100]]])
            tst_sample_label = np.concatenate([tst_sample_label, np.array([i]).repeat(100)])

    query_times = tst_sample_binary.shape[0]
    trainset_len = trn_binary.shape[0]
    AP = np.zeros(query_times)
    precision_radius = np.zeros(query_times)
    Ns = np.arange(1, trainset_len + 1)
    sum_tp = np.zeros(trainset_len)
    hanming_thre_nums=0
    all_query_nums = 0
    right_nums = 0
    total_time_start = time.time()
    for i in range(query_times):
        # print('Query ', i+1)
        query_label = tst_sample_label[i]
        query_binary = tst_sample_binary[i,:]
        #汉明距离
        query_result = np.count_nonzero(query_binary != trn_binary, axis=1)    #don't need to divide binary length
        # for i in np.sort(query_result):
        #     print(i,end=",")
        sort_indices = np.argsort(query_result)
        buffer_yes = np.equal(query_label, trn_label[sort_indices]).astype(int)
        # P = np.cumsum(buffer_yes) / Ns
        #这个汉明距离计算有错误，应该是小于某个数，而且应该设定一定的范围
        # precision_radius[i] = P[np.where(np.sort(query_result)>2)[0][0]-1]
        # AP[i] = np.sum(P * buffer_yes) /sum(buffer_yes)
        # sum_tp = sum_tp + np.cumsum(buffer_yes)

        #计算汉明距离小于等于2的查准率和查全率
        threhold_k = 10
        all_query_nums += np.sum(query_result<=threhold_k)
        for i,j in enumerate(buffer_yes):
            if j ==1 and query_result[sort_indices[i]] <= threhold_k:
                right_nums+=1
    print('total query time = ', time.time() - total_time_start)
    print("hanming threhold  Recall is",right_nums/(trainset_len*100))
    print("hanming threhold  Precision is",right_nums/(all_query_nums))
    # precision_at_k = sum_tp / Ns / query_times
    # index = [1, 50, 100, 600, 800, 1000]
    # index = [i - 1 for i in index]
    # print('precision at k:', precision_at_k[index])
    # np.save('precision_at_k', precision_at_k)
    # print('precision within Hamming radius 2:', np.mean(precision_radius))
    # map = np.mean(AP)
    # print('mAP:', map)



if os.path.exists('./result/train_binary') and os.path.exists('./result/train_label') and \
   os.path.exists('./result/test_binary') and os.path.exists('./result/test_label') :
    train_binary = torch.load('./result/train_binary')
    train_label = torch.load('./result/train_label')
    test_binary = torch.load('./result/test_binary')
    test_label = torch.load('./result/test_label')

else:
    trainloader, testloader = load_data()
    train_binary, train_label = binary_output(trainloader)
    test_binary, test_label = binary_output(testloader)
    if not os.path.isdir('result'):
        os.mkdir('result')
    torch.save(train_binary, './result/train_binary')
    torch.save(train_label, './result/train_label')
    torch.save(test_binary, './result/test_binary')
    torch.save(test_label, './result/test_label')


precision(train_binary, train_label, test_binary, test_label)
