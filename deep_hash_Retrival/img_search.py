import torch
from torchvision import datasets, models, transforms
import numpy as np
from PIL import Image
import argparse
import os
import resnet50_hash

train_binary = torch.load('./result/train_binary')
train_data = torch.load('./result/train_data')
parser = argparse.ArgumentParser(description='Image Search')
parser.add_argument('--pretrained', type=str, default=92, metavar='pretrained_model',
                    help='')
parser.add_argument('--querypath', type=str, default='./4404000000002940408492.jpg', metavar='',
                    help='')
args = parser.parse_args()

transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        # transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((.4, .4, .4), (.08, .08, .08))
    ])

query_pic = Image.open(args.querypath)
query_pic = transform_test(query_pic)

net = resnet50_hash.resnet50(False, 10)
net.load_state_dict(torch.load('./weight/40xxxxx.pth'))

use_cuda = torch.cuda.is_available()
if use_cuda:
    net.cuda()
    query_pic = query_pic.cuda().unsqueeze(0)
net.eval()
#经过sigmoid的激活层
outputs, _ = net(query_pic)
print("48feature",outputs[0])
query_binary = (outputs[0] > 0.5).cpu().numpy()
#hash二值
# print("hash_binary",query_binary)
#print query_binary

#traub_binary 维度（9000，48）每一张图片对应一个hash值
trn_binary = train_binary.cpu().numpy()

print("train_binary",trn_binary,trn_binary.shape)

# 汉明距离
query_result = np.count_nonzero(query_binary != trn_binary, axis=1)    #don't need to divide binary length



# print(query_result)


#欧式距离
def eucldist_generator(coords1, coords2):
    """ Calculates the euclidean distance between 2 lists of coordinates. """
    return sum((x - y)**2 for x, y in zip(coords1, coords2))**0.5






# for i in sort_indices:
#     print("欧氏距离:",eucldist_generator(train_data[i],query_binary),"汉明距离",query_result[i])



img_list = open('./mydataset/train.txt').readlines()


#粗检索
def coarse_research():
    # argsort 返回的是从小到大的索引值
    sort_indices = np.argsort(query_result)
    #根据相似性从大到小排序
    for i in sort_indices:
        print(img_list[i])
#粗检索-细检索
def fine_research():
    # 欧式距离细检索
    # 键-索引 值-欧式距离
    query_result_eu = {}
    for i, j in enumerate(train_data):
        #只对汉明距离为0进行检索
        if query_result[i] == 0:
            query_result_eu[i] = (eucldist_generator(j, outputs[0]).data)

    query_result_list = list(query_result_eu.items())
    query_result_list.sort(key=lambda x: x[1], reverse=False)
    print(query_result_list)
    for i in query_result_list:
        print(img_list[i[0]])
if __name__ == '__main__':
    # coarse_research()
    fine_research()