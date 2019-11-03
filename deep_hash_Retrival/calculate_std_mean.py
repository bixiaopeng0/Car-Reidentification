# coding:utf-8

#将图片地址路径保存到train。txt文件里

import os
import cv2
import numpy as np
#三层路径  train 1  lisense
train_dir = './mydataset/train/'
train_txt_dir = './mydataset/train.txt'
test_dir = './mydataset/test/'
test_txt_dir = './mydataset/test.txt'
def get_img_dir(dir,write_dir):
    f = open(write_dir,'w')
    dirs = os.listdir(dir)
    for path1 in dirs:
        train_path = os.path.join(dir,path1)
        license_dirs = os.listdir(train_path)
        for path2 in license_dirs:
            license_path = os.path.join(train_path,path2)
            img_dir = os.listdir(license_path)
            for path3 in img_dir:
                img_path = os.path.join(license_path,path3)
                f.write(img_path)
                f.write('\n')
    f.close()

#计算均值 方差
def get_img_std_mean(img_dir):
    means = [0,0,0]
    stds = [0,0,0]
    f = open(img_dir)
    img_list = f.read().split('\n')
    img_list.pop()
    img_nums = len(img_list)
    print(img_list)
    print(img_nums)
    for img_path in img_list:
        img = cv2.imread(img_path)
        img = img.astype(np.float32)/255
        for i in range(3):
            ## BGR --> RGB
            means[2-i] += img[i, :, :].mean()
            stds[2-i] += img[i, :, :].std()
    means = np.asarray(means) / img_nums
    stds = np.asarray(stds) / img_nums
    print("{} : normMean = {}".format(type, means))
    print("{} : normstdevs = {}".format(type, stds))

# get_img_std_mean(test_txt_dir)
#获取路径
get_img_dir(train_dir,train_txt_dir)
get_img_dir(test_dir,test_txt_dir)