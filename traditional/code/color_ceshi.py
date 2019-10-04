#对一个文件夹下的图片进行测试，看看这个识别颜色的程序效果咋样

import os
import cv2
import identification_color
#感觉金色的测不准，先把流程走通吧，以后再针对金色想办法
#这里面有些样本本身数据也不咋地，有一定的倾斜角度
#black img path
# img_path = "G:/19\lab\python\Vehicle Re-identification\doc\dataset\VRID\image/1\License_6/"
#金色
# img_path = "G:/19\lab\python\Vehicle Re-identification\doc\dataset\VRID\image/4\License_302/"
#白色
# img_path = 'G:/19\lab\python\Vehicle Re-identification\doc\dataset\VRID\image/6\License_522/'
#红色
# img_path = 'G:/19\lab\python\Vehicle Re-identification\doc\dataset\VRID\image/2\License_109/'
#绿色
img_path = 'G:/19\lab\python\Vehicle Re-identification\doc\dataset\VRID\image/3\License_203/'

str = []
str1 = {'blank': 0, 'gray': 0, 'white': 0, 'red': 0,
              'origin': 0, 'yellow': 0, 'green': 0, 'qing': 0,
              'blue': 0,
              'purple': 0}
dirs = os.listdir(img_path)
for dir in dirs:
    #将文件名和后缀名分隔开
    if(os.path.splitext(dir)[1] == '.jpg' or os.path.splitext(dir)[1] == '.png'):
        str.append(dir)

def test_img():
    for i in str:
        # print(i)
        f = open(img_path+os.path.splitext(i)[0]+"color"+".txt",'w')
        img = cv2.imread(img_path+i)
        color = identification_color.recog_car_color(img)
        for j in color:
            f.write(j)
            f.write('\n')
        f.close()
        f = open(img_path + os.path.splitext(i)[0] + "color" + ".txt", 'r')
        s = f.read().split('\n')
        s.pop()
        print("s",s)
        f.close()
        #     str1[i]+=1
        # print(color)
        # str1[color]+=1
    print(str1)












if __name__ == '__main__':
    test_img()