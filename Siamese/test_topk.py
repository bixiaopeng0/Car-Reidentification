'''
time:2019/12.04
测试查topk
'''

import torch
import torch.nn.functional as F
import time

test_data = torch.load('./test_binary')
test_label = torch.load('./test_label')


#欧式距离
def eucldist_generator(coords1, coords2):
    """ Calculates the euclidean distance between 2 lists of coordinates. """
    return sum((x - y)**2 for x, y in zip(coords1, coords2))**0.5

def cal_top(topk):
    str1 = []

    right_nums = 0
    retrieval_nums = 0
    for i in range(1000):
        start_time = time.time()
        dis_list = []

        for j in range(1000):
            if i == j:
                continue
            else:
                dis = eucldist_generator(test_data[i], test_data[j])
                dis_list.append(dis)
        sort_list =  sorted(range(len(dis_list)), key=lambda k: dis_list[k])
        for t in range(topk):
            if sort_list[t]//10 == i//10:
                right_nums+=1
        # print(sort_list)
        print("nums:",i,"time:",time.time()-start_time)
    print("top",topk,"准确率",right_nums/(topk*1000))




if __name__ == '__main__':
    cal_top(5)



# if __name__== "__main__":
#     cal_top(5)