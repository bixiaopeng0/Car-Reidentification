import numpy as np
import matplotlib.pyplot as plt


f = open("./log.txt")
data_str = f.read().split('\n')
data_str.pop()
data_str.pop()
print(data_str)
data = [float(i) for i in data_str]
print(data)

x = np.arange(0,len(data))
plt.xlabel("iter",FontProperties='STKAITI',fontsize=24)
plt.ylabel("loss",FontProperties='STKAITI',fontsize=24)
plt.plot(x,data, color='r',label = 'Loss')
plt.yticks(np.arange(0,2,0.5))
plt.xticks(np.arange(0,len(data),100))
plt.show()