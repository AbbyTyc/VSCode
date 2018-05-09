import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# b=[[3,2,1],[1,2,3]]
# print(b[1][:])


# a=np.arange(16)
# print(a.reshape(4,2,2))
# a=a.reshape(2,2,4)
# print(a.shape)
# print(a.size)
#
# print(2**3)


# x=np.array([1,2,3])
# y=np.array([3,2,1])
# y.reshape(1,3)
# print(y[:])
# print(x*y)
# print(np.dot(x,y))
#
# for i in range(10):
#     print(i,end=' ')


# num = np.random.random((2,3))
# # print(num)
# for i in num:
#     print(i, end=' ')


# a=np.array([1,2])
# b=np.array([3,4])
# c=np.vstack((a,b))
# print(np.hstack((a,b)))


# print(np.r_[1:5,8,9])
# print(np.array(np.arange(1,7,1)))


# a=np.arange(1,17,1)
# b=a.reshape((2,-1))
# print(b,'\n',a,'\n')
# print(b[0][2])


# palette = np.array([[0, 0, 0], [255, 0, 0],[0, 255, 0],[0, 0, 255],[255, 255, 255]])
# a=np.array([[1,3],[2,4]])
# print('a:',a)
# print('a.T:',a.T)
# print('a.conj().T:',a.conj().T)
# print(palette[a][:][1])


# a=np.arange(10)
# b=a**2
# plt.plot(a,b)
# plt.show()



np.random.seed(1)
print(np.random.random())
# np.random.seed(1)
print(np.random.random())