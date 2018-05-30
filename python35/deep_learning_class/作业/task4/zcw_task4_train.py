from zcw_task4_lenet import Lenet

import tensorflow as tf
# import numpy as np
import matplotlib.pyplot as plt
# from sklearn.utils import shuffle

# 导入input_data用于自动下载和安装MNIST数据集
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("D:\python\m_minist\data", one_hot=True)

lenet=Lenet(0,0.1,True)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    m_acc=[]
    m_loss=[]
    for i in range(20000):
        X,Y = mnist.train.next_batch(50)
        train,loss,ACC=sess.run([lenet.train_step,lenet.cross_entropy,lenet.accuracy],feed_dict={lenet.x_:X, lenet.y_:Y})
        if i%500==0:
            m_acc.append(ACC)
            m_loss.append(loss)
            print("epoch:",i,"   loss:",loss,"          acc:",ACC)

    saver = tf.train.Saver()
    saver.save(sess, "./model.ckpt")


# plot
plt.figure(2)
line1, =plt.plot(m_acc,label='acc',linestyle='-')
line2, =plt.plot(m_loss,label='loss',linestyle='--')
# line3, =plt.plot(val_loss,label='val_loss',linestyle='-.')
plt.legend(handles=[line1,line2], loc=2)
plt.xlabel('epoch/500')
plt.show()