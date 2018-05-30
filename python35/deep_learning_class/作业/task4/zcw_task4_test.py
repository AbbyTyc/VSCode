from zcw_task4_lenet import Lenet

import tensorflow as tf
from tensorflow.contrib.layers import flatten
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

# 导入input_data用于自动下载和安装MNIST数据集
from tensorflow.examples.tutorials.mnist import input_data



mnist = input_data.read_data_sets("D:\python\m_minist\data", one_hot=True)
# x = tf.placeholder("float", [None, 784])
# y_ = tf.placeholder("float", [None, 10])

tf.reset_default_graph()
restore_graph=tf.Graph()

# lenet=Lenet(0,0.1,False)

with tf.Session(graph=restore_graph) as sess:
    saver = tf.train.import_meta_graph("./model.ckpt.meta")
    saver.restore(sess, "./model.ckpt")
    X, Y = mnist.test.next_batch(1000)
    x_ = tf.get_default_graph().get_tensor_by_name("Lenet_var/x_:0")
    y_ = tf.get_default_graph().get_tensor_by_name("Lenet_var/y_:0")
    ACC = tf.get_default_graph().get_tensor_by_name("Lenet_var/test_accuracy:0")

    print("test acc: ", sess.run(ACC,feed_dict={x_:X, y_:Y}))
