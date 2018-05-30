import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

def model_set(x,w1,b1,w2,b2):
    y=tf.nn.relu(tf.matmul(x,w1)+b1)
    g=tf.nn.relu(tf.matmul(y,w2)+b2)

    return g


mnist = input_data.read_data_sets("D:\python\m_minist\data", one_hot=True)

tf.reset_default_graph()
restore_graph=tf.Graph()

with tf.Session(graph=restore_graph) as sess:
    saver = tf.train.import_meta_graph("./model.ckpt.meta")
    saver.restore(sess,"./model.ckpt")
    x1 = tf.get_default_graph().get_tensor_by_name("x1:0")
    x2 = tf.get_default_graph().get_tensor_by_name("x2:0")
    w1=tf.get_default_graph().get_tensor_by_name("mnist_model/w1:0")
    b1 = tf.get_default_graph().get_tensor_by_name("mnist_model/b1:0")
    w2 = tf.get_default_graph().get_tensor_by_name("mnist_model/w2:0")
    b2 = tf.get_default_graph().get_tensor_by_name("mnist_model/b2:0")

    x_1, y_1 = mnist.train.next_batch(1)
    x_2, y_2 = mnist.train.next_batch(1)
    g1 = model_set(x_1,w1,b1,w2,b2)
    g2 = model_set(x_2,w1,b1,w2,b2)
    ew = tf.sigmoid(tf.sqrt(tf.reduce_sum(tf.square(g1 - g2), 1)))
    # loss = tf.round(tf.nn.sigmoid(tf.reduce_mean(ew), name="loss_test"))
    # loss = tf.square(ew) * 2 / q * (1 - y) + y * 2 * q * tf.exp(-2.77 / q * ew)
    pred = sess.run(ew, feed_dict={x1: x_1, x2: x_2})
    # loss = test(x_1, x_2)
    print("x1_label:", y_1, "       x2_label:", y_2)
    print("pred_output:", pred)


    #
    # data_test = tf.get_default_graph().get_tensor_by_name("test_data_set:0")
    #
    # loss_test = tf.get_default_graph().get_tensor_by_name("logistic_regression/loss_valid:0")
    # data = sess.run(data_test)
    # # print(data)
    # loss_val_valid = sess.run([loss_test], feed_dict={test_x: data[:, 0:2], test_y: data[:, 2].reshape((-1, 1))})
    # ACC=tf.get_default_graph().get_tensor_by_name("logistic_regression/ACC:0")
    # print("test acc: ",sess.run(ACC))
    # # print("test_data_set: ",sess.run(tf.get_default_graph().get_tensor_by_name("logistic_regression/ACC:0")))
    #
    # w_= sess.run(tf.get_default_graph().get_tensor_by_name("logistic_regression/weight:0"))
    # b_ = sess.run(tf.get_default_graph().get_tensor_by_name("logistic_regression/bias:0"))
