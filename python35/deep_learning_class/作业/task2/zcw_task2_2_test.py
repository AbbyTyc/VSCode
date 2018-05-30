import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.reset_default_graph()
restore_graph=tf.Graph()

with tf.Session(graph=restore_graph) as sess:
    saver = tf.train.import_meta_graph("./model.ckpt.meta")
    saver.restore(sess,"./model.ckpt")
    test_x=tf.get_default_graph().get_tensor_by_name("valid_x:0")
    test_y = tf.get_default_graph().get_tensor_by_name("valid_y:0")
    data_test = tf.get_default_graph().get_tensor_by_name("test_data_set:0")

    loss_test = tf.get_default_graph().get_tensor_by_name("logistic_regression/loss_valid:0")
    data = sess.run(data_test)
    # print(data)
    loss_val_valid = sess.run([loss_test], feed_dict={test_x: data[:, 0:2], test_y: data[:, 2].reshape((-1, 1))})
    ACC=tf.get_default_graph().get_tensor_by_name("logistic_regression/ACC:0")
    print("test acc: ",sess.run(ACC))
    # print("test_data_set: ",sess.run(tf.get_default_graph().get_tensor_by_name("logistic_regression/ACC:0")))

    w_= sess.run(tf.get_default_graph().get_tensor_by_name("logistic_regression/weight:0"))
    b_ = sess.run(tf.get_default_graph().get_tensor_by_name("logistic_regression/bias:0"))

# plot
plt.plot(data[0:30,0],data[0:30,1],'b+')
plt.plot(data[30:60,0],data[30:60,1],'go')
y1=(-b_ - w_[0]*1)/w_[1]
y2=(-b_ - w_[0]*7)/w_[1]
xx=[1,7]
# print(xx)
yy=[y1[0],y2[0]]
# print(y1[0])
plt.plot(xx,yy,'r')
plt.show()