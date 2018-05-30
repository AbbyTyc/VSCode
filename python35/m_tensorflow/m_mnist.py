import tensorflow as tf
import matplotlib.pyplot as plt
# 导入input_data用于自动下载和安装MNIST数据集
from tensorflow.examples.tutorials.mnist import input_data


# data input
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
mnist = input_data.read_data_sets("D:\python\m_minist\data", one_hot=True)
# print(mnist.validation)
img0=mnist.validation.images[0].reshape(28,28)
fig=plt.figure(figsize=(10,10))
ax0=fig.add_subplot(221)
ax0.imshow(img0)
# ax0=fig.add_axes()
# ax0.imshow(img0)
fig.show()

# # data set
# x=tf.placeholder("float",[None,784])
# W=tf.Variable(tf.zeros([784,10]))
# b=tf.Variable(tf.zeros([10]))

# y = tf.nn.softmax(tf.matmul(x,W)+b)

# y_ = tf.placeholder("float",[None,10])    

# cross_entropy = -tf.reduce_sum(y_*tf.log(y))

# # optimizer set
# optimizer=tf.train.GradientDescentOptimizer(0.01)
# train_step=optimizer.minimize(cross_entropy)

# # data init
# init=tf.initialize_all_variables()

# # start a session
# sess=tf.Session()
# sess.run(init)  
# for i in range(1000):
#     batch_xs,batch_ys=mnist.train.next_batch(100)
#     sess.run(train_step,feed_dict={x:batch_xs, y_:batch_ys})


# # evaluation
# correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
# accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float"))
# print(sess.run(accuracy,feed_dict={x:mnist.test.images, y_:mnist.test.labels}))

# sess.close()
