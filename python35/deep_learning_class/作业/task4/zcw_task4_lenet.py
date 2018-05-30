import tensorflow as tf
from tensorflow.contrib.layers import flatten
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.utils import shuffle

# # 导入input_data用于自动下载和安装MNIST数据集
# from tensorflow.examples.tutorials.mnist import input_data


class Lenet:
    def __init__(self,mu,sigma,is_train):
        # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
        self.mu = mu
        self.sigma = sigma
        self.is_train = is_train

        if is_train==True:
            is_train=False
        else:
            is_train=True

        with tf.variable_scope("Lenet_var",reuse=is_train) as scope:
            self.x_ = tf.placeholder("float", [None, 784],name='x_')
            x=self.x_
            self.y_ = tf.placeholder("float", [None, 10],name='y_')
            x = tf.reshape(x, [-1, 28, 28, 1])  # reshape the image to 4d
            x = tf.pad(x, ((0, 0), (2, 2), (2, 2), (0, 0)), 'CONSTANT')
            self.x = x

            y_conv = self.net_build()
            if self.is_train:
                cross_entropy, train_step, self.accuracy=self.train(y_conv)
                self.cross_entropy=cross_entropy
                self.train_step=train_step
                # self.accuracy=accuracy
                # print(accuracy)
            else:
                accuracy=self.test(y_conv)
                self.accuracy = accuracy

    def layer1_conv(self, x):
        # SOLUTION: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
        conv1_w=tf.Variable(tf.truncated_normal(shape=(5,5,1,6),mean=self.mu,stddev=self.sigma))
        conv1_b=tf.Variable(tf.zeros(6))
        conv1=tf.nn.conv2d(x,conv1_w,strides=[1,1,1,1],padding='VALID')+conv1_b

        # SOLUTION: Activation.
        conv1=tf.nn.relu(conv1)

        # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
        conv1=tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')

        return conv1

    def layer2_conv(self,conv1):
        # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
        conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=self.mu, stddev=self.sigma))
        conv2_b = tf.Variable(tf.zeros(16))
        conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

        # SOLUTION: Activation.
        conv2 = tf.nn.relu(conv2)

        # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        return conv2

    def layer3_full_connect(self,fc0):
        # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
        fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=self.mu, stddev=self.sigma))
        fc1_b = tf.Variable(tf.zeros(120))
        fc1 = tf.matmul(fc0, fc1_W) + fc1_b

        # SOLUTION: Activation.
        fc1 = tf.nn.relu(fc1)
        # fc1 = tf.nn.dropout(fc1, 0.5)

        return fc1

    def layer4_full_connect(self,fc1):
        # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
        fc2_W = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=self.mu, stddev=self.sigma))
        fc2_b = tf.Variable(tf.zeros(84))
        fc2 = tf.matmul(fc1, fc2_W) + fc2_b

        # SOLUTION: Activation.
        fc2 = tf.nn.relu(fc2)
        # fc2 = tf.nn.dropout(fc2, 0.5)

        return fc2

    def layer5_full_connect(self,fc2):
        # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 10.
        fc3_W = tf.Variable(tf.truncated_normal(shape=(84, 10), mean=self.mu, stddev=self.sigma))
        fc3_b = tf.Variable(tf.zeros(10))
        logits = tf.matmul(fc2, fc3_W) + fc3_b

        return logits

    def optim(self,y_conv):
        # optimizer
        cross_entropy = -tf.reduce_mean(self.y_ * tf.log(y_conv))  # jiao cha shang
        optimizer = tf.train.AdamOptimizer(1e-4)
        train_step = optimizer.minimize(cross_entropy)

        return cross_entropy,train_step

    def net_build(self):
        with tf.name_scope('lenet') as scope:
            conv1=self.layer1_conv(self.x)
            conv2=self.layer2_conv(conv1)

            # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
            fc0 = flatten(conv2)

            fc1=self.layer3_full_connect(fc0)
            fc2=self.layer4_full_connect(fc1)
            logits=self.layer5_full_connect(fc2)

            soft_max=tf.nn.softmax(logits)

        return soft_max

    def acc(self,y_conv):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(self.y_, 1),name='correct_pred')
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name='test_accuracy')
        return  accuracy

    def train(self,y_conv):
        # y_conv = self.net_build()
        cross_entropy, train_step=self.optim(y_conv)
        accuracy=self.acc(y_conv)
        # print(accuracy)
        return cross_entropy, train_step, accuracy

    def test(self,y_conv):
        # y_conv = self.net_build()
        accuracy=self.acc(y_conv)
        # print(accuracy)
        return accuracy











