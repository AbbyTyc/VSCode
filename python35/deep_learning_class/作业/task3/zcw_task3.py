import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt



# 导入input_data用于自动下载和安装MNIST数据集
from tensorflow.examples.tutorials.mnist import input_data


def model_set(x,reuse_flag):
    with tf.variable_scope('mnist_model',reuse=reuse_flag):
        w1 = tf.get_variable(name="w1", initializer=tf.random_normal(shape=[784, 500]))
        b1 = tf.get_variable(name="b1", initializer=tf.random_normal(shape=[500]))
        w2 = tf.get_variable(name="w2", initializer=tf.random_normal(shape=[500, 10]))
        b2 = tf.get_variable(name="b2", initializer=tf.random_normal(shape=[10]))
        # w1 = tf.get_variable(shape=(784, 500), name='w1', dtype=tf.float32)
        # b1 = tf.get_variable(shape=(1, 500), name='b1', dtype=tf.float32)
        # w2 = tf.get_variable(shape=(500, 10), name='w2', dtype=tf.float32)
        # b2 = tf.get_variable(shape=(1, 10), name='b2', dtype=tf.float32)

        # scope.reuse_variables()

        y1=tf.nn.relu(tf.matmul(x,w1)+b1)
        # loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=train_y, logits=logits)
        g=tf.nn.relu(tf.matmul(y1,w2)+b2)


    return g



def get_loss(g1,g2,y):
    q=tf.constant([5],tf.float32)
    a=tf.constant([2],tf.float32)
    b = tf.constant([-2.77], tf.float32)
    bias=tf.square(g1-g2)
    ew= tf.sqrt(tf.reduce_sum(bias,1))
    # ew = tf.reduce_sum(bias, 1)
    # print(ew)
    loss1=tf.square(ew)*(1-y)*a/q+y*a*q*tf.exp(ew*b/q)
    # print(loss1)
    loss2=tf.reduce_mean(tf.sigmoid(loss1))
    # print(loss2)


    return bias,ew,loss1,loss2


def train():
    x1 = tf.placeholder(tf.float32, shape=(None, 784), name="x1")
    x2 = tf.placeholder(tf.float32, shape=(None, 784), name="x2")
    y1 = tf.placeholder(tf.float32, shape=(None, 10), name="y1")
    y2 = tf.placeholder(tf.float32, shape=(None, 10), name="y2")

    g1=model_set(x1,False)
    # print(g1)
    g2=model_set(x2,True)

    y_equal_bool=tf.not_equal(y1, y2)
    # print(y_equal_bool)
    y_equal_float=tf.reduce_sum(tf.cast(y_equal_bool, tf.float32),1)
    # print(y_equal_float)
    y_=tf.cast(y_equal_float>0, tf.float32)


    # q1=np.array([[1,2],[3,4],[5,6]])
    # q2 = np.array([[1, 1], [3, 4], [1, 6]])
    # q = np.array(q1 != q2, dtype=np.float32)
    # print(q)
    # y_=np.sum(np.array(y1!=y2, dtype = np.float32),0)
    # print(y_)
    bias,ew1,loss1,loss2=get_loss(g1,g2,y_)

    opt=tf.train.AdamOptimizer(learning_rate=0.01)
    train_opt=opt.minimize(loss2)

    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(6000):
            batch_size = 64
            x_1, y_1 = mnist.train.next_batch(batch_size)
            x_2, y_2 = mnist.train.next_batch(batch_size)
            # if i%100==1:
            #     x_1=x_2
            #     y_1=y_2
            # x, y = mnist.train.next_batch(30)
            # x_1 = np.vstack([x_1, x])
            # y_1 = np.vstack([y_1, y])
            # # print(y_1.shape)
            # x_2 = np.vstack([x_2, x])
            # y_2 = np.vstack([y_2, y])

            train_opt_, ew1_,loss1_,loss2_,g1_,g2_,y_3,bias_ =sess.run([train_opt, ew1,loss1,loss2,g1,g2,y_,bias],feed_dict={x1:x_1,x2:x_2,y1:y_1,y2:y_2})
            # print(y_3)
            if i%200==0:
                # print(y_3)
                # print(ew1_ ,loss1_,"   ",loss2_,'\n')#,"    ",g1_,"     ",g2_)
                print("loss: ",loss2_)

        # model save
        saver = tf.train.Saver()
        saver.save(sess, "./model.ckpt")

        # x_1, y_1 = mnist.train.next_batch(1)
        # x_2, y_2 = mnist.train.next_batch(1)
        # g1 = model_set(x_1, True)
        # g2 = model_set(x_2, True)
        # ew2 = tf.sigmoid(tf.sqrt(tf.reduce_sum(tf.square(g1 - g2), 1)))
        # loss = tf.round(tf.nn.sigmoid(tf.reduce_mean(ew), name="loss_test"))
        # pred,g1_,g2_ = sess.run([ew2,g1,g2], feed_dict={x1: x_1, x2: x_2})
        # loss = test(x_1, x_2)
        # print("y1:", y_1, "       y2:", y_2)
        # print("y1:", np.argmax(y_1), "       y2:", np.argmax(y_2))
        # print("pred:", pred)
        # print("g1:",g1_,"   g2:",g2_)


def test(x_1,x_2):
    g1 = model_set(x1,True)
    g2 = model_set(x2,True)
    ew = tf.sqrt(tf.reduce_sum(tf.square(g1 - g2), 1))
    loss=tf.round(tf.nn.sigmoid(tf.reduce_mean(ew),name="loss_test"))
    with tf.Session() as sess:
        pred=sess.run(loss,feed_dict={x1:x_1,x2:x_2})

    return pred


# data input
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
mnist = input_data.read_data_sets("D:\python\m_minist\data", one_hot=True)
img0=mnist.validation.images[0].reshape(28,28)
fig=plt.figure(figsize=(10,10))
ax0=fig.add_subplot(221)
ax0.imshow(img0)
fig.show()

# data set
# x1=tf.placeholder("float",[None,784],name="x1")
# x2=tf.placeholder("float",[None,784],name="x2")

train()

