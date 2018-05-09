import tensorflow as tf 
import numpy as np

N=tf.truncated_normal(shape=(100,2),mean=(6,3),stddev=(1,1),dtype=tf.float32)
label=tf.ones((100,1))
data_set=tf.concat([N,label],axis=1)
# data_set=np.hstack((N,label))
# data_set=tf.Variable([N,label])

# x=N
# y=label
x=tf.placeholder(tf.float32,shape=(None,2))
y=tf.placeholder(tf.float32,shape=(None,1))
# x=tf.Variable(tf.truncated_normal(shape=(100,2),mean=(6,3),stddev=(1,1),dtype=tf.float32))
# y=tf.ones((100,1))

with tf.variable_scope("logistic_regression"):
    # w=tf.get_variable(initializer=np.random.rand(2,1),name='weight',dtype=tf.float32)
    # b=tf.get_variable(initializer=np.random.rand(1,1),name='bias',dtype=tf.float32)
    # w=tf.get_variable(shape=(2,1),name='weight',dtype=tf.float32)
    # b=tf.get_variable(shape=(1,1),name='bias',dtype=tf.float32)
    w=tf.Variable(np.random.rand(2,1),name='weight',dtype=tf.float32)
    b=tf.Variable(np.random.rand(1,1),name='bias',dtype=tf.float32)

    logits=tf.matmul(x,w)+b
    loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=logits))
    # loss=tf.reduce_mean(loss)

opt=tf.train.AdamOptimizer(0.01)
train_op=opt.minimize(loss)


with tf.Session() as sess:
    # print(N)
    sess.run(tf.global_variables_initializer())
    # sess.run(N)
    print(sess.run(w))
    # data=sess.run(data_set)
    # print(data)
    # print(data_set[:][1])
    # print(x,N)
    # print(y,label)
    # print(w,b)
    data=sess.run(data_set)
    # print(data[:, 0:2])
    # print(data[:,2].reshape((-1,1)).shape)
    for i in range(500):
        train,loss_=sess.run([train_op,loss],feed_dict={x:data[:,0:2], y:data[:,2].reshape((-1,1))})
        print(sess.run(w))
        # train=sess.run(train_op)
        