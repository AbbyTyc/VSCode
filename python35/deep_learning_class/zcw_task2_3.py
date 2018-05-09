import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt


def get_loss(x):
    # with tf.variable_scope("logistic_regression") as scope:
        # w=tf.Variable(np.random.rand(2,1),name='weight',dtype=tf.float32)
        # b=tf.Variable(np.random.rand(1,1),name='bias',dtype=tf.float32)
    w = tf.get_variable(shape=(2, 1), name='weight', dtype=tf.float32)
    b = tf.get_variable(shape=(1, 1), name='bias', dtype=tf.float32)
    # scope.reuse_variables()

    logits = tf.matmul(x, w) + b
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)

    return loss,w,b


# data initial
# train data
train_data1=tf.truncated_normal(shape=(100,2),mean=(3,6),stddev=(1,1),dtype=tf.float32)
train_data2=tf.truncated_normal(shape=(100,2),mean=(6,3),stddev=(1,1),dtype=tf.float32)
train_label1=tf.ones((100,1))
train_label2=tf.zeros((100,1))
train_data=tf.concat([train_data1,train_data2],axis=0)
train_label=tf.concat([train_label1,train_label2],axis=0)
train_data_set=tf.concat([train_data,train_label],axis=1)
# validation data
valid_data1=tf.truncated_normal(shape=(30,2),mean=(3,6),stddev=(1,1),dtype=tf.float32)
valid_data2=tf.truncated_normal(shape=(30,2),mean=(6,3),stddev=(1,1),dtype=tf.float32)
valid_label1=tf.ones((30,1))
valid_label2=tf.zeros((30,1))
valid_data=tf.concat([valid_data1,valid_data2],axis=0)
valid_label=tf.concat([valid_label1,valid_label2],axis=0)
valid_data_set=tf.concat([valid_data,valid_label],axis=1)
# test data
test_data1=tf.truncated_normal(shape=(30,2),mean=(3,6),stddev=(1,1),dtype=tf.float32)
test_data2=tf.truncated_normal(shape=(30,2),mean=(6,3),stddev=(1,1),dtype=tf.float32)
test_label1=tf.ones((30,1))
test_label2=tf.zeros((30,1))
test_data=tf.concat([test_data1,test_data2],axis=0)
test_label=tf.concat([test_label1,test_label2],axis=0)
test_data_set=tf.concat([test_data,test_label],axis=1)



# placeholder
x=tf.placeholder(tf.float32,shape=(None,2))
y=tf.placeholder(tf.float32,shape=(None,1))



# # optimizer
# loss,w,b=get_loss(x)
# loss = tf.reduce_mean(loss)
# opt=tf.train.AdamOptimizer(0.01)
# train_op=opt.minimize(loss)



# Session
with tf.Session() as sess:
    # sess.run(tf.global_variables_initializer())
    # data_train,train_data1_,train_data2_=sess.run([train_data_set,train_data1,train_data2])
    with tf.variable_scope("logistic_regression") as scope:
        # optimizer
        loss, w, b = get_loss(x)
        loss_mean = tf.reduce_mean(loss)
        opt = tf.train.AdamOptimizer(0.01)
        train_op = opt.minimize(loss_mean)

        sess.run(tf.global_variables_initializer())
        data_train, train_data1_, train_data2_ = sess.run([train_data_set, train_data1, train_data2])
        # scope.reuse_variables()
        # loss, w, b = get_loss(x)
        # print(sess.run(loss))
        # # print(tf.get_variable('weight'))

    for i in range(500):
        # data=data_train
        # data=[data_train[(i%5)*20:(i%5+1)*20,:],data_train[((i%5)*20+100):((i%5+1)*20+100),:]][0]
        data = np.vstack([data_train[(i%5)*20:(i%5+1)*20,:], data_train[((i%5)*20+100):((i%5+1)*20+100),:]])
        # print(data.shape)
        train,loss_val=sess.run([train_op,loss_mean],feed_dict={x:data[:,0:2], y:data[:,2].reshape((-1,1))})
        # print(loss_val)
    w_ = sess.run(w)
    b_ = sess.run(b)



# plot
plt.plot(train_data1_[:,0],train_data1_[:,1],'b+')
plt.plot(train_data2_[:,0],train_data2_[:,1],'go')
y1=(-b_ - w_[0]*1)/w_[1]
y2=(-b_ - w_[0]*7)/w_[1]
xx=[1,7]
# print(xx)
yy=[y1[0],y2[0]]
# print(y1[0])
plt.plot(xx,yy,'r')
plt.show()

