import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt


def get_loss():
    # with tf.variable_scope("logistic_regression") as scope:
        # w=tf.Variable(np.random.rand(2,1),name='weight',dtype=tf.float32)
        # b=tf.Variable(np.random.rand(1,1),name='bias',dtype=tf.float32)
    w = tf.get_variable(shape=(2, 1), name='weight', dtype=tf.float32)
    b = tf.get_variable(shape=(1, 1), name='bias', dtype=tf.float32)
    # scope.reuse_variables()

    # logits = tf.matmul(x, w) + b
    # loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)

    return w,b


# data initial
# train data
train_data1=tf.truncated_normal(shape=(100,2),mean=(3,6),stddev=(1,1),dtype=tf.float32,name="train_data1")
train_data2=tf.truncated_normal(shape=(100,2),mean=(6,3),stddev=(1,1),dtype=tf.float32,name="train_data2")
train_label1=tf.ones((100,1),name="train_label1")
train_label2=tf.zeros((100,1),name="train_label2")
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
test_data1=tf.truncated_normal(shape=(30,2),mean=(3,6),stddev=(1,1),dtype=tf.float32,name="test_data1")
test_data2=tf.truncated_normal(shape=(30,2),mean=(6,3),stddev=(1,1),dtype=tf.float32,name="test_data2")
test_label1=tf.ones((30,1),name="test_label1")
test_label2=tf.zeros((30,1),name="test_label2")
test_data=tf.concat([test_data1,test_data2],axis=0)
test_label=tf.concat([test_label1,test_label2],axis=0)
test_data_set=tf.concat([test_data,test_label],axis=1,name="test_data_set")



# placeholder
train_x=tf.placeholder(tf.float32,shape=(None,2),name="train_x")
train_y=tf.placeholder(tf.float32,shape=(None,1),name="train_y")
valid_x=tf.placeholder(tf.float32,shape=(None,2),name="valid_x")
valid_y=tf.placeholder(tf.float32,shape=(None,1),name="valid_y")



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
        # train optimizer
        w, b = get_loss()
        logits = tf.matmul(train_x, w) + b
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=train_y, logits=logits)
        loss_mean = tf.reduce_mean(loss)
        opt = tf.train.AdamOptimizer(0.01)
        train_op = opt.minimize(loss_mean)

        # valid setting
        scope.reuse_variables()
        w, b = get_loss()
        logits_valid = tf.matmul(valid_x, w) + b
        loss_valid = tf.nn.sigmoid(logits_valid,name="loss_valid")
        # print(loss_valid)
        # loss_mean_valid = tf.reduce_mean(loss_valid)

        # variables initial
        sess.run(tf.global_variables_initializer())
        data_train, train_data1_, train_data2_ = sess.run([train_data_set, train_data1, train_data2])
        data_valid=sess.run(valid_data_set)

        # data = data_valid
        # # print(data)
        # loss_val_valid = sess.run([loss_valid], feed_dict={valid_x: data[:, 0:2], valid_y: data[:, 2].reshape((-1, 1))})
        # correct_prediction = tf.equal(tf.round(loss_val_valid[0]), data_valid[:, 2].reshape((-1, 1)))
        # # print((sess.run(correct_prediction)))
        # ACC = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="ACC")

        # begin training and validating
        for i in range(500):
            # data=data_train
            # data=[data_train[(i%5)*20:(i%5+1)*20,:],data_train[((i%5)*20+100):((i%5+1)*20+100),:]][0]
            data = np.vstack([data_train[(i%5)*20:(i%5+1)*20,:], data_train[((i%5)*20+100):((i%5+1)*20+100),:]])
            # print(data.shape)
            train,loss_val=sess.run([train_op,loss_mean],feed_dict={train_x:data[:,0:2], train_y:data[:,2].reshape((-1,1))})
            # print(loss_val)

            data = data_valid
            # print(data)
            loss_val_valid = sess.run([loss_valid], feed_dict={valid_x: data[:, 0:2], valid_y: data[:, 2].reshape((-1, 1))})
            # # correct_prediction = tf.equal(tf.round(loss_val_valid[0]), data_valid[:, 2].reshape((-1,1)))
            # # # print((sess.run(correct_prediction)))
            # # ACC = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="ACC")

            if i%100==99 and i>0:
                correct_prediction = tf.equal(tf.round(loss_val_valid[0]), data_valid[:, 2].reshape((-1, 1)))
                # print((sess.run(correct_prediction)))
                ACC = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="ACC")

                # # num = 0
                # for j in range(len(loss_val_valid[0])):
                #     if data_valid[j, 2] == 1 and loss_val_valid[0][j] > 0.5:
                #         # num += 1
                #         acc+=1
                #     if data_valid[j, 2] == 0 and loss_val_valid[0][j] <= 0.5:
                #         # num += 1
                #         acc+=1

                print("epoch",i+1,"   train loss:",loss_val,"     validation accuracy:", sess.run(ACC))
    w_ = sess.run(w)
    b_ = sess.run(b)

    # model save
    saver = tf.train.Saver()
    saver.save(sess,"./model.ckpt")



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

