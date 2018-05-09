import tensorflow as tf
import numpy as np

np.random.seed(1)

a=np.float32(np.random.rand(3,4))
b=np.float32(np.random.rand(4,3))
c=np.float32(np.random.rand(3,3))

A=tf.placeholder(tf.float32,shape=(3,4))
B=tf.placeholder(tf.float32,shape=(4,3))
C=tf.placeholder(tf.float32,shape=(3,3))


if 'session' in locals() and session is not None:
    print('Close interactive session')
    session.close()

with tf.Session() as sess:
    x1=sess.run(A,feed_dict={A:a})
    x2=sess.run(B,feed_dict={B:b})
    x3=sess.run(C,feed_dict={C:c})
    x4=sess.run(tf.matmul(x1,x2)+x3)
    print(['A:',x1])
    print(['B:',x2])
    print(['C:',x3])
    print(['AxB+C:',x4])
    