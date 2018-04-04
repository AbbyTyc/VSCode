import tensorflow as tf

a=tf.constant([[1,2]])
b=tf.constant([[3],[4]])

c=tf.matmul(a,b)

with tf.Session() as sess:
    result=sess.run(c)
    print(result)
 
