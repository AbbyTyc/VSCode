import tensorflow as tf
import numpy as np 
import traceback

a=tf.Variable(np.random.randint(10))
b=tf.Variable(np.random.randint(10))
c=tf.Variable(np.random.randint(10))

# print(tf.global_variables()[1])

tf.add_to_collection('init',[a,b])

init=tf.variables_initializer(tf.get_collection('init')[0])
# init=tf.variables_initializer([a,b])

with tf.Session() as sess:
    sess.run(init)
    # sess.run(c)
    for i in tf.global_variables():
        try:
            sess.run(i)
        except Exception as e:
            print(e)
            sess.run(tf.variables_initializer([i]))
            # sess.run(tf.global_variables()[2])
            # traceback.print_exc()
            # print("erro")
    sess.run(c)
    # print(tf.get_collection('init'))

    

