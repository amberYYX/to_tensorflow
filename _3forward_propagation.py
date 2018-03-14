# -- coding: utf-8 --
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

w1 = tf.Variable(tf.random_normal([2,3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3,1], stddev=1, seed=1))
#seed = 1,两次产生的随机值结果一样
test = tf.Print(w1, [w1, "w"])

# x = tf.constant([[0.7, 0.9]])
x = tf.placeholder(tf.float32,[3,2],name = "input")

a = tf.matmul(x,w1)
b = tf.matmul(a,w2)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
   
    # 输出w1，w2 两个tensor的值
    print(sess.run(w1))
    print(sess.run(w2 ))
    print('\n')
    print(sess.run(b,feed_dict={x:[[0.7,0.9],[0.1,0.4],[0.5,0.8]]}))
