# -- coding: utf-8 --
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

w1 = tf.Variable(tf.random_normal([2,3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3,1], stddev=1, seed=1))
#seed = 1,两次产生的随机值结果一样

x = tf.constant([[0.7, 0.9]])

a = tf.matmul(x,w1)
b = tf.matmul(a,w2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(b))