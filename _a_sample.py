# -- coding: utf-8 --
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from numpy.random import RandomState #通过numpy数据包生成数据集

#定义batch的大小
batch_size = 8

w1 = tf.Variable(tf.random_normal(shape=[2,3],stddev=1,seed=1))
w2 = tf.Variable(tf.random_normal(shape=[3,1],stddev=1,seed=1))

x = tf.placeholder(tf.float32,shape=(None,2),name="x-input")
y_= tf.placeholder(tf.float32,shape=(None,1),name="y-input")

#正向传播
a = tf.matmul(x,w1)
y = tf.matmul(a,w2)

# tf.matmul()  为矩阵乘法
# tf.multiply() 为矩阵点乘

cross_entropy = -tf.reduce_mean(y_*tf.log(tf.clip_by_value(y, 1e-10,1.0)))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

#通过随机数，生成数据集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size,2)
Y = [[int(x1+x2 < 1)] for (x1,x2) in X]

#创建绘画来运行
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    # 初始化变量
    sess.run(init)

    print(sess.run(w1))
    print(sess.run(w2))
    print("\n")

    #设定训练轮数
    STEPS = 50
    for i in range(STEPS):
        start = (i*batch_size)%dataset_size
        end = min(start+batch_size,dataset_size)

        #通过选取的训练神经网络并更新参数
        sess.run(train_step,feed_dict={x:X[start:end],y:Y[start:end]})

        #每隔一段时间计算交叉熵并输出
        total_cross = sess.run(cross_entropy,feed_dict={x:X,y:Y})
        print("after %d training steps,cross entropy is %g" %(i,total_cross))

    print(sess.run(w1))
    print(sess.run(w2))
    print("\n")

