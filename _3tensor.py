# -- coding: utf-8 --
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
# g = tf.Graph()
a = tf.constant([1.0,2.0],name="a")
b = tf.constant([2.0,3.0],name="b")
result = a + b
# with g.device('/gpu:0'):
##指定运行设备
#     result = a + b

"""
通过a.graph可以查看张量所属的计算图，因为没有特别指定，
所以这个计算图为默认计算图, 下面操作输出为True
"""
print(a.graph is tf.get_default_graph())

g1 = tf.Graph()
with g1.as_default():
    #在g1中定义变量v,并初始化为0
    v = tf.get_variable("v",initializer=tf.zeros(shape=[1]))


#在计算图g1中读取v变量
with tf.Session(graph=g1) as sess:
    sess.run(tf.global_variables_initializer())
    with tf.variable_scope("",reuse=True):
        print(sess.run(tf.get_variable("v")))

"""
about Tensor
tensor(name,shape,type)
"""
#tf.constant 是一个计算，计算结果为一个张量，保存在变量a中
a = tf.constant([1.0,2.0],name="a")
b = tf.constant([2.0,3.0],name="b")
result = tf.add(a,b,name='add')
print(result)
#Tensor("add_1:0", shape=(2,), dtype=float32)

"""
tensor的两大类用途
--对中间计算结果的引用
--计算图完成后，获得计算结果
"""

