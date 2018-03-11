# -- coding: utf-8 --
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

a = tf.constant([1],name='a')
b = tf.constant([2],name='b')
ans = a + b
#创建一个会话，让python的上下文管理器来管理这个会话
with tf.Session() as sess:
    sess.run(ans)
    #使用这个创建的会话来计算关心的结果
    #不需要再调用"Sesssion.close()来关闭会话


#下面的函数将生成的会话，自动设置为默认会话
tf.InteractiveSession()

conifg = tf.ConfigProto(allow_soft_placement = True,
                        log_device_placement = True)


