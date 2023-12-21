import time
start = time.time()
import tensorflow.compat.v1 as tf
import tensorflow as tf1
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt
import csv


def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    # 神经网络预测值
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    # 激励函数
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

#训练集

df = pd.read_csv('training data.CSV',encoding='gbk')


#0-0.25μs
a=df[df.Time>=0]
aa=a[a.Time<0.35]
xa=aa[['Time','Voltage amplitude']]
# print(xa)

xa_data=np.array(xa,dtype='float32')
ya=aa[['Current density']]
ya_data=np.array(ya,dtype='float32')
#print(xa_data)

#0.25-1.0μs
b=df[df.Time>=0.35]
bb=b[b.Time<0.6]
xb=bb[['Time','Voltage amplitude']]
# print(xa)

xb_data=np.array(xb,dtype='float32')
yb=bb[['Current density']]
yb_data=np.array(yb,dtype='float32')

#1-1.25μs
c=df[df.Time>=0.6]
cc=c[c.Time<0.95]
xc=cc[['Time','Voltage amplitude']]
# print(xa)

xc_data=np.array(xc,dtype='float32')
yc=cc[['Current density']]
yc_data=np.array(yc,dtype='float32')

#1.25-2.0μs
d=df[df.Time>=0.95]
dd=d[d.Time<=1.0]
xd=dd[['Time','Voltage amplitude']]
# print(xa)

xd_data=np.array(xd,dtype='float32')
yd=dd[['Current density']]
yd_data=np.array(yd,dtype='float32')

# 定义输入
xs1 = tf.placeholder(tf.float32, [None,2])
ys1 = tf.placeholder(tf.float32, [None,1])
xs12 = tf.placeholder(tf.float32, [None,2])
ys12 = tf.placeholder(tf.float32, [None,1])
xs2 = tf.placeholder(tf.float32, [None,2])
ys2 = tf.placeholder(tf.float32, [None,1])
xs22 = tf.placeholder(tf.float32, [None,2])
ys22 = tf.placeholder(tf.float32, [None,1])

# 定义隐藏层

l1 = add_layer(xs1, 2, 30, activation_function=tf1.nn.relu)
l2 = add_layer(l1, 30, 30, activation_function=tf1.tanh)
l3 = add_layer(l2, 30, 30, activation_function=tf1.tanh)
l4 = add_layer(l3, 30, 30, activation_function=tf1.sigmoid)

l5 = add_layer(xs2, 2, 30, activation_function=tf1.nn.relu)
l6 = add_layer(l5, 30, 30, activation_function=tf1.tanh)
l7 = add_layer(l6, 30, 30, activation_function=tf1.tanh)
l8 = add_layer(l7, 30, 30, activation_function=tf1.sigmoid)

l12 = add_layer(xs12, 2, 30, activation_function=tf1.nn.relu)
l22 = add_layer(l12, 30, 30, activation_function=tf1.tanh)
l32 = add_layer(l22, 30, 30, activation_function=tf1.tanh)
l42 = add_layer(l32, 30, 30, activation_function=tf1.sigmoid)

l52 = add_layer(xs22, 2, 30, activation_function=tf1.nn.relu)
l62 = add_layer(l52, 30, 30, activation_function=tf1.tanh)
l72 = add_layer(l62, 30, 30, activation_function=tf1.tanh)
l82 = add_layer(l72, 30, 30, activation_function=tf1.sigmoid)


# 定义输出层
prediction1 = add_layer(l4, 30, 1, activation_function=None)
prediction2 = add_layer(l8, 30, 1, activation_function=None)
prediction12 = add_layer(l42, 30, 1, activation_function=None)
prediction22 = add_layer(l82, 30, 1, activation_function=None)

# 损失函数，误差平方求均值
loss1 = tf.reduce_mean(tf.square(ys1 - prediction1))
loss2 = tf.reduce_mean(tf.square(ys2 - prediction2))
loss12 = tf.reduce_mean(tf.square(ys12 - prediction12))
loss22 = tf.reduce_mean(tf.square(ys22 - prediction22))

# 优化器最小化损失函数
train_step1 = tf.train.AdamOptimizer(0.0001).minimize(loss1)
train_step2 = tf.train.AdamOptimizer(0.0001).minimize(loss2)
train_step12 = tf.train.AdamOptimizer(0.00001).minimize(loss12)
train_step22 = tf.train.AdamOptimizer(0.00001).minimize(loss22)

# 初始化变量
init = tf.global_variables_initializer()
# 开启对话

# 训练
with tf.Session() as sess:
    sess.run(init)
    is_train = True
    is_mod = True
    is_mod2 = False
    saver = tf.train.Saver(max_to_keep=1)

    #训练
    #注意！下边两个运行周期数要同时改动！！！
    if is_train:
      if is_mod:
       if is_mod2:
        model_file1 = tf1.train.latest_checkpoint('save1/')
        saver.restore(sess, model_file1)
        for i in range(2000001):
         sess.run(train_step1, feed_dict={xs1: xa_data, ys1: ya_data})
         if i % 100 == 0:
          print(sess.run(loss1, feed_dict={xs1: xa_data, ys1: ya_data}), (i/2000001)*100, '%')

        saver.save(sess, 'save1/model1', global_step=i + 1)
       else:
           model_file12 = tf1.train.latest_checkpoint('save12/')
           saver.restore(sess, model_file12)
           for i in range(700001):
               sess.run(train_step12, feed_dict={xs12: xb_data, ys12: yb_data})
               if i % 100 == 0:
                   print(sess.run(loss12, feed_dict={xs12: xb_data, ys12: yb_data}), (i/700001)*100, '%')

           saver.save(sess, 'save12/model12', global_step=i + 1)
      else:
       if is_mod2:
        model_file2 = tf1.train.latest_checkpoint('save2/')
        saver.restore(sess, model_file2)
        for i in range(2000001):
         sess.run(train_step2, feed_dict={xs2: xc_data, ys2: yc_data})
         if i % 100 == 0:
          print(sess.run(loss2, feed_dict={xs2: xc_data, ys2: yc_data}), (i/2000001)*100, '%')

        saver.save(sess, 'save2/model2', global_step=i + 1)
       else:
           model_file22 = tf1.train.latest_checkpoint('save22/')
           saver.restore(sess, model_file22)
           for i in range(700001):
               sess.run(train_step22, feed_dict={xs22: xd_data, ys22: yd_data})
               if i % 100 == 0:
                   print(sess.run(loss22, feed_dict={xs22: xd_data, ys22: yd_data}), (i/700001)*100, '%')

           saver.save(sess, 'save22/model22', global_step=i + 1)

    time_used = (time.time() - start)/3600
    print('Total running time(hour):')
    print(time_used)