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
#0-2mm
a=df[df.Width>=0]
aa=a[a.Width<=1.0]
xa=aa[['Width','Rise rate']]
# print(xa)

xa_data=np.array(xa,dtype='float32')
ya=aa[['CO3- density']]
# ya=aa[['Electron density','Negative density','Positive density','O- density','O2- density','CO3- density','CO4- density','CO2+ density','O2+ density','O density','O2 density','O3 density','CO density','C density','CO2e1 density','CO2va density','CO2vb density','CO2vc density','CO2v1a density','CO2v1 density','CO2v2 density','CO2v3 density','CO2v4 density','CO2v5 density','CO2v6 density','CO2v7 density','CO2v8 density','Electric field']]
ya_data=np.array(ya,dtype='float32')
#print(xa_data)

# 定义输入
xs1 = tf.placeholder(tf.float32, [None,2])
ys1 = tf.placeholder(tf.float32, [None,1])

# 定义隐藏层
l1 = add_layer(xs1, 2, 30, activation_function=tf1.nn.relu)
l2 = add_layer(l1, 30, 30, activation_function=tf1.tanh)
l3 = add_layer(l2, 30, 30, activation_function=tf1.tanh)
l4 = add_layer(l3, 30, 30, activation_function=tf1.sigmoid)

# 定义输出层
prediction1 = add_layer(l4, 30, 1, activation_function=None)

# 损失函数，误差平方求均值
loss1 = tf.reduce_mean(tf.square(ys1 - prediction1))


# 优化器最小化损失函数
train_step1 = tf.train.AdamOptimizer(0.000001).minimize(loss1)


# 初始化变量
init = tf.global_variables_initializer()
# 开启对话
# 训练
with tf.Session() as sess:
    sess.run(init)
    saver = tf.train.Saver(max_to_keep=1)

    model_file1 = tf1.train.latest_checkpoint('save1/')
    saver.restore(sess, model_file1)
    #注意！下边两个运行周期数要同时改动！！！
    for i in range(3000001):
        sess.run(train_step1, feed_dict={xs1: xa_data, ys1: ya_data})
        if i % 100 == 0:
            print(sess.run(loss1, feed_dict={xs1: xa_data, ys1: ya_data}), (i/3000001) * 100, '%')
    saver.save(sess, 'save1/model1', global_step=i + 1)

    time_used = (time.time() - start)/3600
    print('Total running time(hour):')
    print(time_used)

