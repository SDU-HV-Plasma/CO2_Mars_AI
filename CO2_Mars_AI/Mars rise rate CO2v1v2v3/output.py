import time
start = time.time()
import tensorflow.compat.v1 as tf
import tensorflow as tf1
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
tf.disable_v2_behavior()
import numpy as np
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

#测试集
#2005
dft = pd.read_csv('24.CSV',encoding='gbk')
at=dft[dft.Width>=0]
aat=at[at.Width<=1.0]
xat=aat[['Width','Rise rate']]
xat_data=np.array(xat,dtype='float32')
yat=aat[['CO2v1 density','CO2v2 density','CO2v3 density']]
yat_data=np.array(yat,dtype='float32')

# 定义输入
xs1 = tf.placeholder(tf.float32, [None,2])
ys1 = tf.placeholder(tf.float32, [None,3])
# 定义隐藏层
l1 = add_layer(xs1, 2, 30, activation_function=tf1.nn.relu)
l2 = add_layer(l1, 30, 30, activation_function=tf1.tanh)
l3 = add_layer(l2, 30, 30, activation_function=tf1.tanh)
l4 = add_layer(l3, 30, 30, activation_function=tf1.sigmoid)

# 定义输出层
prediction1 = add_layer(l4, 30, 3, activation_function=None)
# 损失函数，误差平方求均值
# lossh1 = tf.reduce_mean(tf.square(ys1 - prediction1))
lossh1 = tf.reduce_mean(tf.square(ys1[:, 0] - prediction1[:, 0]))
lossh2 = tf.reduce_mean(tf.square(ys1[:, 1] - prediction1[:, 1]))
lossh3 = tf.reduce_mean(tf.square(ys1[:, 2] - prediction1[:, 2]))
# 初始化变量
init = tf.global_variables_initializer()
# 开启对话

# 训练
with tf.Session() as sess:
 sess.run(init)
 saver = tf.train.Saver(max_to_keep=1)

 model_file1=tf1.train.latest_checkpoint('save1/')
 saver.restore(sess,model_file1)

 with open("I24.CSV", "w", newline='') as f:
     b_csv = csv.writer(f)
     b_csv.writerows(sess.run(prediction1, feed_dict={xs1: xat_data}))
     # print('Mean square error of Electron density:')
     # print(sess.run(lossh1, feed_dict={xs1: xat_data, ys1: yat_data}))
     print('Mean square error of O2 density:')
     print(sess.run(lossh1, feed_dict={xs1: xat_data, ys1: yat_data}))
     print('Mean square error of O density:')
     print(sess.run(lossh2, feed_dict={xs1: xat_data, ys1: yat_data}))
     print('Mean square error of O density:')
     print(sess.run(lossh3, feed_dict={xs1: xat_data, ys1: yat_data}))

     time_used = time.time() - start
     print('Total running time(second):')
     print(time_used)
