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
plt.ion()
plt.show()

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
dft = pd.read_csv('24.CSV',encoding='gbk')
xt=dft[['Width']]
xt_data=np.array(xt,dtype='float32')

yt=dft[['CO2v7 density','CO2v8 density']]
yti_data=np.array(yt,dtype='float32')



#0-2mm
at=dft[dft.Width>=0]
aat=at[at.Width<=1.0]
xat=aat[['Width','Rise rate']]
xat_data=np.array(xat,dtype='float32')

xha=aat[['Width']]
xha_data=np.array(xha,dtype='float32')
yat=aat[['CO2v7 density','CO2v8 density']]
yat_data=np.array(yat,dtype='float32')


# 定义输入
xs1 = tf.placeholder(tf.float32, [None,2])
ys1 = tf.placeholder(tf.float32, [None,2])


# 定义隐藏层

l1 = add_layer(xs1, 2, 30, activation_function=tf1.nn.relu)
l2 = add_layer(l1, 30, 30, activation_function=tf1.tanh)
l3 = add_layer(l2, 30, 30, activation_function=tf1.tanh)
l4 = add_layer(l3, 30, 30, activation_function=tf1.sigmoid)


# 定义输出层
prediction1 = add_layer(l4, 30, 2, activation_function=None)
# prediction2 = add_layer(l8, 30, 1, activation_function=None)
# prediction12 = add_layer(l42, 30, 1, activation_function=None)
# prediction22 = add_layer(l82, 30, 1, activation_function=None)

# 损失函数，误差平方求均值
# loss1 = tf.reduce_mean(tf.square(ys1 - prediction1))
# loss2 = tf.reduce_mean(tf.square(ys2 - prediction2))

lossh1 = tf.reduce_mean(tf.square(ys1 - prediction1))
# lossh2 = tf.reduce_mean(tf.square(ys2 - prediction2))
# lossh12 = tf.reduce_mean(tf.square(ys12 - prediction12))
# lossh22 = tf.reduce_mean(tf.square(ys22 - prediction22))

# lossh1 = 100*tf.reduce_mean(tf.abs((ys1 - prediction1)/ys1))
# lossh2 = 100*tf.reduce_mean(tf.abs((ys2 - prediction2)/ys2))
# lossh12 = 100*tf.reduce_mean(tf.abs((ys12 - prediction12)/ys12))
# lossh22 = 100*tf.reduce_mean(tf.abs((ys22 - prediction22)/ys22))

# 初始化变量
init = tf.global_variables_initializer()
# 开启对话

# 训练
with tf.Session() as sess:
 sess.run(init)

 saver = tf.train.Saver(max_to_keep=1)


# 建一个画布，facecolor是背景色
 plt.figure(facecolor='w')
 plt.plot(xt_data, yti_data, 'r-', linewidth=1, label='real')
#plt.plot(xtime_data, yti_data, 'r-', linewidth=1, label='timereal')

 model_file1=tf1.train.latest_checkpoint('save1/')
 saver.restore(sess,model_file1)
 plt.plot(xha_data,sess.run(prediction1, feed_dict={xs1: xat_data}), 'g-', linewidth=1, label='5200V-predict')#模型1
 print(sess.run(lossh1, feed_dict={xs1: xat_data, ys1: yat_data}))
 with open("I24.CSV","w",newline='') as f:
         b_csv = csv.writer(f)
         #b_csv.writerow("ID")
         b_csv.writerows(sess.run(prediction1, feed_dict={xs1: xat_data}))

         # model_file12 = tf1.train.latest_checkpoint('save12/')
         # saver.restore(sess, model_file12)
         # plt.plot(xhb_data, sess.run(prediction12, feed_dict={xs12: xbt_data}), 'g-', linewidth=1)  # 模型12
         # print(sess.run(lossh12, feed_dict={xs12: xbt_data, ys12: ybt_data}))
         # b_csv.writerows(sess.run(prediction12, feed_dict={xs12: xbt_data}))
         #
         # model_file2 = tf1.train.latest_checkpoint('save2/')
         # saver.restore(sess, model_file2)
         # b_csv.writerows(sess.run(prediction2, feed_dict={xs2: xct_data}))
         # # b_csv.writerows(yhd_data - yhd_data)
         # plt.plot(xhc_data, sess.run(prediction2, feed_dict={xs2: xct_data}), 'g-', linewidth=1)#模型2
         # print(sess.run(lossh2, feed_dict={xs2: xct_data, ys2: yct_data}))
         #
         # model_file22 = tf1.train.latest_checkpoint('save22/')
         # saver.restore(sess, model_file22)
         # b_csv.writerows(sess.run(prediction22, feed_dict={xs22: xdt_data}))
         # # b_csv.writerows(yhd_data - yhd_data)
         # plt.plot(xhd_data, sess.run(prediction22, feed_dict={xs22: xdt_data}), 'g-', linewidth=1)  # 模型22
         # print(sess.run(lossh22, feed_dict={xs22: xdt_data, ys22: ydt_data}))

# 显示图例，设置图例的位置
         plt.legend(loc = 'upper right')
         #plt.legend(loc='upper right')
         plt.title("I", fontsize=20)

# 加网格
         plt.grid(b=True)
         plt.pause(60)
         plt.show()

         time_used = time.time() - start
         print('Total running time(second):')
         print(time_used)