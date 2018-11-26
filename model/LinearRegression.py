import tensorflow as tf
import numpy as np

# 构造训练数据，x_data随机生成，y_data由x_data的线性函数生成
x_data=np.float32(np.random.rand(2,100))
y_data=np.dot([0.1,0.2],x_data)+0.3
# print(y_data)

# 构造线性模型，初始化参数W和b
b=tf.Variable(tf.zeros([1]))
W=tf.Variable(tf.random_uniform([1,2],-1,1))
y=tf.matmul(W,x_data)+b

# 使用均方差构造损失函数
loss=tf.reduce_mean(tf.square(y-y_data))
optimizer=tf.train.GradientDescentOptimizer(0.5)
train=optimizer.minimize(loss)

# 初始化变量
init=tf.initialize_all_variables()

# 启动图
sess=tf.Session()
sess.run(init)

# 训练
for i in range(800):
    sess.run(train)
    if i % 100 == 0:
        print(i,sess.run(W),sess.run(b),sess.run(loss))


