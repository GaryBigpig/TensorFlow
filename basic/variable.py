import tensorflow as tf
import numpy as np

# 描绘到计算图
b1=tf.Variable(initial_value=tf.ones([3,3]),dtype=tf.float32)
b2=tf.Variable(initial_value=tf.ones([3,3])*2,dtype=tf.float32)
w1=tf.Variable(initial_value=tf.ones([2,2]),dtype=tf.float32,validate_shape=False)
w2=tf.Variable(initial_value=tf.ones([2,2])*3,dtype=tf.float32)
# print(b1)
# print(b2)
# 创建会话
sess=tf.Session()
# 变量初始化
init=tf.global_variables_initializer()
# 将计算发送到计算内核去执行
sess.run(init)
print('b1:\n',sess.run(b1))
print('b2:\n',sess.run(b2))
print('w1:\n',sess.run(w1))
print('w2:\n',sess.run(w2))

#变量计算
print('b1 dot b2:\n',sess.run(b1*b2))
print('b1*b2:\n',sess.run(tf.matmul(b1,b2)))

#变量更新
# update_b1=tf.assign(b1,[[1,2,3],[1,2,3],[1,2,3]])
# print('Update b1:\n',sess.run(update_b1))
# update_w1=tf.assign(w1,[[1,2,3],[1,2,3]],validate_shape=False)
# print('Update w1:\n',sess.run(update_w1))
#保存全部变量
saver1=tf.train.Saver()
saver2=tf.train.Saver()
# save_path1=saver1.save(sess,'variable/1.ckpt')
#保存部分变量
saver2=tf.train.Saver({'a1':w1,'a2':w2})
# save_path2=saver2.save(sess,'variable/2.ckpt')

#读取变量
# save_path1=saver1.restore(sess,'variable/1.ckpt')
save_path2=saver2.restore(sess,'variable/2.ckpt')
print('b1:\n',sess.run(b1))
print('b2:\n',sess.run(b2))
print('w1:\n',sess.run(w1))
print('w2:\n',sess.run(w2))