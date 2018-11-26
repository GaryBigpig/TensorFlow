import tensorflow as tf
import numpy as np

# tf.zeros(shape, dtype=None, name=None),返回一个类型为dtype，并且维度为shape的tensor，并且所有的元素均为０
# shape: 用于表示维度，通常为一个int32类型数组，或者一个一维(1-D)的tf.int32数字．注意不能直接使用数字，
# dtype: 所要创建的tensor对象的数据类型，name: 一个该操作的别名(可选的).
# 描绘到计算图
a1=tf.zeros([5],dtype=tf.int32,name='a1')
a2=tf.zeros([2,3],dtype=tf.float32)
print(a1)
print(a2)
#创建会话去执行
sess=tf.Session()
# 将计算发送到计算内核去执行
print(sess.run(a1))
print(sess.run(a2))
sess.close()

# tf.zeros_like(tensor, dtype=None, name=None),
# 返回一个类似当前给定tensor对象类型以及维度的对象，所有元素的值均为０
# tensor: tensor对象，dtype: 返回的tensor对象类型，不设置(为空时)时返回类型同参数tensor一致，name: 该操作别名 (可选).
list = [[1,2,3],[4,5,6]]
a1=tf.convert_to_tensor(list)
a2=tf.zeros_like(list,dtype=tf.float32,name='a2')
a3=tf.zeros_like(a1,dtype=tf.float32,name='a3')
print(a1)
print(a2)
print(a3)
#创建会话去执行
sess=tf.Session()
# 将计算发送到计算内核去执行
print(sess.run(a1))
print(sess.run(a2))
print(sess.run(a3))
sess.close()

# tf.ones(shape, dtype=None, name=None),返回一个类型为dtype，并且维度为shape的tensor，并且所有的元素均为1
# shape: 用于表示维度，通常为一个int32类型数组，或者一个一维(1-D)的tf.int32数字．注意不能直接使用数字，
# dtype: 所要创建的tensor对象的数据类型，name: 一个该操作的别名(可选的).
# 描绘到计算图
a1=tf.ones([5],dtype=tf.int32,name='a1')
a2=tf.ones([2,3],dtype=tf.float32)
print(a1)
print(a2)
#创建会话去执行
sess=tf.Session()
# 将计算发送到计算内核去执行
print(sess.run(a1))
print(sess.run(a2))
sess.close()

# tf.ones_like(tensor, dtype=None, name=None),
# 返回一个类似当前给定tensor对象类型以及维度的对象，所有元素的值均为1
# tensor: tensor对象，dtype: 返回的tensor对象类型，不设置(为空时)时返回类型同参数tensor一致，name: 该操作别名 (可选).
list = [[1,2,3],[4,5,6]]
a1=tf.convert_to_tensor(list)
a2=tf.ones_like(list,dtype=tf.float32,name='a2')
a3=tf.ones_like(a1,dtype=tf.float32,name='a3')
print(a1)
print(a2)
print(a3)
#创建会话去执行
sess=tf.Session()
# 将计算发送到计算内核去执行
print(sess.run(a1))
print(sess.run(a2))
print(sess.run(a3))
sess.close()

# tf.constant(value,dtype=None,shape=None,name=None)，
# 创建一个常量tensor，按照给出value来赋值，可以用shape来指定其形状。value可以是一个数，也可以是一个list。
# 如果是一个数，那么这个常亮中所有值的按该数来赋值。  如果是list,那么len(value)一定要小于等于shape展开后的长度。
# 赋值时，先将value中的值逐个存入。不够的部分，则全部存入value的最后一个值。
#描绘到计算图中
a1=tf.constant(np.ones([3,3]))
a2=tf.constant(np.ones([3,3]))
a3=tf.ones([4,4])
a = tf.constant(2,shape=[2])
b = tf.constant(2,shape=[2,2])
c = tf.constant([1,2,3],shape=[6])
d = tf.constant([1,2,3],shape=[3,2])
#矩阵相乘
a1_dot_a2=tf.matmul(a1,a2)
print(a1)
print(a2)
print(a3)
print(a1_dot_a2)
print(a)
print(b)
print(c)
print(d)
#下面要创建会话去执行
sess=tf.Session()
# 将计算发送到计算内核去执行
print(sess.run(a1))
print(sess.run(a2))
print(sess.run((a1_dot_a2)))
print(sess.run(a3))
print(sess.run(a))
print(sess.run(b))
print(sess.run(c))
print(sess.run(d))
sess.close()

# tf.fill(dims, value, name=None),创建一个维度为dims，值为value的tensor对
# dims: 类型为int32的tensor对象，用于表示输出的维度(1 - D, n - D)，通常为一个int32数组，如：[1], [2, 3]等
# value: 常量值(字符串，数字等)，该参数用于设置到最终返回的tensor对象值中
# name: 当前操作别名(可选)
# 返回:tensor对象，类型和value一致
# 当value为０时，该方法等同于tf.zeros();当value为１时，该方法等同于tf.ones()
a1=tf.fill([2,3],5,name='a1')
a2=tf.fill([2,3],'5',name='a2')
print(a1)
print(a2)
#下面要创建会话去执行
sess=tf.Session()
# 将计算发送到计算内核去执行
print(sess.run(a1))
print(sess.run(a2))