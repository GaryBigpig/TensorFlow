import tensorflow as tf
import numpy as np

a=np.array([[1,2,3],[4,5,6],[7,8,9]])
b=tf.constant(a)
print ('a是数组\n',a)
print ('b是tensor\n',b)
tensor_a = tf.convert_to_tensor(a)
print('a转换为了tensor\n', tensor_a)
with tf.Session() as sess:
    # b.eval()就得到tensor的数组形式
    print('打印b的元素')
    for x in b.eval():
        print (x)

