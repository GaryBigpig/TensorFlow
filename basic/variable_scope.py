import tensorflow as tf

# 未为变量定义作用域
a=tf.Variable(tf.ones([5,10]),name="W")
b=tf.Variable(tf.ones(10),name="b")
a1=tf.Variable(tf.ones([10,5]),name="W")
print(a)
print(b)
print(a1)

# 为了使得计算图更加清晰，使用了variable_scope()为变量定义作用域
# a和a1的属于不同的变量，可以看到虽然操作中name的名字是相同的，但是在计算的过程中依然是当成不同的变量
with tf.variable_scope("layer1"):
    a=tf.Variable(tf.ones([5,10]),name="W")
    b=tf.Variable(tf.ones(10),name="b")
    a1=tf.Variable(tf.ones([5,10]),name="W")
print(a)
print(b)
print(a1)

# 同一scope的同一变量可以通过get_variable()函数,a和a1其实是指向同一个变量W，类似作用域里的全局变量
with tf.variable_scope("layer1") as scope:
    # 获取W变量，如果没有就创建
    a = tf.get_variable('W', [5, 10])
    # b=tf.Variable(tf.ones(10),name="b")
    b = tf.get_variable('b', [10])
    # 如果再次使用W变量，需要这句话，不然会报错
    scope.reuse_variables()
    a1=tf.get_variable('W',shape=[5,10])
print(a)
print(b)
print(a1)

#可以在不同的定义域里定义相同名称的变量
with tf.variable_scope("layer1",reuse=True):
    a2 = tf.get_variable('W', [5, 10])
    # b2=tf.Variable(tf.ones(10),name="b")
    b2=tf.get_variable('b',[10])
print(a2)
print(b2)


with tf.variable_scope("layer2"):
    a = tf.get_variable('W', [5, 10])
    b=tf.Variable(tf.ones(10),name="b")
print(a)
print(b)