import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import os

project_dir=os.path.abspath(os.path.join(os.getcwd(),'..'))
print(project_dir)

mnist_data_dir=os.path.join(project_dir,'mnist_data/')
mnist_data = input_data.read_data_sets(mnist_data_dir, one_hot=True)
# print(mnist_data.train.images[0])
# print(mnist_data.train.labels[0])

x=tf.placeholder('float',[None,784])
y_=tf.placeholder('float',[None,10])

W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))

y=tf.nn.softmax(tf.matmul(x,W)+b)

cross_entropy=-tf.reduce_sum(y_*tf.log(y))

train_step=tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init=tf.initialize_all_variables()

sess=tf.Session()
sess.run(init)
#Train
for i in range(1000):
    print('epoch:{}'.format(i))
    batch_xs,batch_ys=mnist_data.train.next_batch(100)
    sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})
print(sess.run(W))
print(sess.run(b))
#Test
correct_predication=tf.equal(tf.arg_max(y,1),tf.arg_max(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_predication,'float'))
print(sess.run(accuracy,feed_dict={x:mnist_data.test.images,y_:mnist_data.test.labels}))