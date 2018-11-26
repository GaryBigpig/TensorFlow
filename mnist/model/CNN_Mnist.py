import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import os

#load mnist data
project_dir=os.path.abspath(os.path.join(os.getcwd(),'..'))
mnist_data_dir=os.path.join(project_dir,'mnist_data/')
mnist_data = input_data.read_data_sets(mnist_data_dir, one_hot=True)

x=tf.placeholder('float',shape=[None,784])
y_=tf.placeholder('float',shape=[None,10])

def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="SAME")

def max_pool_22(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

sess=tf.Session()

x_image=tf.reshape(x,[-1,28,28,1])
# print(mnist_data.train.images[0])
sess.run(x_image,feed_dict={x:mnist_data.train.images})
print(tf.convert_to_tensor(mnist_data.train.images).get_shape())
print(x_image.get_shape())

#1st CNN layer
W_conv1=weight_variable([5,5,1,32])
b_conv1=weight_variable([32])
h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
h_pool1=max_pool_22(h_conv1)
print(h_pool1.get_shape())

#2cd CNN layer
W_conv2=weight_variable([5,5,32,64])
b_conv2=weight_variable([64])
h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2=max_pool_22(h_conv2)
print(h_pool2.get_shape())

#3rd FC layer
W_fc1=weight_variable([7*7*64,1024])
b_fc1=weight_variable([1024])

h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
print(h_pool2_flat.get_shape())

h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
print(h_fc1.get_shape())

# dropt out
keep_prob=tf.placeholder('float')
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

W_fc2=weight_variable([1024,10])
b_fc2=bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)

cross_entropy=-tf.reduce_sum(y_*tf.log(y_conv))
train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction=tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,'float'))

sess.run(tf.global_variables_initializer())
#train
with sess.as_default():
    for i in range(20000):
        batch=mnist_data.train.next_batch(50)
        if i%100==0:
            train_accuracy=accuracy.eval(feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0})
            print('Step %d, training accuracy %g'%(i,train_accuracy))
        train_step.run(feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})

#test
    print('Test accuracy %g'%accuracy.eval(feed_dict={x:mnist_data.test.images,y_:mnist_data.test.labels,keep_prob:1.0}))