import tensorflow as tf
import numpy as np

a=np.random.randint(1,10,[5,2])
b=tf.nn.embedding_lookup(a,[2,4])

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(a)
    print(sess.run(b))
