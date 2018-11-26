# https://tensorflow.google.cn/api_guides/python/threading_and_queues

import tensorflow as tf

input_data=[[3.,2.,1.],[11.,22.,33.],[111.,222.,333.]]
# input_data1=[[33.,22.,11.],[11.,22.,33.],[111.,222.,333.]]

# input_data = [
#     [[3., 2., 1.], [11., 22., 33.], [111., 222., 333.]],
#     [[23., 22., 21.], [211., 222., 233.], [2111., 2222., 2333.]],
# ]

# print(tf.shape(input_data))

# q=tf.FIFOQueue(3,dtypes=[tf.float32,tf.float32,tf.float32],shapes=[[],[],[]])
q=tf.FIFOQueue(3,dtypes=[tf.float32,tf.float32],shapes=[[],[]])
# q=tf.FIFOQueue(3,dtypes=[tf.float32],shapes=[[]])
# q = tf.FIFOQueue(3, dtypes=[tf.float32])
init = q.enqueue_many(input_data)
# init=q.enqueue(input_data)
# init1=q.enqueue(input_data1)

# output_data = q.dequeue()
output_data_many=q.dequeue_many(2)


with tf.Session() as sess:
    init.run()
    # init.run()
    # print("1：", sess.run(output_data))
    # print("2：", sess.run(output_data))
    # print("3：", sess.run(output_data))
    print("Many：", sess.run(output_data_many))
    sess.run(q.close(cancel_pending_enqueues=True))
    print(sess.run(q.is_closed()))

