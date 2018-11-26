import tensorflow as tf

# 构造字符串张量，包含要读取的文件名
input_files=['inputs/input1.txt','inputs/input2.txt']
# 将input_files中包含的文件打包生成一个先入先出队列（FIFOQueue）
# 并且在计算图的QUEUE_RUNNER集合中添加一个QueueRunner（QueueRunner包含一个队列的一系列的入列操作）
# shuffle=True时，会对文件名进行乱序处理
input_queues=tf.train.string_input_producer(string_tensor=input_files,shuffle=True)

# 生成文件阅读器Reader，设置每次从文件中读取固定长度（4）的字符串记录Record
reader=tf.FixedLengthRecordReader(record_bytes=4)
# 文件阅读器读取队列的值返回key,value
key,value=reader.read(input_queues)

output_key=key
output_val=value

sess=tf.Session()
# 在运算图中运行队列操作
tf.train.start_queue_runners(sess=sess)

# print('key:',sess.run(output_key),'val:',sess.run(output_val))
# print('key:',sess.run(output_key),'val:',sess.run(output_val))
# print('key:',sess.run(output_key),'val:',sess.run(output_val))
# print('key:',sess.run(output_key),'val:',sess.run(output_val))
# print('key:',sess.run(output_key),'val:',sess.run(output_val))
# print('key:',sess.run(output_key),'val:',sess.run(output_val))
# print('key:',sess.run(output_key),'val:',sess.run(output_val))
# print('key:',sess.run(output_key),'val:',sess.run(output_val))
# print('key:',sess.run(output_key),'val:',sess.run(output_val))
# print('\n')
print('key:',sess.run(output_key))
print('key:',sess.run(output_key))
print('key:',sess.run(output_key))
print('key:',sess.run(output_key))
print('key:',sess.run(output_key))
print('key:',sess.run(output_key))
print('val:',sess.run(output_val))
print('val:',sess.run(output_val))
print('val:',sess.run(output_val))
print('val:',sess.run(output_val))
print('val:',sess.run(output_val))
print('val:',sess.run(output_val))

sess.close()


