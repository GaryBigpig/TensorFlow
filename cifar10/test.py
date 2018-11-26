import tensorflow as tf

if __name__ == "__main__":
    # print(20//3)
    # print(r'''hello,\n
    # world''')
    # s3 = r'Hello, "Bart"'
    # s4 = r'''Hello,
    # Lisa!'''
    # print(s3)
    # print(s4)
    # print('%2d-%02d' % (3, 1))
    # print('%.2f' % 3.1415926)

    # import os
    #
    # print('Process (%s) start...' % os.getpid())
    # # Only works on Unix/Linux/Mac:
    # pid = os.fork()
    # if pid == 0:
    #     print('I am child process (%s) and my parent is %s.' % (os.getpid(), os.getppid()))
    # else:
    #     print('I (%s) just created a child process (%s).' % (os.getpid(), pid))

    import time, threading


    # 新线程执行的代码:
    def loop():
        print('thread %s is running...' % threading.current_thread().name)
        n = 0
        while n < 5:
            n = n + 1
            print('thread %s >>> %s' % (threading.current_thread().name, n))
            time.sleep(1)
        print('thread %s ended.' % threading.current_thread().name)


    print('thread %s is running...' % threading.current_thread().name)
    t = threading.Thread(target=loop, name='LoopThread')
    t.start()
    t.join()
    print('thread %s ended.' % threading.current_thread().name)


    # s=set([1,2,3])
    # print(s)
    # s.add((1,2))
    # print(s)
    # s.add((1,[1,2]))
    # print(s)
    # # 定义一个32位浮点数的变量，初始值位0.0
    # v1 = tf.Variable(dtype=tf.float32, initial_value=0.)
    #
    # # 衰减率decay，初始值位0.99
    # decay = 0.99
    #
    # # 定义num_updates，同样，初始值位0
    # num_updates = tf.Variable(0, trainable=False)
    #
    # # 定义滑动平均模型的类，将衰减率decay和num_updates传入。
    # ema = tf.train.ExponentialMovingAverage(decay=decay, num_updates=num_updates)
    #
    # # 定义更新变量列表
    # update_var_list = [v1]
    #
    # # 使用滑动平均模型
    # ema_apply = ema.apply(update_var_list)
    #
    # # Tensorflow会话
    # with tf.Session() as sess:
    #     # 初始化全局变量
    #     sess.run(tf.global_variables_initializer())
    #
    #     # 输出初始值
    #     print(sess.run([v1, ema.average(v1)]))
    #     # [0.0, 0.0]（此时 num_updates = 0 ⇒ decay = .1, ），
    #     # shadow_variable = variable = 0.
    #
    #     # 将v1赋值为5
    #     sess.run(tf.assign(v1, 5))
    #
    #     # 调用函数，使用滑动平均模型
    #     sess.run(ema_apply)
    #
    #     # 再次输出
    #     print(sess.run([v1, ema.average(v1)]))
    #     # 此时，num_updates = 0 ⇒ decay =0.1,  v1 = 5;
    #     # shadow_variable = 0.1 * 0 + 0.9 * 5 = 4.5 ⇒ variable
    #
    #     # 将num_updates赋值为10000
    #     sess.run(tf.assign(num_updates, 10000))
    #
    #     # 将v1赋值为10
    #     sess.run(tf.assign(v1, 10))
    #
    #     # 调用函数，使用滑动平均模型
    #     sess.run(ema_apply)
    #
    #     # 输出
    #     print(sess.run([v1, ema.average(v1)]))
    #     # decay = 0.99,shadow_variable = 0.99 * 4.5 + .01*10 ⇒ 4.555
    #
    #     # 再次使用滑动平均模型
    #     sess.run(ema_apply)
    #
    #     # 输出
    #     print(sess.run([v1, ema.average(v1)]))
    #     # decay = 0.99，shadow_variable = .99*4.555 + .01*10 = 4.609
    #     for i in range(1000):
    #         sess.run(ema_apply)
    #         print(sess.run([v1, ema.average(v1)]))
    #
    #     a=[[[1, 2, 3]] * 2] * 3
    #     print(a)
    #     print([9]+[0]*3+[128]*3+[256]*3)

        # with tf.Graph().as_default():
        #     # 从默认计算图中返回glob step tensor
        #     global_step = tf.train.get_or_create_global_step()
        #     sess = tf.Session()
        #     with sess.as_default():
        #         sess.run(tf.global_variables_initializer())
        #         print(sess.run(global_step))
        # print(time.time())
        # print(datetime.now())