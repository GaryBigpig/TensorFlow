from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

import cifar10
# 定义参数变量
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '/cifar10_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/cifar10_train_check',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 10000,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', False,
"""Whether to run eval only once.""")

def eval_once(saver,summary_writer,top_k_op,summary_op):
    # 创建上下文管理器，定义会话sess
    with tf.Session() as sess:
        # 获取保存的模型，模型路径由FLAGS.checkpoint_dir定义
        ckpt=tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        # 如果ckpt和ckpt.model_checkpoint_path非空
        if ckpt and ckpt.model_checkpoint_path:
            # 从chekpoint中恢复参数
            saver.restore(sess, ckpt.model_checkpoint_path)
            # 获取训练的总步数，模型文件的名称为'model.ckpt-总步数'
            global_step=ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
        # 如果ckpt和ckpt.model_checkpoint_path为空，打印提示，退出函数
            print('No checkpoint file found')
            return
        # 开启一个多线程协调器，协调线程间的关系
        coord=tf.train.Coordinator()
        try:
            # 定义数组存储线程
            threads=[]
            # 所有队列管理器被默认加入图的tf.GraphKeys.QUEUE_RUNNERS集合中
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                # 队列创建线程来做入列操作，并将创建的线程存入threads数组
                threads.extend(qr.create_threads(sess=sess,coord=coord,daemon=True,start=True))
            # 总样本数除以每个批次的样本数量向上取整，获取总的批次数量也即是迭代次数
            num_iter=int(math.ceil(FLAGS.num_examples/FLAGS.batch_size))
            # 预测结果的真值数量初始化为0
            true_count=0
            # 迭代次数乘以每个批次的样本数量得到总的样本数量
            total_sample_count=num_iter*FLAGS.batch_size
            # 循环的步数初始化为0
            step=0
            while step<num_iter and not coord.should_stop():
                # 调用top_k_op=tf.nn.in_top_k执行预测，返回bool值
                predictions=sess.run([top_k_op])
                # 对预测结果中的真值进行累加
                true_count+=np.sum(predictions)
                # 循环的步数自动加1
                step+=1
            # 预测真值的数量除以总的样本数量得到预测的精度
            precision=true_count/total_sample_count
            # 打印系统当前的时间和预测的精度
            print('%s: precision @ 1 = %.3f' % (datetime.now(),predictions))
            # 定义tf.Summary()的对象，将训练过程中的信息，包括预测信息写入到FLAGS.eval_dir下的图文件，支持后续tensorboard打印
            summary=tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='Precision @ 1', simple_value=predictions)
            summary_writer.add_summary(summary,global_step)
        except Exception as e:
            # 如果程序运行出错，多线程协调器coord停止其他线程，并把错误报告给coord
            coord.request_stop(e)
        # 多线程协调器coord停止其他线程
        coord.request_stop()
        # 多线程协调器coord等待所有的线程终止，并给出10秒的宽限期
        # 如果超过10秒还有线程存在且ignore_live_threads参数为False，会报RuntimeError
        # 如果request_stop()被传入Exception信息，则会替代RuntimeError，报Exception信息
        coord.join(threads,stop_grace_period_secs=10)

def evaluate():
    # 使用系统默认图作为计算图
    with tf.Graph().as_default() as g:
        # 设置变量eval_data为'test'
        eval_data=FLAGS.eval_data=='test'
        # 调用cifar10.inputs函数对测试数据进行处理，得到评估使用的图像和标签数据
        images,labels=cifar10.inputs(eval_data=eval_data)
        # 使用cifar10.inference函数构建的深度学习模型对测试数据进行评估
        # 返回预测的分类数据logits，logits为预测为各个类别的概率
        logits=cifar10.inference(images)
        # tf.nn.in_top_k按照logtis中的元素从大到小的顺序对元素坐标排序，再按照labels中的元素从大到小的顺序对坐标排序，
        # 然后取第一位进行比较，如果相同返回true，否则返回false
        # logits中的元素为预测的各个类别的概率，最大概率也就是预测的类别，labels中正确的类别对应的元素为1，否则为0
        # 所以logits最大元素的坐标如果和labels中最大元素的坐标一致，说明预测正确
        top_k_op=tf.nn.in_top_k(logits,labels,1)
        # 设置移动平均模型，设置参数decay=cifar10.MOVING_AVERAGE_DECA
        # 因为训练的时候使用了移动平均模型，评估的时候也需要使用
        variable_averages=tf.train.ExponentialMovingAverage(cifar10.MOVING_AVERAGE_DECAY)
        # 使用variables_to_restore函数在加载模型的时候将影子变量直接映射到变量的本身，
        # 所以在获取变量的滑动平均值的时候只需要获取到变量的本身值而不需要去获取影子变量
        # 参考文章https://blog.csdn.net/sinat_29957455/article/details/78508793
        variable_to_restore=variable_averages.variables_to_restore()
        # 定义saver向checkpoint保存和读取变量
        saver=tf.train.Saver(variable_to_restore)
        # 定义操作对象，合并操作点，将所有训练时的summary信息保存到磁盘，以便tensorboard显示
        summary_op=tf.summary.merge_all()
        # 定义操作对象，将计算图的信息保存到文件存储在FLAGS.eval_dir定义的路径，以备tensorboard打印时使用
        summary_writer=tf.summary.FileWriter(FLAGS.eval_dir,g)

        while True:
            # 循环调用函数单次评估函数eval_once做数据评估
            eval_once(saver=saver,summary_writer=summary_writer,top_k_op=top_k_op,summary_op=summary_op)
            # 如果FLAGS.run_once设置为True，也即是只运行一次评估函数
            if FLAGS.run_once:
                # 退出循环
                break
            # eval_once每FLAGS.eval_interval_secs==5分钟运行一次
            time.sleep(FLAGS.eval_interval_secs)

def main(argv=None):
    # 下载并解压cifar10的数据，防止没有数据可评估
    cifar10.maybe_download_and_extract()
    # 创建目录存储Log文件，存储评估的结果，供tensorboard打印使用
    # 如果目录存在，说明有历史数据，删除重新创建，存储最新的评估结果
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    # 评估训练结果
    evaluate()


if __name__ == '__main__':
    # 处理FLAGS参数解析，运行main()函数
    tf.app.run()





