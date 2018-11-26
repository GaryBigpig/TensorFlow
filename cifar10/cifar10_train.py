from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time

import tensorflow as tf

import cifar10

# 设置输入参数
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    "train_dir",
    "/cifar10_train_check",
    "/cifar10_train_logs",
    """Directory where to write event logs and checkpoint.""",
)
tf.app.flags.DEFINE_integer("max_steps", 100000, """Number of batches to run.""")
tf.app.flags.DEFINE_boolean(
    "log_device_placement", False, """Whether to log device placement."""
)
tf.app.flags.DEFINE_integer(
    "log_frequency", 10, """How often to log results to the console."""
)

tf.app.flags.DEFINE_integer('batch_size', 128,
"""Number of samples per batch.""")



def train():
    # 创建上下文管理器，使用默认计算图
    with tf.Graph().as_default():
        # 从默认计算图中返回glob step tensor变量，初始化后为0
        global_step = tf.train.get_or_create_global_step()
        # 设置运行在cpu下
        with tf.device("/cup:0"):
            # 对cifar10的数据做切割、翻转、随机调整亮度、对比度和标准化处理，并按照FLAGS.batch_size参数生成批次数据
            images, labels = cifar10.distorted_inputs()
        # 通过cifar10.inference定义的深层学习框架（2层卷积，3层全链接）对cifar10.distorted_inputs()处理的图片数据进行学习
        # 得到各个分类的特征值
        logits = cifar10.inference(images)
        # 将各个分类的特征值和标签数据通过cifar10.loss得到损失值
        loss = cifar10.loss(logits, labels)
        # 通过cifar10.train对损失值进行训练，训练的总步数为global_step
        train_op = cifar10.train(loss, global_step)

        # 创建类_LoggerHook，是继承tf.train.SessionRunHook的子类，生成钩子程序，用来监视训练过程
        class _LoggerHook(tf.train.SessionRunHook):
            # 创建会话之前调用，调用begin()时，default graph会被创建，
            def begin(self):
                # 初始化训练的起始步数
                self._step = -1
                # 获取当前时间的时间戳（1970纪元后经过的浮点秒数）初始化会话的起始时间
                self._start_time = time.time()

            # 每个sess.run()执行之前调用，返回tf.train.SessRunArgs(op/tensor),在即将运行的会话中加入op/tensor loss
            # 加入的loss会和sess.run()中已定义的op/tensor合并，然后一起执行
            def before_run(self, run_context):
                # 叠加训练的步数，第一次训练从步数0开始
                self._step += 1
                # 返回SessionRunArgs对象，作为即将运行的会话的参数，将loss添加到会话中
                return tf.train.SessionRunArgs(loss)

            # 每个sess.run()执行之后调用，run_values是befor_run()中的op/tensor loss的返回值
            # 可以调用run_context.qeruest_stop()用于停止迭代，sess.run抛出任何异常after_run不会被调用
            def after_run(
                # tf.train.SessRunContext提供会话运行所需的信息，tf.train.SessRunValues保存会话运行的结果
                self, run_context, run_values  # pylint: disable=unused-argument
            ):
                # 判断迭代步数是否为FLAGS.log_frequency=10的整数倍
                if self._step % FLAGS.log_frequency == 0:
                    # 获取当前时间的时间戳（1970纪元后经过的浮点秒数）
                    current_time = time.time()
                    # 获取每10个会话运行的持续时间
                    duration = current_time - self._start_time
                    # 更新会话的起始时间
                    self._start_time = current_time
                    # 获取before_run中加入的操作loss的返回值
                    loss_value = run_values.results
                    # 计算每秒钟处理的样本数
                    examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
                    # 计算每个会话的运行时间，单位为秒
                    sec_per_batch = float(duration / FLAGS.log_frequency)
                    # 打印当前系统时间，当前步数下的loss的值（标示：每秒处理的样本数和每个批次样本处理所需要的时间）
        class _LoggerHook(tf.train.SessionRunHook):
            def begin(self):
                self._step = -1
                self._start_time = time.time()

            def before_run(self, run_context):
                self._step += 1
                return tf.train.SessionRunArgs(loss)

            def after_run(
                self, run_context, run_values  # pylint: disable=unused-argument
            ):
                if self._step % FLAGS.log_frequency == 0:
                    current_time = time.time()
                    duration = current_time - self._start_time
                    self._start_time = current_time

                    loss_value = run_values.results
                    examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
                    sec_per_batch = float(duration / FLAGS.log_frequency)

                    format_str = (
                        "%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)"
                    )
                    print(
                        format_str
                        % (
                            datetime.now(),
                            self._step,
                            loss_value,
                            examples_per_sec,
                            sec_per_batch,
                        )
                    )

        with tf.train.MonitoredTrainingSession(
            # 设置恢复变量的文件路径为FLAGS.train_dir
            checkpoint_dir=FLAGS.train_dir,
            hooks=[
                # 设置HOOK程序在FLAGS.max_steps=100000后停止
                tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
                # 设置如果loss的值为Nan，停止训练
                tf.train.NanTensorHook(loss),
                # 调用自己定义的_LoggerHook() HOOK类
                _LoggerHook(),
            ],
            # 对会话进行设置，log_device_placement为True时，会在终端打印出各项操作是在哪个设备上运行的
            config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement),
        ) as mon_sess:
            # 创建循环在没有符合程序退出条件的情况下，运行train_op训练数据
            while not mon_sess.should_stop():
                mon_sess.run(train_op)


def main(argv=None):
    # 下载并解压cifar10的数据，防止没有数据可训练
    cifar10.maybe_download_and_extract()
    # 创建目录存储Log文件，如果目录存在，删除重新创建，以保证保存最新的训练信息
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    # 训练数据
    train()


if __name__ == "__main__":
    # 处理FLAGS参数解析，运行main()函数
    tf.app.run()


