from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import re
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import cifar10

# 定义参数变量
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train_check',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
"""Whether to log device placement.""")

def tower_loss(scope, images, labels):
  """Calculate the total loss on a single tower running the CIFAR model.
  Args:
    scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'
    images: Images. 4D tensor of shape [batch_size, height, width, 3].
    labels: Labels. 1D tensor of shape [batch_size].
  Returns:
     Tensor of shape [] containing the total loss for a batch of data
  """

  # 通过cifar10.inference定义的深层学习框架（2层卷积，3层全链接）对images图片数据进行学习
  # 得到各个分类的特征值
  logits = cifar10.inference(images)

  # 将各个分类的特征值和标签数据通过cifar10.loss得到损失值
  _ = cifar10.loss(logits, labels)

  # 获取当前tower下的所有loss
  losses = tf.get_collection('losses',scope)

  # 汇总当前tower下的所有loss，命名total_loss
  total_loss = tf.add_n(losses, name='total_loss')

  # ？
  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    loss_name = re.sub('%s_[0-9]*/' % cifar10.TOWER_NAME, '', l.op.name)
    tf.summary.scalar(loss_name, l)

  return total_loss


def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.
  Note that this function provides a synchronization point across all towers.
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  # 初始化数组average_grads，用来存储所有GPU的平均梯度值
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
    return average_grads

def train():
  # 创建上下文管理器，使用默认的计算图，运行在CPU上
  with tf.Graph().as_default(), tf.device('/cpu:0'):
    # Create a variable to count the number of train() calls. This equals the
    # number of batches processed * FLAGS.num_gpus.
    # 创建变量global_step，初始值为0，不可在优化过程中使用
    global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)

    # Calculate the learning rate schedule.
    # 每个epoch训练的batch数量=每个epoch要训练的样本数/每个batch中的样本数/GPU的个数
    num_batches_per_epoch = (cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN /
                             FLAGS.batch_size / FLAGS.num_gpus)
    # 设置每个epoch中learning rate的衰减次数
    decay_steps = int(num_batches_per_epoch * cifar10.NUM_EPOCHS_PER_DECAY)

    # 初始化学习率INITIAL_LEARNING_RATE后，训练过程中按照LEARNING_RATE_DECAY_FACTOR比例衰减学习率，以免学习率过大造成震荡
    # staircase为True,每decay_steps步数后，更新learning_rate=learning_rate*(decay_rate**decay_steps)
    # global_step为学习步数
    lr = tf.train.exponential_decay(cifar10.INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    cifar10.LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)

    # 设置梯度下降优化算法，学习率lr为随着学习的步数逐渐衰减
    opt = tf.train.GradientDescentOptimizer(lr)

    # Get images and labels for CIFAR-10.
    # 对cifar10的数据做切割、翻转、随机调整亮度、对比度和标准化处理，并按照FLAGS.batch_size参数生成批次数据
    images, labels = cifar10.distorted_inputs()
    # ？
    batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
          [images, labels], capacity=2 * FLAGS.num_gpus)
    # 初始化数组tower_grads，用于存储所有GPU的梯度
    tower_grads = []
    # 创建上下文管理器，设定变量命名空间为当前变量命名空间?
    with tf.variable_scope(tf.get_variable_scope()):
      for i in xrange(FLAGS.num_gpus):
        # 创建上下文管理器，默认设备使用'/gpu:i'
        with tf.device('/gpu:%d' % i):
          # 创建命名空间'tower_i'
          with tf.name_scope('%s_%d' % (cifar10.TOWER_NAME, i)) as scope:
            # 从batch_queue出列一个batch的数据用于在当前的GPU上进行训练
            image_batch, label_batch = batch_queue.dequeue()
            # 对image_batch, label_batch进行训练，得到在当前GPU下的loss
            loss = tower_loss(scope, image_batch, label_batch)

            # 设置变量可以在下一个GPU中再次使用？
            tf.get_variable_scope().reuse_variables()

            # 获取当前的GPU的summary操作？
            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

            # 根据当前GPU下的loss计算梯度
            grads = opt.compute_gradients(loss)

            # 将当前GPU下的梯度存储到tower_grads数组
            tower_grads.append(grads)

    # We must calculate the mean of each gradient. Note that this is the
    # synchronization point across all towers.
    grads = average_gradients(tower_grads)

    # Add a summary to track the learning rate.
    summaries.append(tf.summary.scalar('learning_rate', lr))

    # Add histograms for gradients.
    for grad, var in grads:
      if grad is not None:
        summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

    # Apply the gradients to adjust the shared variables.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
      summaries.append(tf.summary.histogram(var.op.name, var))

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        cifar10.MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # Group all updates to into a single train op.
    train_op = tf.group(apply_gradient_op, variables_averages_op)

    # Create a saver.
    saver = tf.train.Saver(tf.global_variables())

    # Build the summary operation from the last tower summaries.
    summary_op = tf.summary.merge(summaries)

    # Build an initialization operation to run below.
    init = tf.global_variables_initializer()

    # Start running operations on the Graph. allow_soft_placement must be set to
    # True to build towers on GPU, as some of the ops do not have GPU
    # implementations.
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=FLAGS.log_device_placement))
    sess.run(init)

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

    for step in xrange(FLAGS.max_steps):
      start_time = time.time()
      _, loss_value = sess.run([train_op, loss])
      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % 10 == 0:
        num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = duration / FLAGS.num_gpus

        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print (format_str % (datetime.now(), step, loss_value,
                             examples_per_sec, sec_per_batch))

      if step % 100 == 0:
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)

      # Save the model checkpoint periodically.
      if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)

def main(argv=None):
  # 下载并解压cifar10的数据，防止没有数据可训练
  cifar10.maybe_download_and_extract()
  # 创建目录存储Log和checkpoint文件，如果目录存在，删除重新创建，以保证保存最新的训练信息
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  # 训练数据
  train()


if __name__ == '__main__':
    # 处理FLAGS参数解析，运行main()函数
    # tf.app.run()
    sess=tf.InteractiveSession()
    print(sess.run(tf.get_variable_scope()))