from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf

import cifar10_input

# 创建命令行参数
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer(
    "batch_size", 128, """Number of images to process in a batch."""
)
tf.app.flags.DEFINE_string(
    "data_dir",
    os.path.join(os.getcwd(), "cifar10_train_data"),
    """Path to the CIFAR-10 data directory.""",
)
tf.app.flags.DEFINE_boolean("use_fp16", False, """Train the model using fp16.""")

# 设置全局变量
IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0  # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1  # Initial learning rate.
TOWER_NAME = "tower"
DATA_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"


def _activation_summary(x):
    # 设置tensor_name，如果输入的tensor的名称里，包含'tower_'+数字+'/'的字符串，将其替换成''
    tensor_name = re.sub("%s_[0-9]*/" % TOWER_NAME, "", x.op.name)
    # 为输入的tensor创建直方图，节点命名为tensor_name + "/activations"，可以在tensorboard中显示
    tf.summary.histogram(tensor_name + "/activations", x)
    # 因为relu激活函数有可能造成大量参数为0，所以使用tf.nn.zero_fraction计算输入tensor x中0元素个数在所有元素个数中的比例
    # 在tensorboar中打印，节点命名为tensor_name + "/sparsity"
    # 参考https://blog.csdn.net/fegang2002/article/details/83539768
    tf.summary.scalar(tensor_name + "/sparsity", tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
    # 创建变量运行在CPU中
    with tf.device("/cpu:0"):
        # 根据参数FLAGS.use_fp16，设置变量的类型
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        # 初始化变量
        var = tf.get_variable(
            name=name, shape=shape, initializer=initializer, dtype=dtype
        )
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    # 根据参数FLAGS.use_fp16设置确定变量的类型
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    # 使用函数_variable_on_cpu创建变量，使用tf.truncated_normal随机生成数据
    # 在tf.truncated_normal中如果x的取值在区间（μ-2σ，μ+2σ）之外则重新进行选择
    # 横轴区间（μ-2σ，μ+2σ）内的面积为95.449974%，这样保证了生成的值都在均值附近。
    var = _variable_on_cpu(
        name=name,
        shape=shape,
        initializer=tf.truncated_normal(stddev=stddev, dtype=dtype),
    )
    if wd is not None:
        # 如果wd参数有值的话，将张量var的各个元素的平方和除以2，
        # 然后与wd点乘，命名为weight_loss
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name="weight_loss")
        # 将weight_decay加入losses集合
        tf.add_to_collection("losses", weight_decay)
    return var


def distorted_inputs():
    # 如果FLAGS.data_dir为空，报错提示
    if not FLAGS.data_dir:
        raise ValueError("Please supply a data_dir")
    # cifar10的数据解压后保存在data_dir
    data_dir = os.path.join(FLAGS.data_dir, "cifar-10-batches-bin")
    # 对cifar10的数据做切割、翻转、随机调整亮度、对比度和标准化处理，并按照FLAGS.batch_size参数生成批次数据
    images, labels = cifar10_input.distorted_inputs(
        data_dir=data_dir, batch_size=FLAGS.batch_size
    )
    # 如果FLAGS.use_fp16设置为真，将images,labels的类型转换为tf.float16
    if FLAGS.use_fp16:
        images = tf.cast(images, tf.float16)
        labels = tf.cast(labels, tf.float16)
    return images, labels


def inputs(eval_data):
    # 如果FLAGS.data_dir为空，报错提示
    if not FLAGS.data_dir:
        raise ValueError("Please supply a data_dir")
    # cifar10的数据解压后保存在data_dir
    data_dir = os.path.join(FLAGS.data_dir, "cifar-10-batches-bin")
    # 调用cifar10_input.inputs函数对cifar10数据进行处理
    images, labels = cifar10_input.inputs(
        eval_data=eval_data, data_dir=data_dir, batch_size=FLAGS.batch_size
    )
    # 如果FLAGS.use_fp16设置为真，将images,labels的类型转换为tf.float16
    if FLAGS.use_fp16:
        images = tf.cast(images, tf.float16)
        labels = tf.cast(labels, tf.float16)
    return images, labels


def inference(images):
    # 定义卷积层，变量的作用域为conv1
    with tf.variable_scope("conv1") as scope:
        # 创建变量kernel作为卷积核，命名为'weights'，
        # 卷积核高为5，宽为5，输入通道也即是图片的通道为3，输出通道也即是卷积核的数量为64
        # 按照标准差为5e-2的正态分布随机生成数据，抛弃均值左右2倍标准差外的数据
        # 不设置调整参数wd
        kernel = _variable_with_weight_decay(
            "weights", shape=[5, 5, 3, 64], stddev=5e-2, wd=None
        )
        # 创建卷积层，输入为images[image_batch,image_height,image_width,image_channel]
        # 卷积核为kernel[kernel_height,kernel_width,image_channel,kernel_channel]，kernel_channel为卷积核的数量
        # 步长strides为[1, 1, 1, 1]，为卷积核分别在images的四个维度[image_batch,image_height,image_width,image_channel]上的步长，
        # padding： string类型，值为“SAME” 和 “VALID”，表示的是卷积的形式，是否考虑边界。
        # "SAME"是考虑边界，不足的时候用0去填充周围，"VALID"则不考虑
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding="SAME")
        # 设置偏置，命名为'biases'，长度为64的一维向量，所有元素都为0
        biases = _variable_on_cpu("biases", [64], tf.constant_initializer(0.0))
        # 将卷积层和偏置加到一起
        pre_activation = tf.nn.bias_add(conv, biases)
        # 卷积层和偏置加在一起后添加relu的激活函数，得到第一层卷积，命名为'conv1'，relu激活函数可能会造成大量参数为0
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
        # 在tensorboad中打印conv1的分布和0元素占比，0元素占比可以反映此层对于训练的作用，占比高作用小，占比低作用大
        _activation_summary(conv1)

        # 设置最大池化层对conv1层做最大池化处理，命名为"pool1"
        # 池化窗口的大小设置为[1, 3, 3, 1]，分别对应conv1的四个维度[batch,height,width,channel]
        # 步长为[1, 2, 2, 1]，分别对应conv1的四个维度[batch,height,width,channel]
        # padding设置"SAME"，当滑动到边界尺寸不足时用'0'填充
        pool1 = tf.nn.max_pool(
            conv1,
            ksize=[1, 3, 3, 1],
            strides=[1, 2, 2, 1],
            padding="SAME",
            name="pool1",
        )
        # 对池化后的结果pool1做局部相应标准化处理，类似于dropout，防止过拟合
        norm1 = tf.nn.lrn(
            pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name="norm1"
        )

    # 定义卷积层，变量的作用域为conv2
    with tf.variable_scope("conv2") as scope:
        # 创建变量kernel作为卷积核，命名为'weights'，
        # 卷积核高为5，宽为5，输入通道也即是图片的通道为64，输出通道也即是卷积核的数量为64
        # 按照标准差为5e-2的正态分布随机生成数据，抛弃均值左右2倍标准差外的数据
        # 不设置调整参数wd
        kernel = _variable_with_weight_decay(
            "weights", shape=[5, 5, 64, 64], stddev=5e-2, wd=None
        )
        # 创建卷积层，输入为norm1，卷积核为kernel，步长strides为[1, 1, 1, 1]
        # padding："SAME"是考虑边界，不足的时候用0去填充周围
        conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding="SAME")
        # 设置偏置，命名为'biases'，长度为64的一维向量，所有元素都为0.1
        biases = _variable_on_cpu("biases", [64], tf.constant_initializer(0.1))
        # 将卷积层和偏置加到一起
        pre_activation = tf.nn.bias_add(conv, biases)
        # 卷积层和偏置加在一起后添加relu的激活函数，得到第二层卷积，命名为'conv2'，relu激活函数可能会造成大量参数为0
        conv2 = tf.nn.relu(pre_activation, name=scope.name)
        # 在tensorboad中打印conv2的分布和0元素占比，0元素占比可以反映此层对于训练的作用，占比高作用小，占比低作用大
        _activation_summary(conv2)

        # 对卷积后的结果conv2做局部相应标准化处理，类似于dropout，防止过拟合
        norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name="norm2")
        # 设置最大池化层对norm2层做最大池化处理，命名为"pool2"
        # 池化窗口的大小设置为[1, 3, 3, 1]，分别对应pool2的四个维度[batch,height,width,channel]
        # 步长为[1, 2, 2, 1]，分别对应pool2的四个维度[batch,height,width,channel]
        # padding设置"SAME"，当滑动到边界尺寸不足时用'0'填充
        pool2 = tf.nn.max_pool(
            norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool2"
    )

    # 定义全连接层，变量的作用域为local3
    with tf.variable_scope("local3") as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        # 将pool2转化为二维张量，第一维的尺寸为image的第一维的尺寸，其余的元素自动换算为第二维的尺寸，返回张量reshape
        reshape = tf.reshape(pool2, [images.get_shape().as_list()[0], -1])
        # 获取reshape张量的第二维的尺寸
        dim = reshape.get_shape()[1].value
        # 创建权重变量weights为二维张量，第一维的尺寸为张量reshape的第二维尺寸，第二维的尺寸为384
        # 按照标准差为0.04的正态分布随机生成数据，抛弃均值左右2倍标准差外的数据
        # 设置调整参数wd=0.004
        weights = _variable_with_weight_decay(
            "weights", shape=[dim, 384], stddev=0.04, wd=0.004
        )
        # 设置偏置，命名为'biases'，长度为64的一维向量，所有元素都为0.1
        biases = _variable_on_cpu("biases", [384], tf.constant_initializer(0.1))
        # 设置local3为reshape和weights相乘再与biases相加，再添加激活函数relu，relu激活函数可能会造成大量参数为0
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        # 在tensorboad中打印local3的分布和0元素占比，0元素占比可以反映此层对于训练的作用，占比高作用小，占比低作用大
        _activation_summary(local3)

    # 定义全连接层，变量的作用域为local4
    with tf.variable_scope("local4") as scope:
        # 创建权重变量weights为二维张量，第一维的尺寸为384，第二维的尺寸为192
        # 按照标准差为0.04的正态分布随机生成数据，抛弃均值左右2倍标准差外的数据
        # 设置调整参数wd=0.004
        weights = _variable_with_weight_decay(
            "weights", shape=[384, 192], stddev=0.04, wd=0.004
        )
        # 设置偏置，命名为'biases'，长度为192的一维向量，所有元素都为0.1
        biases = _variable_on_cpu("biases", [192], tf.constant_initializer(0.1))
        # 设置local4为local3和weights相乘再与biases相加，再添加激活函数relu，relu激活函数可能会造成大量参数为0
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
        # 在tensorboad中打印local4的分布和0元素占比，0元素占比可以反映此层对于训练的作用，占比高作用小，占比低作用大
        _activation_summary(local4)

    # 定义全连接层，变量的作用域为softmax_linear
    with tf.variable_scope("softmax_linear") as scope:
        # 创建权重变量weights为二维张量，第一维的尺寸为192，第二维的尺寸为图片的分类数
        # 按照标准差为1/192.0的正态分布随机生成数据，抛弃均值左右2倍标准差外的数据
        # 不设置调整参数wd
        weights = _variable_with_weight_decay(
            "weights", [192, NUM_CLASSES], stddev=1 / 192.0, wd=None
        )
        # 设置偏置，命名为'biases'，长度为图片分类数的一维向量，所有元素都为0
        biases = _variable_on_cpu("biases", [NUM_CLASSES], tf.constant_initializer(0.0))
        # 设置softmax_linear为local4和weights相乘再与biases相加
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
        # 在tensorboad中打印softmax_linear的分布和0元素占比，0元素占比可以反映此层对于训练的作用，占比高作用小，占比低作用大
        _activation_summary(softmax_linear)
    # 张量softmax_linear作为函数输出
    return softmax_linear


def loss(logits, labels):
    # 将labels的类型转化为tf.int64
    labels = tf.cast(labels, tf.int64)
    # 求logits和labels之间的交叉熵，命名"cross_entropy_per_example"
    # tf.nn.sparse_softmax_cross_entropy_with_logits（）比tf.nn.softmax_cross_entropy_with_logits多了一步将labels稀疏化
    # 此例用非稀疏的标签，所以用tf.nn.sparse_softmax_cross_entropy_with_logits（）
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name="cross_entropy_per_example"
    )
    # 对logits和labels之间的交叉熵cross_entropy求均值返回给cross_entropy_mean，命名"cross_entropy"
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name="cross_entropy")
    # 将cross_entropy_mean存入losses集合
    tf.add_to_collection("losses", cross_entropy_mean)
    # 将集合中的元素相加作为函数的返回值
    return tf.add_n(tf.get_collection("losses"), name="total_loss")


def _add_loss_summaries(total_loss):
    # 设置移动平均模型，设置参数decay=0.9？
    # 参考https://blog.csdn.net/qq_14845119/article/details/78767544
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name="avg")
    # 从集合Losses中取出损失函数的值
    losses = tf.get_collection("losses")
    # 将从集合Losses中取出损失函数的值losses和输入的total_loss作和，然后做移动平均，作为函数的返回值
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        # 在tensorboard中打印所有的lose的值
        tf.summary.scalar(l.op.name + " (raw)", l)
        # 在tensorboard中打印所有的lose移动平均之后的值？
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op


def train(total_loss, global_step):
    # 设置每个epoch训练的batch数
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    # 设置每个epoch中learning rate的衰减次数
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
    # 初始化学习率INITIAL_LEARNING_RATE后，训练过程中按照LEARNING_RATE_DECAY_FACTOR比例衰减学习率，以免学习率过大造成震荡
    # staircase为True,每decay_steps步数后，更新learning_rate=learning_rate*(decay_rate**decay_steps)
    # staircase为False，每一步更新learning_rate=learning_rate*decay_rate
    # global_step为学习步数
    lr = tf.train.exponential_decay(
        INITIAL_LEARNING_RATE,
        global_step,
        decay_steps,
        LEARNING_RATE_DECAY_FACTOR,
        staircase=True,
    )
    # 在tensorboard中打印Learning rate
    tf.summary.scalar("learning_rate", lr)
    # 将从集合Losses中取出损失函数的值losses和输入的total_loss作和，然后做移动平均
    loss_averages_op = _add_loss_summaries(total_loss)
    # 上下文管理器，控制计算流图，指定计算顺序，优先执行loss_averages_op
    with tf.control_dependencies([loss_averages_op]):
        # 设置梯度下降优化算法，学习率lr为随着学习的步数逐渐衰减
        opt = tf.train.GradientDescentOptimizer(lr)
        # 计算total_loss的梯度
        grads = opt.compute_gradients(total_loss)
    # 执行梯度下降，执行之前根据上下文管理器先操作loss_averages_op对total_loss做移动平均
    # 然后再对total_loss做梯度下降优化
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    # 在tensorboard中打印所有的可训练变量
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)
    # 在tensorboard中打印所有梯度优化过程中更新的梯度
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + "/gradients", grad)
    # 设置移动平均模型，设置参数decay=MOVING_AVERAGE_DECAY，num_updates=global_step？
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step
    )
    # 上下文管理器，控制计算流图，指定计算顺序，优先执行apply_gradient_op
    with tf.control_dependencies([apply_gradient_op]):
        # 先执行apply_gradient_op，更新tf.trainable_variables()
        # 然后再对tf.trainable_variables()做移动平均
        variables_averages_op = variable_averages.apply(tf.trainable_variables())
    # 返回根据梯度下降优化且经过移动平均的参数变量
    return variables_averages_op


def maybe_download_and_extract():
    # 下载cifar10的样本数据，并解压
    # 设置cifar10样本数据存储的文件夹，如果文件夹不存在，系统自动创建
    dest_directory = FLAGS.data_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    # 将cifar10的样本数据的下载链接DATA_URL按照'/'截取后取最后一个元素，其为文件名称
    filename = DATA_URL.split("/")[-1]
    # 组合cifar10的样本数据的完整路径
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        # 如果cifar10的样本数据在系统中不存在，下载
        # 定义_progress回调函数，显示下载的进度
        def _progress(count, block_size, total_size):
            # 打印下载进度
            sys.stdout.write(
                "\r>> Downloading %s %.1f%%"
                % (filename, float(count * block_size) / float(total_size) * 100.0)
            )
            # linux系统下系统刷新输出，每秒输出一个结果，windows系统不需要，总是每秒输出一个结果
            sys.stdout.flush()

        # 从DATA_URL下载cifar10的样本数据，保存为filepath
        # 使用回到函数_progress显示下载进度
        # urlretrieve每下载一部分数据块后将下载的数据块数量count、数据库大小block_size和
        # 下载文件的总大小total_size传给回调函数_progress处理，打印下载进度
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        # 获取cifar10样本数据的系统状态信息
        statinfo = os.stat(filepath)
        # 打印cifar10样本数据下载成功信息，显示下载后的文件路径、名称和大小
        print("Successfully downloaded", filename, statinfo.st_size, "bytes.")
    # cifar10样本数据解压后会生成文件夹cifar-10-batches-bin
    extracted_dir_path = os.path.join(dest_directory, "cifar-10-batches-bin")
    if not os.path.exists(extracted_dir_path):
        # 如果extracted_dir_path在系统中不存在，说明cifar10样本数据还未解压
        # 将cifar10样本数据解压后保存到dest_directory
        tarfile.open(filepath, "r:gz").extractall(dest_directory)