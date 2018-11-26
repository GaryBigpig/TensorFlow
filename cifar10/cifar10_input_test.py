from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

import cifar10.cifar10_input


class CIFAR10InputTest(tf.test.TestCase):
    def _record(self, label, red, green, blue):
        # 计算每张图的像素点总数，cifar10的图片的像素是32*32
        image_size = 32 * 32
        # 将每张图的特征抽象成一个特征向量，由标签值和像素数个RGB值组成，
        # 既每个特征向量包括一个标签值，32个Red值，32个Green值和32个Blue值
        # [label,red,...,red,green,...,green,blue,...,blue]
        # 最后转换为二进制数组
        record = bytes(
            bytearray(
                [label]
                + [red] * image_size
                + [green] * image_size
                + [blue] * image_size
            )
        )
        # 构造数组模拟图片像素，为32行，32列，每个行列交叉处为RGB的值组成的向量[red,green,blue]
        expected = [[[red, green, blue]] * 32] * 32
        return record, expected

    def testSimple(self):
        # 设置标签向量，包含三个标签值
        labels = [9, 3, 0]
        # 构造测试数组，包含三个记录
        records = [
            self._record(labels[0], 0, 128, 255),
            self._record(labels[1], 255, 0, 1),
            self._record(labels[2], 254, 255, 0),
        ]
        # 将测试数组中的特征向量record值组合在一起形成一个字符串contents
        contents=b"".join([record for record,_ in records])
        # 将测试数组中的expected值组合在一起形成数组
        expected=[expected for _,expected in records]
        # 由tf.test.TestCase类随机生成一个临时路径存放临时文件"cifar"
        filename=os.path.join(self.get_temp_dir(),"cifar")
        #  生成临时文件"cifar"，将contents值写入文件
        open(filename,"wb").write(contents)

        with self.test_session() as sess:
            q=tf.FIFOQueue(99,[tf.string],shapes=())
            q.enqueue([filename]).run()
            q.close().run()
            result=cifar10.cifar10_input.read_cifar10(q)

            for i in range(3):
                key,label,uint8image=sess.run([result.key,result.label,result.uint8image])
                # print(key)
                # print(label)
                self.assertEqual("%s:%d" % (filename,i),tf.compat.as_text(key))
                self.assertEqual(labels[i],label)
                self.assertAllEqual(expected[i],uint8image)

            with self.assertRaises(tf.errors.OutOfRangeError):
                sess.run([result.key,result.uint8image])

if __name__ == '__main__':
    tf.test.main()


