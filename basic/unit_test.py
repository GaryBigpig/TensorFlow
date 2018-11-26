import tensorflow as tf


class SquareTest(tf.test.TestCase):
    def testSquare(self):

        with self.test_session():
            # 平方操作
            x = tf.square([2, 3])
            # 测试x的值是否等于[4,9]
            self.assertAllEqual(x.eval(), [3, 9])

class GetTempDir(tf.test.TestCase):
    def testDir(self):
        with self.test_session():
            print(self.get_temp_dir())
    def testDir1(self):
        with self.test_session():
            print(self.get_temp_dir())

if __name__ == "__main__":
    tf.test.main()
