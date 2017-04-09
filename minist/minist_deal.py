# 处理已经处理过的minist数据集，使用 Softmax 回归模型
import pickle as p
import tensorflow as tf
import numpy as np

# 读取数据


def load_train(filename):
    with open(filename, 'rb') as f:
        training_data, validation_data, test_data = p.load(f, encoding='latin1')
    return training_data, validation_data, test_data

# one-hot vectors


def one_hot(vec, size):
    one_hot_v = np.zeros(shape=[size, 10])
    for i in range(size):
        one_hot_v[i, vec[i]] = 1
    return one_hot_v

def main():
    sess = tf.InteractiveSession()
    train_data, validation_data, test_data = load_train("E:/datasets/minist_datasets/minist_deal/mnist.pkl")
    # 训练集转化
    train_x = train_data[0]
    train_y = train_data[1]  # 由于与y_不匹配，需要把标签数据是"one-hot vectors"， 一个one-hot向量除了某一位的数字是1以外其余各维度数字都是0
    train_y_one_hot = one_hot(train_y, train_y.shape[0])

    # 测试集转化
    test_x = test_data[0]
    test_y = test_data[1]
    test_y_one_hot = one_hot(test_y, test_y.shape[0])

    print("测试集",train_data[0].shape)
    print("测试集labels",train_data[1].shape)

    print("校正集",validation_data[0].shape)
    print("校正集labels",validation_data[1].shape)
    
    print("测试集",test_data[0].shape)
    print("测试集labels",test_data[1].shape)



    print(train_y)
    print(train_y_one_hot)

    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    w = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    y = tf.nn.softmax(tf.matmul(x, w) + b)

    # 交叉熵，交叉熵产生于信息论里面的信息压缩编码技术
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    init_op = tf.global_variables_initializer()
    # 第一种
    init_op.run()
    for i in range(1000):
        index = i * 50
        train_step.run(feed_dict={x: train_x[index:index+50, :], y_: train_y_one_hot[index:index+50, :]})
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("精确度:", accuracy.eval(feed_dict={x: test_x[:, :], y_: test_y_one_hot[:, :]}))

    # 第二种

    # with tf.Session() as s:
    #     s.run(init_op)
    #     for i in range(1000):
    #         index = i * 50
    #         train_step.run(feed_dict={x: train_x[index:index+50, :], y_: train_y_one_hot[index:index+50, :]})
    #         correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    #         accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #      print("精确度:", accuracy.eval(feed_dict={x: train_x[:, :], y_: train_y_one_hot[:, :]}))

if __name__ == "__main__":
    main()
