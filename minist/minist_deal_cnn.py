# Cnn 处理已经处理过的minist数据集，使用 Softmax 回归模型

#!/usr/bin/python
import pickle as p
import tensorflow as tf
import numpy as np

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

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') #卷积时在图像每一维的步长，这是一个一维的向量，长度4，strides[0]=strides[3]=1，返回与输入x相同的tensor

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def main():
	train_data, validation_data, test_data = load_train("E:/datasets/minist_datasets/minist_deal/mnist.pkl")
	# 训练集
	train_x = train_data[0]  # 特征矩阵向量 50000x784
	
	train_y = train_data[1]  # 由于与y_不匹配，需要把标签数据是"one-hot vectors"， 一个one-hot向量除了某一位的数字是1以外其余各维度数字都是0
	train_y_one_hot = one_hot(train_y, train_y.shape[0]) # label转为1x10的向量，相应的位置为1其余为0,例如label=2[0,0,1,0...]

	# 测试集转化
	test_x = test_data[0]
	test_y = test_data[1]
	test_y_one_hot = one_hot(test_y, test_y.shape[0])
	print(test_data[0].shape)
	print(test_x)
	print(test_y_one_hot)
	sess = tf.InteractiveSession()

	x = tf.placeholder("float", shape=[None, 784])
	y_ = tf.placeholder("float", shape=[None, 10])

	W_conv1 = weight_variable([5, 5, 1, 32]) #[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]
	b_conv1 = bias_variable([32])

	x_image = tf.reshape(x, [-1, 28, 28, 1]) # 转为4d，[训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]

	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # 修正线性
	h_pool1 = max_pool_2x2(h_conv1)

	W_conv2 = weight_variable([5, 5, 32, 64])
	b_conv2 = bias_variable([64])

	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)

	# Now image size is reduced to 7*7
	W_fc1 = weight_variable([7 * 7 * 64, 1024])
	b_fc1 = bias_variable([1024])

	h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

	keep_prob = tf.placeholder("float")
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

	W_fc2 = weight_variable([1024, 10])
	b_fc2 = bias_variable([10])
	y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
	#以上是整个卷积结构的过程，调用tensorflow的函数

	cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	sess.run(tf.global_variables_initializer())
	for i in range(1000):
  		index = i * 50
  		if i%50 == 0:
  			train_accuracy = accuracy.eval(feed_dict={x: train_x[index:index+50, :], y_: train_y_one_hot[index:index+50, :], keep_prob: 1.0})
  			print("step %d, training accuracy %.3f"%(i, train_accuracy))
  		train_step.run(feed_dict={x: train_x[index:index+50, :], y_: train_y_one_hot[index:index+50, :], keep_prob: 0.5})

	mean_acc = 0
	for j in range(100):
		index = j * 100
		acc = accuracy.eval(feed_dict={x: test_x[index:index+100, :], y_: test_y_one_hot[index:index+100, :], keep_prob:1.0})
		mean_acc = mean_acc+acc*0.01

	print("精确度:",mean_acc )
	print ("Training finished")
if __name__ == "__main__":
    main()
