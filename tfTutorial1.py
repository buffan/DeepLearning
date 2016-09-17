import tensorflow as tf

x = tf.placeholder(tf.float32, [None, 784])
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros[10])
y = tf.nn.softmax(tf.matmul(x, w) + )

y_ = tf.placeholder(tf.float32, [None, 10])


