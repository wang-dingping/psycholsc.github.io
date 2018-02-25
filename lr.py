import numpy as np
from matplotlib import pyplot as plt
w = np.random.random((1, 1))
xlist = []
ylist = []
size = 10000
for i in range(size):
    noise = np.random.normal()
    x = np.random.random((1, 1)) * 100
    y = np.dot(w.T, x)
    xlist.append(x[0])
    ylist.append(y[0] + noise)
plt.plot(xlist, ylist, 'ro')
plt.show()
xcopy, ycopy = xlist[:], ylist[:]
xlist, ylist = np.mat(xlist), np.mat(ylist)
w1 = np.dot(np.dot(np.dot(xlist.T, xlist).I, xlist.T), ylist)
print(w1, w)
y1 = np.dot(xlist, w1)
loss = np.mat(y1) - ylist
sum_ = 0
for i in loss:
    sum_ += i**2
print("loss:{}".format(float(sum_ / size)))

import tensorflow as tf
batch_size = 5
x_data = tf.placeholder('float', [None, 1])
Weight = tf.Variable(tf.random_normal((1, 1)))
y = tf.matmul(x_data, Weight)
y_data = tf.placeholder('float', [None, 1])
loss = tf.reduce_mean(tf.square(y_data - y))
optimizer = tf.train.GradientDescentOptimizer(0.0000002).minimize(loss)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    k = 0
    for i in range(10000):
        if k == size:
            k = 0
        sess.run(optimizer, feed_dict={
                 x_data: xcopy[k:k + batch_size], y_data: ycopy[k:k + batch_size]})
        k += batch_size
        if i % 100 == 0:
            print(sess.run(loss, feed_dict={
                x_data: xcopy[k:k + batch_size], y_data: ycopy[k:k + batch_size]}))

    print(sess.run(Weight), w1, w)
