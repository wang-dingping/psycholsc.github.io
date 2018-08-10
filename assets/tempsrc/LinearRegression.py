import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
# Suppose in h_theta = theta_0 + theta_1 x
# theta_0 = 1
# theta_1 = 1.5
x = np.random.uniform(0, 10, 25)
y = 1.5 * x + 1 + np.random.normal(0, 0.5, 25)
plt.plot(x, y, 'ro')
# plt.show()

theta_0 = tf.Variable(tf.random_uniform([1]))
theta_1 = tf.Variable(tf.random_uniform([1]))
X = x
h_theta = tf.add(theta_0, theta_1 * x)

cost = tf.reduce_sum((h_theta - y)**2) / 50

alpha = 0.005
optimizer = tf.train.GradientDescentOptimizer(alpha)
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(10000):
        sess.run(train)
        if i % 2000 == 0:
            print('loss:{}'.format(sess.run(cost)))
    print(sess.run(theta_0), sess.run(theta_1))
