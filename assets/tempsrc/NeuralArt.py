import time
import scipy.io
import os
import numpy as np
import tensorflow as tf

# Pre-defination
# Define Width & Height of in/output picture
image_Width = 800
image_Height = 600

# the picture that gives content info.
content_image = './images/LX.jpg'

# the picture that gives Style info.
style_image = './images/nh1.jpg'

# Define output
output_dir = './results'
time0 = time.localtime()
output_img = '{}.result.png'.format(str(time.mktime(time.localtime()))[0:-2])

# Using pre-trained VGG model
vgg_model_path = './imagenet-vgg-verydeep-19.mat'

# Other hyper-parameters
noise_ratio = 0.5
style_strength = 500  # 500

# iterations
iteration = 2000

content_layers = [('conv4_2', 1.)]  # Previously 4_2
style_layers = [('conv1_1', 1.), ('conv2_1', 1.),
                ('conv3_1', 1.), ('conv4_1', 1.), ('conv5_1', 1.)]

step = 2.0

# Mean Value for statistical use
MEAN_VALUES = np.array([123, 117, 104]).reshape((1, 1, 1, 3))


# VGG class, build a vgg structure.


class VGG(object):
    '''Actually these codes are piece of shit.'''

    def __init__(self):
        print('VGG Model preparing...')

    def build(self, process_type, prev_input, weight_bias=None):
        print('Preparing to build {} type of CNN.'.format(str(process_type)))
        if process_type == 'conv':
            return tf.nn.relu(tf.nn.conv2d(prev_input, weight_bias[0], strides=[1, 1, 1, 1], padding='SAME') + weight_bias[1])
        elif process_type == 'pool':
            return tf.nn.avg_pool(prev_input, ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1], padding='SAME')

    def build_vgg19(self, path):
        '''Build a modified VGG - 19 net and return.'''
        net = {}
        # load and extract vgg net
        vgg_raw = scipy.io.loadmat(path)
        vgg_layers = vgg_raw['layers'][0]
        # build input layer
        net['input'] = tf.Variable(
            np.zeros((1, image_Height, image_Width, 3)).astype('float32'))
        # build layer_1 - classic layer with 2 conv and 1 avg_pooling
        net['conv1_1'] = self.build(
            'conv', net['input'], self.get_weight_bias(vgg_layers, 0))
        net['conv1_2'] = self.build(
            'conv', net['conv1_1'], self.get_weight_bias(vgg_layers, 2))
        net['pool1'] = self.build('pool', net['conv1_2'])
        # build layer_2 - the same as 1
        net['conv2_1'] = self.build(
            'conv', net['pool1'], self.get_weight_bias(vgg_layers, 5))
        net['conv2_2'] = self.build(
            'conv', net['conv2_1'], self.get_weight_bias(vgg_layers, 7))
        net['pool2'] = self.build('pool', net['conv2_2'])
        # build layer_3 - 4 conv and 1 avg_pooling
        net['conv3_1'] = self.build(
            'conv', net['pool2'], self.get_weight_bias(vgg_layers, 10))
        net['conv3_2'] = self.build(
            'conv', net['conv3_1'], self.get_weight_bias(vgg_layers, 12))
        net['conv3_3'] = self.build(
            'conv', net['conv3_2'], self.get_weight_bias(vgg_layers, 14))
        net['conv3_4'] = self.build(
            'conv', net['conv3_3'], self.get_weight_bias(vgg_layers, 16))
        net['pool3'] = self.build('pool', net['conv3_4'])
        # build layer_4 - the same as 3
        net['conv4_1'] = self.build(
            'conv', net['pool3'], self.get_weight_bias(vgg_layers, 19))
        net['conv4_2'] = self.build(
            'conv', net['conv4_1'], self.get_weight_bias(vgg_layers, 21))
        net['conv4_3'] = self.build(
            'conv', net['conv4_2'], self.get_weight_bias(vgg_layers, 23))
        net['conv4_4'] = self.build(
            'conv', net['conv4_3'], self.get_weight_bias(vgg_layers, 25))
        net['pool4'] = self.build('pool', net['conv4_4'])
        # build layer_5 - the same as 3
        net['conv5_1'] = self.build(
            'conv', net['pool4'], self.get_weight_bias(vgg_layers, 28))
        net['conv5_2'] = self.build(
            'conv', net['conv5_1'], self.get_weight_bias(vgg_layers, 30))
        net['conv5_3'] = self.build(
            'conv', net['conv5_2'], self.get_weight_bias(vgg_layers, 32))
        net['conv5_4'] = self.build(
            'conv', net['conv5_3'], self.get_weight_bias(vgg_layers, 34))
        net['pool5'] = self.build('pool', net['conv5_4'])
        return net

    def get_weight_bias(self, vgg_layers, i,):
        '''read vgg layers' weights and bias as constant'''
        weights = vgg_layers[i][0][0][0][0][0]
        weights = tf.constant(weights)
        bias = vgg_layers[i][0][0][0][0][1]
        bias = tf.constant(np.reshape(bias, (bias.size)))
        return weights, bias

    def build_loss_content(self, p, x):
        M = p.shape[1] * p.shape[2]
        N = p.shape[3]
        loss = (1. / (2 * N**0.5 * M**0.5)) * tf.reduce_sum(tf.pow((x - p), 2))
        return loss

    def build_loss_style(self, a, x):
        M = a.shape[1] * a.shape[2]
        N = a.shape[3]
        A = self.gram_matrix_val(a, M, N)
        G = self.gram_matrix(x, M, N)
        loss = (1. / (4 * N**2 * M**2)) * tf.reduce_sum(tf.pow((G - A), 2))
        return loss

    def gram_matrix(self, x, area, depth):
        x1 = tf.reshape(x, (area, depth))
        g = tf.matmul(tf.transpose(x1), x1)
        return g

    def gram_matrix_val(self, x, area, depth):
        x1 = x.reshape(area, depth)
        g = np.dot(x1.T, x1)
        return g


def readImage(path):
    image = scipy.misc.imread(path)
    image = scipy.misc.imresize(image, (image_Height, image_Width))
    image = image[np.newaxis, :, :, :]
    image = image - MEAN_VALUES
    return image


def writeImage(path, image):
    image = image + MEAN_VALUES
    image = image[0]
    image = np.clip(image, 0, 255).astype('uint8')
    scipy.misc.imsave(path, image)


def main():
    '''Reference: https://github.com/ckmarkoh/neuralart_tensorflow'''
    vgg_model = VGG()
    net = vgg_model.build_vgg19(vgg_model_path)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    noise_img = np.random.uniform(-20, 20, (1,
                                            image_Height, image_Width, 3)).astype('float32')
    content_img = readImage(content_image)
    style_img = readImage(style_image)
    sess.run([net['input'].assign(content_img)])
    cost_content = sum(map(
        lambda l, : l[1] * vgg_model.build_loss_content(sess.run(net[l[0]]),  net[l[0]]), content_layers))
    sess.run([net['input'].assign(style_img)])
    cost_style = sum(map(
        lambda l: l[1] * vgg_model.build_loss_style(sess.run(net[l[0]]),  net[l[0]]), style_layers))
    cost_total = cost_content + style_strength * cost_style
    optimizer = tf.train.AdamOptimizer(step)

    train = optimizer.minimize(cost_total)
    sess.run(tf.global_variables_initializer())
    # initial input == 0.7*noise + 0.3*content
    sess.run(net['input'].assign(noise_ratio *
                                 noise_img + (1. - noise_ratio) * content_img))

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for i in range(iteration):
        sess.run(train)
        if i % 20 == 0:
            result_img = sess.run(net['input'])
            print('Iteration:{} with total_loss:{}'.format(
                i, sess.run(cost_total)))
            writeImage(os.path.join(output_dir, '%s.png' %
                                    (str(i).zfill(4))), result_img)

    writeImage(os.path.join(output_dir, output_img), result_img)


if __name__ == '__main__':
    main()
