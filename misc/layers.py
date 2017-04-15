'''
wrapper functions for tensorflow layers
'''
import tensorflow as tf

def conv_layer(bottom, weight, bias=None, s=1, padding='SAME', relu=True):
    conv = tf.nn.conv2d(bottom, weight, [1, s, s, 1], padding=padding)
    if bias is None:
        if relu:
            return tf.nn.relu(conv)
        else:
            return conv
    else:
        bias = tf.nn.bias_add(conv, bias)
        if relu:
            return tf.nn.relu(bias)
        else:
            return bias

def max_pool(bottom, k=3, s=1, padding='SAME'):
     return tf.nn.max_pool(bottom, ksize=[1, k, k, 1], strides=[1, s, s, 1], padding=padding)

def fully_connected(bottom, weight, bias):
    fc = tf.nn.bias_add(tf.matmul(bottom, weight), bias)
    return fc

def batch_norm(bottom, weight):
    bn = tf.nn.batch_normalization(bottom, weight['mean'], weight['variance'], weight['offset'], weight['scale'], 1e-5)
    return bn

def inception_block(bottom, name, weights, biases):
    with tf.name_scope(name+'1x1'):
        branch_1 = conv_layer(bottom, weights['inception_'+name+'_1x1'], biases['inception_'+name+'_1x1'])

    with tf.name_scope(name+'3x3'):
        branch_2 = conv_layer(bottom, weights['inception_'+name+'_3x3_reduce'], biases['inception_'+name+'_3x3_reduce'])
        branch_2 = conv_layer(branch_2, weights['inception_'+name+'_3x3'], biases['inception_'+name+'_3x3'])

    with tf.name_scope(name+'5x5'):
        branch_3 = conv_layer(bottom, weights['inception_'+name+'_5x5_reduce'], biases['inception_'+name+'_5x5_reduce'])
        branch_3 = conv_layer(branch_3, weights['inception_'+name+'_5x5'], biases['inception_'+name+'_5x5'])

    with tf.name_scope(name+'pool'):
        branch_4 = max_pool(bottom)
        branch_4 = conv_layer(branch_4, weights['inception_'+name+'_pool_proj'], biases['inception_'+name+'_pool_proj'])

    return tf.concat(axis=3, values=[branch_1, branch_2, branch_3, branch_4])

def res_block(bottom, name, weights, stride=1, first=False):
    with tf.name_scope(name+'_a'):
        c1 = conv_layer(bottom, weights['res'+name+'_branch2a']['weights'], s=stride, relu=False)
        bn1 = batch_norm(c1, weights['bn'+name+'_branch2a'])
        r1 = tf.nn.relu(bn1)

    with tf.name_scope(name+'_b'):
        c2 = conv_layer(r1, weights['res'+name+'_branch2b']['weights'], relu=False)
        bn2 =  batch_norm(c2, weights['bn'+name+'_branch2b'])
        r2 = tf.nn.relu(bn2)

    with tf.name_scope(name+'_c'):
        c3 = conv_layer(r2, weights['res'+name+'_branch2c']['weights'], relu=False)
        bn3 = batch_norm(c3, weights['bn'+name+'_branch2c'])

    if first:
        with tf.name_scope(name+'_1'):
            c4 = conv_layer(bottom, weights['res'+name+'_branch1']['weights'], s=stride, relu=False)
            bn4 = batch_norm(c4, weights['bn'+name+'_branch1'])
        return tf.nn.relu(tf.add(bn4,bn3))
    else:
        return tf.nn.relu(tf.add(bottom,bn3))
