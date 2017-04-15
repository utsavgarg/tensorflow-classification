from nets.vgg_f import vggf
from nets.vgg_16 import vgg16
from nets.googlenet import googlenet
from nets.resnet_152 import resnet152
from misc.utils import img_preprocess
import tensorflow as tf
import numpy as np
import argparse

def validate_arguments(args):
    nets = ['vggf', 'vgg16', 'googlenet', 'resnet152']
    if not(args.network in nets):
        print ('invalid network')
        exit (-1)
    if args.evaluate:
        if args.img_list is None or args.gt_labels is None:
            print ('provide image list and labels')
            exit (-1)

def choose_net(network):
    #placeholder to pass image
    input_image = tf.placeholder(shape=[None, 224, 224, 3],dtype='float32', name='input_image')
    if network == 'vggf':
        return vggf(input_image), input_image
    elif network == 'vgg16':
        return vgg16(input_image), input_image
    elif network == 'googlenet':
        return googlenet(input_image), input_image
    else:
        return resnet152(input_image), input_image

def evaluate(net, im_list, in_im, labels):
    top_1 = 0
    top_5 = 0
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    imgs = open(im_list).readlines()
    gt_labels = open(labels).readlines()
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for i,name in enumerate(imgs):
            im = img_preprocess(name.strip())
            softmax_scores = sess.run(net['prob'], feed_dict={in_im: im})
            inds = np.argsort(softmax_scores[0])[::-1][:5]
            if i!=0 and i%1000 == 0:
                print 'iter: {:5d}\ttop-1: {:04.2f}\ttop-5: {:04.2f}'.format(i, (top_1/float(i))*100, (top_5)/float(i)*100)
            if inds[0] == int(gt_labels[i].strip()):
                top_1 += 1
                top_5 += 1
            elif int(gt_labels[i].strip()) in inds:
                top_5 += 1
    print 'Top-1 Accuracy = {:.2f}'.format(top_1/500.0)
    print 'Top-5 Accuracy = {:.2f}'.format(top_5/500.0)

def predict(net, im_path, in_im):
    synset = open('misc/ilsvrc_synsets.txt').readlines()
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        im = img_preprocess(im_path)
        softmax_scores = sess.run(net['prob'], feed_dict={in_im: im})
        inds = np.argsort(softmax_scores[0])[::-1][:5]
        print '{:}\t{:}'.format('Score','Class')
        for i in inds:
            print '{:.4f}\t{:}'.format(softmax_scores[0,i], synset[i].strip().split(',')[0])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', default='googlenet', help='The network eg. googlenet')
    parser.add_argument('--img_path', default='misc/sample.jpg',  help='Path to input image')
    parser.add_argument('--evaluate', default=False,  help='Flag to evaluate over full validation set')
    parser.add_argument('--img_list',  help='Path to the validation image list')
    parser.add_argument('--gt_labels', help='Path to the ground truth validation labels')
    args = parser.parse_args()
    validate_arguments(args)
    net, inp_im  = choose_net(args.network)
    if args.evaluate:
        evaluate(net, args.img_list, inp_im, args.gt_labels)
    else:
        predict(net, args.img_path, inp_im)

if __name__ == '__main__':
    main()
