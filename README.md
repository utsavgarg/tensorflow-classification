# tensorflow-classification

Different neural network architechtures implemented in tensorflow for image classification. Weights converted from caffemodels. Some weights were converted using `misc/convert.py` others using [caffe-tensorflow](https://github.com/ethereon/caffe-tensorflow). The weights can be downloaded from [here](https://www.dropbox.com/sh/qpuqj03gv00ba85/AAApqsIe4SqSOrsfpwrYjOema?dl=0). Tested with Tensorflow 1.0. Contributions are welcome!

## Features

* A single call program to classify images using different architechtures (vgg-f, vgg-16, googlenet, resnet-152)
* Retures networks as a dictionary of layers, so accessing activations at intermediate layers is easy
* Functions to classify single image or evaluate on whole validation set

## Usage

* For classification of a single image, `python classify.py --network 'resnet152' --img_path 'misc/sample.jpg'`
* For evaluation over whole ilsvrc validation set `python classify.py --network 'resnet152' --img_list '<list with image names>' --gt_labels '<list with gt labels corresponding to images>'`
* Currently the `--network` argument can take vggf, vgg16, googlenet, resnet152.
