# tensorflow-classification

Different neural network architechtures implemented in tensorflow for image classification. Weights converted from caffemodels. Some weights were converted using `misc/convert.py` others using [caffe-tensorflow](https://github.com/ethereon/caffe-tensorflow). The weights can be downloaded from [here](https://www.dropbox.com/sh/qpuqj03gv00ba85/AAApqsIe4SqSOrsfpwrYjOema?dl=0). Tested with Tensorflow 1.0. Contributions are welcome!

## Features

* A single call program to classify images using different architechtures (vgg-f, vgg-16, googlenet, resnet-152)
* Returns networks as a dictionary of layers, so accessing activations at intermediate layers is easy
* Functions to classify single image or evaluate on whole validation set

## Usage

* For classification of a single image, `python classify.py --network 'resnet152' --img_path 'misc/sample.jpg'`
* For evaluation over whole ilsvrc validation set `python classify.py --network 'resnet152' --img_list '<list with image names>' --gt_labels '<list with gt labels corresponding to images>'`
* Currently the `--network` argument can take vggf, vgg16, googlenet, resnet152.

## Performance
These converted models have the following performance on the ilsvrc validation set, with each image resized to 224x224, and per channel mean subtraction.

| Network        | Top-1 Accuracy           | Top-5 Accuracy  |
| ------------- |:-------------:| :-----:|
| VGG-F      | 53.43% | 77.43% |
| VGG-16      | 65.77%      |   86.65% |
| GoogLeNet | 67.92%      |    88.29% |
| ResNet-152 | 72.64% |    90.92% |
