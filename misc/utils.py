import numpy as np
from skimage.io import imread
from skimage.transform import resize

#util layers
def img_preprocess(img_path, size=224):
    mean = [103.939, 116.779, 123.68]
    img = imread(img_path)
    img = resize(img, (size, size))*255.0
    if len(img.shape) == 2:
        img = np.dstack([img,img,img])
    img[:,:,0] -= mean[2]
    img[:,:,1] -= mean[1]
    img[:,:,2] -= mean[0]
    img[:,:,[0,1,2]] = img[:,:,[2,1,0]]
    img = np.reshape(img,[1,size,size,3])
    return img

def v3_preprocess(img_path):
    img = imread(img_path)
    img = resize(img, (299, 299), preserve_range=True)
    img = (img - 128) / 128
    if len(img.shape) == 2:
        img = np.dstack([img,img,img])
    img = np.reshape(img,[1,299,299,3])
    return img
