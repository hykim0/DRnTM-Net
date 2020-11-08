import cv2 as cv
import random
import tensorflow as tf
import numpy as np
from cv2.ximgproc import guidedFilter


def decomposition(input):
    # 1. Logarithm
    log_input = np.log(input + 1e-6)

    # 2. Scale to [0,1]
    log_input = (log_input - log_input.min() + 1e-6) / (log_input.max() - log_input.min() + 1e-6)

    # 3. Guided filter
    base = guidedFilter(log_input.astype('single'), log_input.astype('single'), radius=3, eps=0.1)

    # 4. Decomposition
    detail = log_input - base

    return base, detail


def decomposition_NotLog(input):
    # 1. Logarithm
    # log_input = np.log(input + 1e-6)

    # 2. Scale to [0,1]
    # log_input = (log_input - log_input.min() + 1e-6) / (log_input.max() - log_input.min() + 1e-6)

    # 3. Guided filter
    base = guidedFilter(input.astype('single'), input.astype('single'), radius=3, eps=0.1)

    # 4. Decomposition
    detail = input - base

    return base, detail

def randomCropHL(img, img2, width, height):
    if img.shape[0] < height or img.shape[1] < width:
        img = cv.resize(img, dsize=(width, height))
        img2 = cv.resize(img2, dsize=(width, height))
    else:
        x = random.randint(np.round((img.shape[1] - width)/2.1), np.round((img.shape[1] - width)*1.1/2.1))
        y = random.randint(np.round((img.shape[0] - height)/2.1), np.round((img.shape[0] - height)*1.1/2.1))
        img = img[y:y + height, x:x + width]
        img2 = img2[y:y + height, x:x + width]
    return img, img2


def randomCrop(img, width, height):
    if img.shape[0] < height or img.shape[1] < width:
        img = cv.resize(img, dsize=(width, height))
    else:
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        img = img[y:y + height, x:x + width]
    return img


def ref_padding(tensor, pad_size=1):
    output = tf.pad(tensor, [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]], "REFLECT")
    return output