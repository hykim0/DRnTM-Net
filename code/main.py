import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np
import cv2 as cv
import os
import time
import random
import pickle
from module import *
from unet import *
from guided_filter_tf.guided_filter import fast_guided_filter, guided_filter

# Proposed network
Width = 320
Height = 320
batch_size = 1
nClass = 2

# Variables
I = tf.placeholder(tf.float32, shape=[None, None, None, 1])  # 0~1 input image
LR = tf.placeholder(tf.float32)

# initial base & detail layers
I_d = tf.placeholder(tf.float32, shape=[None, None, None, 1])  # 0~1 input image
I_b = tf.placeholder(tf.float32, shape=[None, None, None, 1])  # 0~1 input image

# DR-Net
I_dr = unet(I_d, outputChn=1, name='test_Dt_rstr', reuse=False)
I_dr = tf.nn.tanh(I_dr)
I_dgt = tf.placeholder(tf.float32, shape=[None, None, None, 1])
Loss_dr = tf.reduce_mean(tf.square(I_dr - I_dgt))

# TM-Net
I_r = tf.clip_by_value(I_b+tf.stop_gradient(I_dr), 0.0, 1.0)
I_o = unet(I_r, outputChn=1, name='test_TMO', reuse=False)
I_o = tf.nn.sigmoid(I_o)

Y_8bit = (tf.round(I_o*255)) / 255.0
Y_8bit_for_loss = tf.stop_gradient(Y_8bit - I_o) + I_o

# Structural similarity loss
Loss_ss = 0
Loss_ss = Loss_ss + 1e-0 * tf.reduce_mean(1.0 - tf.image.ssim(I, I_o, max_val=1.0))

# Detail preservation loss
dXdx, dXdy = tf.image.image_gradients(I_r)
dYdx, dYdy = tf.image.image_gradients(I_o)

Loss_dp = 0
Loss_dp = Loss_dp - tf.reduce_mean((dXdx*dYdx + dXdy*dYdy))

# Relative thickness loss
# 1. local
parameter_RT = 1e-1
Loss_thickness_1 = tf.reduce_mean(tf.abs(tf.nn.tanh(dXdx * parameter_RT) - tf.nn.tanh(dYdx * parameter_RT)) + tf.abs(tf.nn.tanh(dXdy * parameter_RT) - tf.nn.tanh(dYdy * parameter_RT)))

# 2. (2,2)
Y_avg_2 = tf.nn.avg_pool(I_r, ksize=(1,2,2,1), strides=(1,2,2,1), padding='VALID')
Y_8bit_avg_2 = tf.nn.avg_pool(I_o, ksize=(1,2,2,1), strides=(1,2,2,1), padding='VALID')
dXdx2, dXdy2 = tf.image.image_gradients(Y_avg_2)
dYdx2, dYdy2 = tf.image.image_gradients(Y_8bit_avg_2)
Loss_thickness_2 = tf.reduce_mean(tf.abs(tf.nn.tanh(dXdx2 * parameter_RT) - tf.nn.tanh(dYdx2 * parameter_RT)) + tf.abs(tf.nn.tanh(dXdy2 * parameter_RT) - tf.nn.tanh(dYdy2 * parameter_RT)))

# 3. (4,4)
Y_avg_4 = tf.nn.avg_pool(I_r, ksize=(1,4,4,1), strides=(1,4,4,1), padding='VALID')
Y_8bit_avg_4 = tf.nn.avg_pool(I_o, ksize=(1,4,4,1), strides=(1,4,4,1), padding='VALID')
dXdx4, dXdy4 = tf.image.image_gradients(Y_avg_4)
dYdx4, dYdy4 = tf.image.image_gradients(Y_8bit_avg_4)
Loss_thickness_4 = tf.reduce_mean(tf.abs(tf.nn.tanh(dXdx4 * parameter_RT) - tf.nn.tanh(dYdx4 * parameter_RT)) + tf.abs(tf.nn.tanh(dXdy4 * parameter_RT) - tf.nn.tanh(dYdy4 * parameter_RT)))

# 4. (8,8)
Y_avg_8 = tf.nn.avg_pool(I_r, ksize=(1,8,8,1), strides=(1,8,8,1), padding='VALID')
Y_8bit_avg_8 = tf.nn.avg_pool(I_o, ksize=(1,8,8,1), strides=(1,8,8,1), padding='VALID')
dXdx8, dXdy8 = tf.image.image_gradients(Y_avg_8)
dYdx8, dYdy8 = tf.image.image_gradients(Y_8bit_avg_8)
Loss_thickness_8 = tf.reduce_mean(tf.abs(tf.nn.tanh(dXdx8 * parameter_RT) - tf.nn.tanh(dYdx8 * parameter_RT)) + tf.abs(tf.nn.tanh(dXdy8 * parameter_RT) - tf.nn.tanh(dYdy8 * parameter_RT)))

Loss_thickness = Loss_thickness_1 + Loss_thickness_2 + Loss_thickness_4 + Loss_thickness_8

# 5. (16,16)
Y_avg_16 = tf.nn.avg_pool(I_r, ksize=(1,16,16,1), strides=(1,16,16,1), padding='VALID')
Y_8bit_avg_16 = tf.nn.avg_pool(I_o, ksize=(1,16,16,1), strides=(1,16,16,1), padding='VALID')
dXdx16, dXdy16 = tf.image.image_gradients(Y_avg_16)
dYdx16, dYdy16 = tf.image.image_gradients(Y_8bit_avg_16)
Loss_thickness_16 = tf.reduce_mean(tf.abs(tf.nn.tanh(dXdx16 * parameter_RT) - tf.nn.tanh(dYdx16 * parameter_RT)) + tf.abs(tf.nn.tanh(dXdy16 * parameter_RT) - tf.nn.tanh(dYdy16 * parameter_RT)))

# 6. (32,32)
Y_avg_32 = tf.nn.avg_pool(I_r, ksize=(1,32,32,1), strides=(1,32,32,1), padding='VALID')
Y_8bit_avg_32 = tf.nn.avg_pool(I_o, ksize=(1,32,32,1), strides=(1,32,32,1), padding='VALID')
dXdx32, dXdy32 = tf.image.image_gradients(Y_avg_32)
dYdx32, dYdy32 = tf.image.image_gradients(Y_8bit_avg_32)
Loss_thickness_32 = tf.reduce_mean(tf.abs(tf.nn.tanh(dXdx32 * parameter_RT) - tf.nn.tanh(dYdx32 * parameter_RT)) + tf.abs(tf.nn.tanh(dXdy32 * parameter_RT) - tf.nn.tanh(dYdy32 * parameter_RT)))

# 7. (64,64)
Y_avg_64 = tf.nn.avg_pool(I_r, ksize=(1,64,64,1), strides=(1,64,64,1), padding='VALID')
Y_8bit_avg_64 = tf.nn.avg_pool(I_o, ksize=(1,64,64,1), strides=(1,64,64,1), padding='VALID')
dXdx64, dXdy64 = tf.image.image_gradients(Y_avg_64)
dYdx64, dYdy64 = tf.image.image_gradients(Y_8bit_avg_64)
Loss_thickness_64 = tf.reduce_mean(tf.abs(tf.nn.tanh(dXdx64 * parameter_RT) - tf.nn.tanh(dYdx64 * parameter_RT)) + tf.abs(tf.nn.tanh(dXdy64 * parameter_RT) - tf.nn.tanh(dYdy64 * parameter_RT)))

Loss_rt = Loss_thickness_1 + Loss_thickness_2 + Loss_thickness_4 + Loss_thickness_8 + Loss_thickness_16 + Loss_thickness_32 + Loss_thickness_64

# Total loss
Loss = 0
Loss_ss = Loss_ss * 1.0e+0
Loss = Loss + Loss_ss
Loss_dp = Loss_dp * 3.0e+2
Loss = Loss + Loss_dp
Loss_rt = Loss_rt * 1.0e+2
Loss = Loss + Loss_rt

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

var = [v for v in tf.global_variables() if v.name.startswith('test_Dt_rstr')]
var2 = [v for v in tf.global_variables() if v.name.startswith('test_TMO')]

with tf.control_dependencies(update_ops):
    optimize = tf.train.AdamOptimizer(learning_rate=LR).minimize(Loss_dr, var_list=var)
    optimize2 = tf.train.AdamOptimizer(learning_rate=LR).minimize(Loss, var_list=var2)

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.35
config.gpu_options.allow_growth = False
sess = tf.Session(config=config)
init = tf.global_variables_initializer()
sess.run(init)
saver = tf.train.Saver()

lr_init = 1e-5
lr = lr_init

start_time = time.time()

test_path = 'D:/Research/200207_tone_mapping/3.data/testset0821/'
fileList = os.listdir(test_path)
test_list = [file for file in fileList if file.endswith("png")]

test_num = len(test_list)

train_path = 'D:/Research/191024_xray_data/200429_train/'
fileList = os.listdir(train_path)
train_list = [file for file in fileList if file.endswith("png")]
gt_list = [file for file in fileList if file.endswith("pickle")]

train_num = len(train_list)

rst_path = 'D:/Research/191024_xray_data/experiment/imgs/200819_Lss_1_5Ldp_Lrt/'

# # weight restore
# print('Weight Restoring.....')
# Restore = tf.train.import_meta_graph('D:/Research/191024_xray_data/experiment/models/200819_Lss_1_5Ldp_Lrt/iter 200000/Train_200000.meta')
# Restore.restore(sess, tf.train.latest_checkpoint('D:/Research/191024_xray_data/experiment/models/200819_Lss_1_5Ldp_Lrt/iter 200000/'))
# print('Weight Restoring Finish!')

for iter_count in range(1, 200001):

    if iter_count > 100000:
        lr = ((200000 - iter_count) / 100000) * lr_init

    IMG = np.zeros((batch_size, Height, Width))
    D_input = np.zeros((batch_size, Height, Width))
    B_input = np.zeros((batch_size, Height, Width))
    GT_input = np.zeros((batch_size, Height, Width))

    for batch in range(batch_size):
        # img
        img = cv.imread(train_path + train_list[(iter_count * batch_size + batch) % train_num], -1)
        with open('%s%s' % (train_path, gt_list[(iter_count * batch_size + batch) % train_num]), 'rb') as f:
            gt_train = pickle.load(f)
        img, gt_train = randomCropHL(img, gt_train, height=Height, width=Width)
        IMG[batch] = img / 65535.0
        GT_input[batch] = gt_train
        B_input[batch], D_input[batch] = decomposition_NotLog(IMG[batch])

    IMG = IMG.reshape((batch_size, Height, Width, 1)).astype('float32')
    B_input = B_input.reshape((batch_size, Height, Width, 1)).astype('float32')
    D_input = D_input.reshape((batch_size, Height, Width, 1)).astype('float32')
    GT_input = GT_input.reshape((batch_size, Height, Width, 1)).astype('float32')
    _, Loss_dr_train = sess.run([optimize, Loss_dr], feed_dict={I: IMG, I_d: D_input, I_b: B_input, I_dgt: GT_input, LR: lr})
    _, Loss1, Loss_rt_train, Loss_ss_train, Loss_dp_train = sess.run([optimize2, Loss, Loss_rt, Loss_ss, Loss_dp],
                        feed_dict={I: IMG, I_d: D_input, I_b: B_input, I_dgt: GT_input})
    if iter_count % 50 == 0:
        consume_time = time.time() - start_time
        print(iter_count)
        print('Loss: %.5f  Loss_thickness: %.5f  Loss_dataterm: %.5f  Loss_detail: %.5f  computation: %.5f ' % (Loss1, Loss_rt_train, Loss_ss_train, Loss_dp_train, consume_time))
        start_time = time.time()

    if iter_count == 1 or (iter_count % 10000 == 0) or (iter_count < 10000 and iter_count % 1000 == 0):
        if (iter_count % 10000 == 0):
            print('SAVING MODEL')
            Temp = ('D:/Research/191024_xray_data/experiment/models/200525_/iter %s/' % iter_count)

            if not os.path.exists(Temp):
                os.makedirs(Temp)

            SaveName = (Temp + 'Train_%s' % (iter_count))
            saver.save(sess, SaveName)

        for i in range(test_num):
            img_test = np.zeros((Height, Width))
            img_test = cv.imread(test_path + test_list[i], -1)

            h, w = img_test.shape
            img_test = (img_test / 65535.0)
            IMG_test = img_test.reshape((1,h,w,1))
            B_test, D_test = decomposition_NotLog(img_test)
            B_test = B_test.reshape((1, h, w, 1))
            D_test = D_test.reshape((1, h, w, 1))

            I_o_test, Y_8bit_test, I_r_test, I_test, I_dr_test = sess.run(
                [I_o, Y_8bit, tf.clip_by_value(I_r, 0.0, 1.0), I, I_dr], feed_dict={I: IMG_test, I_d: D_test, I_b: B_test})


            bias = 300

            Y_8bit_test = (np.round(Y_8bit_test[0].reshape((h, w)) * 255.0)).astype('uint8')
            rst_path2 = 'D:/Research/200207_tone_mapping/3.data/results/results_od/'
            if not os.path.exists(rst_path2 + 'iter %s/' % iter_count):
                os.makedirs(rst_path2 + 'iter %s/' % iter_count)
            cv.imwrite(rst_path2 + 'iter %s/' % iter_count + test_list[i][:-3] + 'png', Y_8bit_test)
