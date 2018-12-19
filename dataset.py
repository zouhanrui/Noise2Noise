# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import tensorflow as tf
import sys

def parse_tfrecord_tf(record):
    features = tf.parse_single_example(record, features={
        'shape': tf.FixedLenFeature([3], tf.int64),
        'data': tf.FixedLenFeature([], tf.string)})
    data = tf.decode_raw(features['data'], tf.uint8)
    return tf.reshape(data, features['shape'])

# [c,h,w] -> [h,w,c]
def chw_to_hwc(x):
    return tf.transpose(x, perm=[1, 2, 0])

# [h,w,c] -> [c,h,w]
def hwc_to_chw(x):
    return tf.transpose(x, perm=[2, 0, 1])

def resize_small_image(x):
    shape = tf.shape(x)
    return tf.cond(
        tf.logical_or(
            tf.less(shape[2], 256),
            tf.less(shape[1], 256)
        ),
        true_fn=lambda: hwc_to_chw(tf.image.resize_images(chw_to_hwc(x), size=[256,256], method=tf.image.ResizeMethod.BICUBIC)),
        false_fn=lambda: tf.cast(x, tf.float32)
     )

def random_crop_noised_clean(x, add_noise):
    cropped = tf.random_crop(resize_small_image(x), size=[3, 256, 256]) / 255.0 - 0.5
    return (add_poisson_noise_tf(cropped), shuffle(cropped), cropped)

def add_poisson_noise_tf(x):
    chi_rng = tf.random_uniform(shape=[1, 1, 1], minval=0.001, maxval=50.0)
    return tf.random_poisson(chi_rng*(x+0.5), shape=[])/chi_rng - 0.5

def add_gaussian_noise_tf(x):
        shape = tf.shape(x)
        rng_stddev = tf.random_uniform(shape=[1, 1, 1], minval=0.0/255.0, maxval=50.0/255.0)
        return x + tf.random_normal(shape) * rng_stddev
    
def add_gamma_noise(x):
    a = tf.random_uniform(shape=[1, 1, 1], minval=0.001, maxval=50)
    #beta = tf.random_uniform(shape=[1, 1, 1], minval=0.001, maxval=10)
    return tf.random_gamma(shape=[], alpha = a*(x+0.5))/a - 0.5

def shuffle(x):
    shape = tf.shape(x)
    a = tf.reshape(x, [-1, 1])
    tf.random_shuffle(a)
    x_shuffled = tf.reshape(a, shape)
    return x_shuffled




def create_dataset(train_tfrecords, minibatch_size, add_noise):
    print ('Setting up dataset source from', train_tfrecords)
    buffer_mb   = 256
    num_threads = 2
    dset = tf.data.TFRecordDataset(train_tfrecords, compression_type='', buffer_size=buffer_mb<<20)
    
    dset = dset.repeat()
    buf_size = 1000
    
    
    dset = dset.prefetch(buf_size)
    

    #This transformation applies parse_tfrecord_tf to each element of this dataset, 
    #and returns a new dataset containing the transformed elements, 
    #in the same order as they appeared in the input.
    dset = dset.map(parse_tfrecord_tf, num_parallel_calls=num_threads)
    
    dset = dset.shuffle(buffer_size=buf_size)
    
    #Transformation: 1.crop all images to shape(3,256,256), 2.DATA NOISE AUGMENTATION: return noised data, noised data, clean data.
    dset = dset.map(lambda x: random_crop_noised_clean(x, add_noise))
    #<MapDataset shapes: ((3, 256, 256), (3, 256, 256), (3, 256, 256)), types: (tf.float32, tf.float32, tf.float32)>
    
    #Combines consecutive elements of this dataset into batches.
    dset = dset.batch(minibatch_size)
    #<BatchDataset shapes: ((?, 3, 256, 256), (?, 3, 256, 256), (?, 3, 256, 256)), types: (tf.float32, tf.float32, tf.float32)>
    
    it = dset.make_one_shot_iterator()
    print()
    print('create dataset:========= Created an Iterator for enumerating the data of this dataset.')
    """
    while True:
        try:
        # get the next item
            element = it.get_next()
        # do something with element
            print(element)
            print('*************line*********')
        except StopIteration:
        # if StopIteration is raised, break from loop
            break
    """
   
    return it

