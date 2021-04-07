import tensorflow as tf
import os, glob, datetime
from time import time
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
import pickle
from utils import *
ch = 3
tfrecord_path = '/home/jieyang/Ziyi/Dataset/Dataset_Part1/New/Tfrecord_256_256' #****************************************change!
tfrecord_file_list = glob.glob(tfrecord_path + '/*.{}'.format('tfrecords'))

def load_data(bs, patch_size):
  dataset = tf.data.TFRecordDataset(tfrecord_file_list)
  dataset = dataset.apply(tf.data.experimental.ignore_errors())
  
  feature_description = {
      'in_LDR': tf.io.FixedLenFeature([], tf.string),
      'ref_HDR': tf.io.FixedLenFeature([], tf.string)
  }
  
  def read_and_decode(example_string):
  
      feature_dict = tf.io.parse_single_example(example_string, feature_description)
      image = tf.io.decode_raw(feature_dict['in_LDR'], tf.float32)
      image = tf.cast(image, dtype='float32')
      image = tf.reshape(image, [patch_size, patch_size, ch * 2])
  
      label = tf.io.decode_raw(feature_dict['ref_HDR'], tf.float32)
      label = tf.cast(label, dtype='float32')
      label = tf.reshape(label, [patch_size, patch_size, ch])
  
      ######### distortions #########
      distortions = tf.random_uniform([2], 0, 1.0, dtype=tf.float32)
  
      # flip horizontally
      image = tf.cond(tf.less(distortions[0], 0.5), lambda: tf.image.flip_left_right(image), lambda: image)
      label = tf.cond(tf.less(distortions[0], 0.5), lambda: tf.image.flip_left_right(label), lambda: label)
  
      # rotate
      k = tf.cast(distortions[1] * 4 + 0.5, tf.int32)
      image = tf.image.rot90(image, k)
      label = tf.image.rot90(label, k)
      ######### distortions #########
      return image, label
  
  #dataset = dataset.repeat()
  dataset = dataset.map(read_and_decode)
  dataset = dataset.shuffle(bs*8)
  #dataset = dataset.shuffle(100)
  batch = dataset.batch(batch_size=bs)
  
  return batch
  
if __name__ == "__main__":
    data = load_data()
    iter = data.make_one_shot_iterator()
    el = iter.get_next()
     
    with tf.Session() as sess:
        count = 500
        bs = 20
        while(count):
            index = 0
            img, lab = sess.run(el)
            for i in range(bs//10): 
                plt.figure(1)
                for j in range(10):
                      plt.subplot(4,5,j+1).set_title('count:{} bs:{}'.format(count, i))
                      plt.imshow(np.squeeze(img[index:index+1, :, : , 0:3]))
               
                      plt.subplot(4,5,j+11)
                      plt.imshow(np.squeeze(lab[index:index+1, :, : , :]))
                      
                      index = index + 1
                plt.show()
     
            count = count -1
        
    
    