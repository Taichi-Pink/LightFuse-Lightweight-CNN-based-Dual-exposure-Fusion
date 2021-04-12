import tensorflow as tf
tf.enable_eager_execution()
import matplotlib.pyplot as plt
import math, cv2, glob, time
import numpy as np
import imageio as io
from load_data import load_data
from utils import *
import os

patch_size = 256
epochs     = 80
bs         = 1
ch         = 6

def train(model):
  dataset = load_data(bs, patch_size) 
  loss_history = []
  print(model.summary())
  
  for epoch in range(epochs):
      loss1 = []
      step  = 0
      
      for ldr, hdr in dataset:
          ldr = tf.image.resize_images(tf.cast(ldr, tf.float32), (patch_size, patch_size))*2.0-1.0
          hdr = tf.image.resize_images(tf.cast(hdr, tf.float32), (patch_size, patch_size))*2.0-1.0
          
          l = model.train_on_batch(ldr, hdr)
          loss1.append(l)       
          print('epoch:%d, step:%d, model_loss:%f'%(epoch, step, l))
          step = step+1                
      
      
      loss_mean = np.mean(loss1)
      loss_history.append(loss_mean)  
      
  fig = plt.figure()
  plt.plot(loss_history)
  fig.savefig('loss_history.png', dpi=fig.dpi)
  model.save_weights('./lightfuse_model_weights.h5')
                                             
if __name__ == "__main__":
    model = supervised_model(patch_size, patch_size, ch)
    train(model)
   

      
    
