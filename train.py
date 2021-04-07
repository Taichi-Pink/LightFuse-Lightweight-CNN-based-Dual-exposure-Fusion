import tensorflow as tf
tf.enable_eager_execution()
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1,2"
import numpy as np
from utils import *
from load_data import load_data
import matplotlib.pyplot as plt
import math, cv2, glob, time
import imageio as io

patch_size = 256
bs = 1
def train(model, indexx):
  dataset = load_data(bs, patch_size) 
  loss_history = []
  print(model.summary())
  #model.load_weights('./model1_3.h5')
  
  for epoch in range(0,2):
      loss1 = []
      step=0
      
      for ldr, hdr in dataset:
          ldr = tf.image.resize_images(tf.cast(ldr, tf.float32), (256,256))*2.0-1.0
          hdr = tf.image.resize_images(tf.cast(hdr, tf.float32), (256,256))*2.0-1.0
          
          c1 = model.train_on_batch(ldr, hdr)
          loss1.append(c1)       
          #print('Epoch:%d, step:%d, model1:%f'%(epoch, step, c1))
          step = step+1                
      model.save_weights('./model'+str(indexx)+'_'+ str(epoch) +'.h5')
      
      loss_mean = np.mean(loss1)
      loss_history.append(loss_mean)
      print(loss_mean)
      
  fig = plt.figure()
  plt.plot(loss_history)
  fig.savefig('temp'+str(indexx)+'.png', dpi=fig.dpi)
  
                                             
if __name__ == "__main__":
  for i in range(1,2):
    if i == 1:
      model = model_test1(256,256,6)
    elif i == 2:
      model = model_test2(256,256,6)
      
    train(model, i)
   

      
    