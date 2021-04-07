import tensorflow as tf
tf.enable_eager_execution()
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
from tensorflow import keras
import numpy as np
from tensorflow.keras import layers, Model
from utils import *
from load_data import load_data
import matplotlib.pyplot as plt
import math, cv2, glob, time
import imageio as io
from tensorflow.keras import backend as K
from tensorflow.keras.applications.vgg16 import VGG16
import xlrd

p = 'exposure_value_part1.xls'
wb = xlrd.open_workbook(p)
sheet1 = wb.sheet_by_index(0)

patch_size = 256
scene_no = 230
bs = 1
results_p = './result_final/'
result_path = './log/'

if not os.path.exists(results_p):
  os.makedirs(results_p)
if not os.path.exists(result_path):
  os.makedirs(result_path)

def build(model, w, h, c):
  input_ = keras.Input((w, h, c))
  x1 = layers.Conv2D(256, (1, 1), strides=(1, 1), padding="same", activation='relu')(input_)
  x1 = layers.Conv2D(64, (1, 1), strides=(1, 1), padding="same", activation='relu')(x1)
  x1 = layers.Conv2D(3, (1, 1), strides=(1, 1), padding="same", activation='relu')(x1)
  
  x2 = layers.DepthwiseConv2D((3,3), strides = (2, 2), padding="same")(input_)
  x2 = layers.DepthwiseConv2D((3,3), strides = (2, 2), padding="same")(x2)
  x2 = layers.SeparableConv2D(3, (3,3), strides = (2, 2), padding="same")(x2)
  
  x2 = layers.UpSampling2D(size=(2, 2))(x2)  
  x2 = layers.UpSampling2D(size=(2, 2))(x2) 
  x2 = layers.UpSampling2D(size=(2, 2))(x2)
  
  x = layers.Add()([x1, x2])
  output_ = layers.Activation('tanh')(x) 
  model_ = Model(inputs=input_, outputs=output_)
  
  for i in range(1, len(model.layers)):
    model_.layers[i+1].set_weights(model.layers[i].get_weights())
  
  return model_

def test(path, model): 
  model.load_weights(path)
  psnr = []
  ssim = []

  data_dir = '/home/jieyang/Ziyi/Dataset/Dataset_Part1/New/test/'
  scene_dirs = [scene_dir for scene_dir in os.listdir(data_dir) if scene_dir!="Label"]
  scene_dirs = sorted(scene_dirs, key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))
  num_scenes = len(scene_dirs)

  h = 2736
  w = 1824
  t_total = 0.0
  for index in range(1):
    cur_path = scene_dirs[index]
    if cur_path=="Label":
      continue
    
    cur_path = os.path.join(data_dir, cur_path)
    
    over_exp = cv2.imread(os.path.join(cur_path, '4.JPG'))
    over_exp = over_exp[:, :, ::-1]
    over_exp = cv2.resize(over_exp, (h, w))
    over_exp = norm_0_to_1(over_exp)*2.0-1.0
    
    under_exp = cv2.imread(os.path.join(cur_path, '3.JPG'))
    under_exp = under_exp[:, :, ::-1]
    under_exp = cv2.resize(under_exp, (h, w))
    under_exp = norm_0_to_1(under_exp)*2.0-1.0
        
    # finding corresponding ldr image 
    no = int(scene_dirs[index])
    label_p = os.path.join(data_dir, 'Label', str(no) + '.JPG')    
    label = cv2.imread(label_p)
    label = label[:, :, ::-1]
    label = cv2.resize(label, (h, w))   
    label = norm_0_to_1(label) 
    
    img = np.concatenate([under_exp, over_exp], axis=2)
    img = np.expand_dims(img, axis=0)          
    
    t_start = time.time()
    ldr = model.predict(img)
    t_total = t_total + time.time() - t_start
    ldr = np.squeeze(ldr)
    ldr0 = ((ldr+1.)/2.)
    
    tem = ldr0*255.
    tem = tem.astype(np.uint8)
    #if no%12 == 0:
    plt.imshow(tem)
    plt.show()
    
    psnr_result = tf.image.psnr(ldr0, label, max_val=1.0)
    ssim2 = tf.image.ssim(tf.squeeze(ldr0), tf.squeeze(label), max_val=1.0)
    #print('no%d, psnr_result:%f, ssim:%f'%(no, psnr_result, ssim2))
    #io.imwrite(results_p+str(no)+'.png', tem)
    ssim.append(ssim2)
    psnr.append(psnr_result)
    
  print('total_time:', t_total)
  print('average_psnr:', np.mean(psnr))
  print('average_ssim:', np.mean(ssim))

                                               
if __name__ == "__main__":
  model = model_test1(1824,2736,6)   
  path = './1blocks(25)/model1_80.h5'
  test(path, model)

      
     