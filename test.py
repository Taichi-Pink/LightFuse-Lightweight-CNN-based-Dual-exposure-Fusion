import tensorflow as tf
tf.enable_eager_execution()
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1,2,3"
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
results_p = './result_final_test/'
result_path = './log/'

if not os.path.exists(results_p):
  os.makedirs(results_p)
if not os.path.exists(result_path):
  os.makedirs(result_path)

def test(indexx, path, model): 
  model1_ = model
  model1_.load_weights(path)
  psnr = []
  ssim = []

  data_dir = '/home/jieyang/Ziyi/Dataset/Dataset_Part1/New/test/'
  scene_dirs = [scene_dir for scene_dir in os.listdir(data_dir) if scene_dir!="Label"]
  scene_dirs = sorted(scene_dirs, key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))
  num_scenes = len(scene_dirs)

  h = 2736
  w = 1824
  t_total = 0.0
  for index in range(num_scenes):
    cur_path = scene_dirs[index]
    if cur_path=="Label":
      continue
    
    cur_path = os.path.join(data_dir, cur_path)
  
    no = int(scene_dirs[index])
    under_index = sheet1.cell_value(no, 1)
    over_index = sheet1.cell_value(no, 2)
    under_index = int(under_index)
    over_index = int(over_index)
    
    
    over_exp = cv2.imread(os.path.join(cur_path, str(over_index)+ '.JPG'))
    over_exp = over_exp[:, :, ::-1]
    under_exp = cv2.imread(os.path.join(cur_path, str(under_index)+ '.JPG'))
    under_exp = under_exp[:, :, ::-1]
        
    # finding corresponding ldr image 
    label_p = os.path.join(data_dir, 'Label', str(no) + '.JPG')    
    label = cv2.imread(label_p)
    label = label[:, :, ::-1]
    
    over_exp = cv2.resize(over_exp, (h, w))
    under_exp = cv2.resize(under_exp, (h, w))
    label = cv2.resize(label, (h, w))   
    
    ''' bring to [0,1] '''
    over_exp = norm_0_to_1(over_exp)
    under_exp = norm_0_to_1(under_exp)
    label = norm_0_to_1(label) 
    
    img = np.concatenate([under_exp, over_exp], axis=2)
    img = np.expand_dims(img, axis=0)          
    
    img = img*2.0-1.0
    #label = label*2.0-1.0
    
    t_start = time.time()
    ldr = model1_.predict(img)
    
    t_total = t_total + time.time() - t_start
    ldr = np.squeeze(ldr)
    ldr0 = ((ldr+1.)/2.)
    
    tem = ldr0*255.
    tem = tem.astype(np.uint8)
    #if no%12 == 0:
    #  plt.imshow(tem)
    #  plt.show()
    
    psnr_result = tf.image.psnr(ldr0, label, max_val=1.0)
    ssim2 = tf.image.ssim(tf.squeeze(ldr0), tf.squeeze(label), max_val=1.0)
    #print('no%d, psnr_result:%f, ssim:%f'%(no, psnr_result, ssim2))
    io.imwrite(results_p+str(no)+'.png', tem)
    ssim.append(ssim2)
    psnr.append(psnr_result)
    
  print('total_time:', t_total)
  print('average_psnr:', np.mean(psnr))
  print('average_ssim:', np.mean(ssim))

                                               
if __name__ == "__main__":
  for i in range(1,2):  
    if i == 1:
      model = model_test1(1824,2736,6)
    elif i == 2:
      model = model_test2(1824,2736,6)
    elif i == 3:
      model = model_test3(1824,2736,6)
    elif i == 4:
      model = model_test4(1824,2736,6)
      
    for j in range(0,1):
      path = './model'+str(i)+'_'+ str(j) +'.h5'
      test(j, path, model)

      
     