import tensorflow as tf
import cv2, glob, os
import numpy as np
import matplotlib.pyplot as plt
import math
import imageio as io
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

scene_no = 50
data_dir = '/home/jieyang/Ziyi/Dataset/Dataset_Part1/Dataset_Part2/'


for no in range(1,scene_no):
  file_p = data_dir+ str(no) +'/'
  file_list = sorted(glob.glob(file_p + '/*.{}'.format('JPG')))
  length_images = len(file_list)
  results_p = '/home/jieyang/Ziyi/Dataset/Dataset_Part1/Dataset_Part2_cp/'+ str(no) +'/'
  
  print('length_images:',length_images)
  data = []
  for i in range(length_images):
      ldr_img = cv2.imread(file_list[i])
      avg_color_per_row = np.average(ldr_img, axis=2)
      avg_color = np.average(avg_color_per_row, axis=0)
      avg_color = np.average(avg_color, axis=0)
      print("average:", avg_color, "index:", i)
      data.append([i, avg_color])
  
  sorted_by_second = sorted(data, key=lambda tup: tup[1]) #sort ldr images according to the average color
  
  for i in range(length_images):
       plt.subplot(5, max(length_images//4+1,1), i+1).set_title(str(i+1))
       ind = sorted_by_second[i][0]
       p = file_list[ind]
       im = cv2.imread(p)
       plt.imshow(im)
   plt.show()
  
  for i in range(length_images):
    ev = cv2.imread(file_list[sorted_by_second[i][0]])
    im_p = results_p+str(i+1)+'.png'
    if not os.path.exists(results_p):
       os.makedirs(results_p)
    cv2.imwrite(im_p, ev)
   
    #Convert 2 images to numpy arrays and compare pixel by pixel
    ev_cp = cv2.imread(im_p)
    flag = (ev == ev_cp).all()
    print(flag)
    
    
    
  
 
 