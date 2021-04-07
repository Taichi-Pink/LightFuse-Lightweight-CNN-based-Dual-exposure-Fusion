import tensorflow as tf
tf.enable_eager_execution()
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3,2"
from tensorflow import keras
import numpy as np
from tensorflow.keras import layers, Model
from gen_dis import *
from utils import *
from load_data import load_data
import matplotlib.pyplot as plt
import math
import imageio as io

def feature_map_train(generator):
  dataset = load_data()
  ixs = [4, 10, 11]
  outputs = [generator.layers[i].output for i in ixs]
  model = keras.Model(inputs=generator.inputs, outputs=outputs)
  
  for img , _ in dataset: 
    index = 0 
    feature_maps = model.predict(img)
      # plot the output from each block
    for fmap in feature_maps:
     	# plot all 64 maps in an 8x8 squares
      ix = 1
      
      square = fmap.shape[-1]
      square_sqrt = int(math.sqrt(square))
      square_div = int(square/square_sqrt)
       
      for _ in range(square_sqrt):
        for _ in range(square_div):
  		    # specify subplot and turn of axis
          ax = plt.subplot(square_sqrt, square_div, ix)
          ax.set_title('layer_'+str(ixs[index]))
   			  # plot filter channel in grayscale
          plt.imshow(fmap[0, :, :, ix-1])
          ix += 1
      index = index + 1
      plt.show()

def filters(model):  
  for layer in model.layers: 
      for weight, weights_numpy_array in zip(layer.weights, layer.get_weights()):
          if weights_numpy_array.ndim == 4:
            print(weight.name)
            square = weights_numpy_array.shape[-1]
            square_sqrt = int(math.sqrt(square))
            square_div = int(square/square_sqrt)
            ix = 1
            for _ in range(square_sqrt):
              for _ in range(square_div):
  		    
                ax = plt.subplot(square_sqrt, square_div, ix)
                ax.set_title(weight.name)
                plt.imshow(weights_numpy_array[:, :,0, ix-1])
                ix += 1
           
            plt.show()
h = 512
w = 320
def feature_map_test(generator):
  ixs = [4]
  outputs = [generator.layers[i].output for i in ixs]
  model = keras.Model(inputs=generator.inputs, outputs=outputs)
  
  cur_path = '/home/jieyang/Ziyi/Dataset/Dataset_Part1/Dataset_Part1/100/'
  under_index = 1
  over_index = 7
  
  over_exp = cv2.imread(os.path.join(cur_path, str(over_index)+ '.JPG'))
  over_exp = over_exp[:, :, ::-1]
  under_exp = cv2.imread(os.path.join(cur_path, str(under_index)+ '.JPG'))
  under_exp = under_exp[:, :, ::-1]
  over_exp = cv2.resize(over_exp, (h, w))
  under_exp = cv2.resize(under_exp, (h, w))
  over_exp = norm_0_to_1(over_exp)
  under_exp = norm_0_to_1(under_exp)
    
  img = np.concatenate([under_exp, over_exp], axis=2)
  img = np.expand_dims(img, axis=0)          
  img = img*2.0-1.0
    
  feature_maps = model.predict(img)
   
  for fmap in feature_maps:
    square = fmap.shape[-1]
    for i in range(square):
      plt.imshow(np.squeeze(fmap[0, :, :, i]))
      plt.show()

filters_p = './filters_check/'
def filters_light(model):  
  for layer in model.layers: 
      for weight, weights_numpy_array in zip(layer.weights, layer.get_weights()):
          if weights_numpy_array.ndim == 4:
            name = os.path.split(weight.name)[-1]
            print(name)
            if weights_numpy_array.shape[0]==3:
              square = weights_numpy_array.shape[-1]
              square_sqrt = int(math.sqrt(square))
              square_div = int(square/square_sqrt)
              ix = 1
              for _ in range(square_sqrt):
                for _ in range(square_div):
            
                  ax = plt.subplot(square_sqrt, square_div, ix)
                  ax.set_title(weight.name)
                  plt.imshow(weights_numpy_array[:, :,0, ix-1])
                  ix += 1
              
              #plt.show()
              
            else:
              fig = plt.figure()
              plt.imshow(np.squeeze(weights_numpy_array[:, :, :, :]))
              #plt.show()
              fig.savefig(filters_p+name+'.png', dpi=fig.dpi)
                  
              
if __name__ == "__main__":
  model = model_test1(320,512,6)
  model.load_weights('./1blocks(25)/model1_80.h5')
  feature_map_test(model)
  #filters_light(model)
  