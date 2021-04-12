import tensorflow as tf
tf.enable_eager_execution()
import numpy as np
import matplotlib.pyplot as plt
import imageio as io
import math, os
from tensorflow.keras import layers, Model
from tensorflow import keras
from load_data import load_data
from utils import *

def feature_map_train(generator):
  dataset = load_data()
  ixs     = [4, 10, 11]
  outputs = [generator.layers[i].output for i in ixs]
  model   = keras.Model(inputs=generator.inputs, outputs=outputs)
  
  for img , _ in dataset: 
    index = 0 
    feature_maps = model.predict(img)
    ## plot the output from each block
    for fmap in feature_maps:
     	## plot all 64 maps in an 8x8 squares
      ix = 1
      square      = fmap.shape[-1]
      square_sqrt = int(math.sqrt(square))
      square_div  = int(square/square_sqrt)
       
      for _ in range(square_sqrt):
        for _ in range(square_div):
  		    ## specify subplot and turn of axis
          ax = plt.subplot(square_sqrt, square_div, ix)
          ax.set_title('layer_'+str(ixs[index]))
   			  ## plot filter channel in grayscale
          plt.imshow(fmap[0, :, :, ix-1])
          ix += 1
      index = index + 1
      plt.show()

h = 512
w = 320
def feature_map_test(generator):
  ixs = [4]
  outputs = [generator.layers[i].output for i in ixs]
  model   = keras.Model(inputs=generator.inputs, outputs=outputs)
  
  cur_path    = './Dataset/1/'
  under_index = 1
  over_index  = 6
  
  over_exp  = cv2.imread(os.path.join(cur_path, str(over_index)+ '.JPG'))
  over_exp  = over_exp[:, :, ::-1]
  under_exp = cv2.imread(os.path.join(cur_path, str(under_index)+ '.JPG'))
  under_exp = under_exp[:, :, ::-1]
  over_exp  = cv2.resize(over_exp, (h, w))
  under_exp = cv2.resize(under_exp, (h, w))
  over_exp  = norm_0_to_1(over_exp)
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


def filters_lightfuse_model(model):
  filters_p = './filters_check/'
  if not os.path.exists(filters_p):
    os.makedirs(filters_p)
  
  for layer in model.layers: 
      for weight, weights_numpy_array in zip(layer.weights, layer.get_weights()):
          if weights_numpy_array.ndim == 4:
            name = os.path.split(weight.name)[-1]
            print(name)
            if weights_numpy_array.shape[0]==3:
              square      = weights_numpy_array.shape[-1]
              square_sqrt = int(math.sqrt(square))
              square_div  = int(square/square_sqrt)
              ix = 1
              for _ in range(square_sqrt):
                for _ in range(square_div):
            
                  ax = plt.subplot(square_sqrt, square_div, ix)
                  ax.set_title(weight.name)
                  plt.imshow(weights_numpy_array[:, :,0, ix-1])
                  ix += 1
              
              plt.show()
            else:
              fig = plt.figure()
              plt.imshow(np.squeeze(weights_numpy_array[:, :, :, :]))
              plt.show()
              fig.savefig(filters_p + name + '.png', dpi=fig.dpi)
                  
              
if __name__ == "__main__":
  model = supervised_model(320,512,6)
  model.load_weights('addtion_merge.h5')
  feature_map_test(model)
  #filters_light(model)
  
