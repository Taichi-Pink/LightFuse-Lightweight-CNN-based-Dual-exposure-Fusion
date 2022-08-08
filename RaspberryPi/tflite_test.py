import tensorflow as tf
#tf.enable_eager_execution()
from tensorflow import keras
from tensorflow.keras import layers, Model
# import matplotlib.pyplot as plt
import cv2, time
import numpy as np
import imageio as io
ch = 6
w = 336#1344#256
h = 224#896#128

def norm_0_to_1(img):
    img = np.float32(img)
    img_flat = img.flatten()
    max_value = np.max(img_flat)
    min_value = np.min(img_flat)
    new_img = (img - min_value) * 1 / (max_value - min_value)
    return new_img

def test_tflite():
    # Load TFLite model and allocate tensors.
    path_p = 'test_cust_s56.png'#change
    interpreter = tf.lite.Interpreter(model_path="lite_model_cust_336x224_no_tile.tflite")#change
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # load data
    p_over = '1.jpg'#'./over_exp256x128.png'
    over_exp = cv2.imread(p_over)
    over_exp = over_exp[:, :, ::-1]

    p_under = '2.jpg'#'./under_exp256x128.png'
    under_exp = cv2.imread(p_under)
    under_exp = under_exp[:, :, ::-1]

    over_exp = cv2.resize(over_exp, (w, h))
    under_exp = cv2.resize(under_exp, (w, h))
    over_exp = norm_0_to_1(over_exp)
    under_exp = norm_0_to_1(under_exp)

    img = np.concatenate([under_exp, over_exp], axis=2)
    img = np.expand_dims(img, axis=0)
    img = img * 2.0 - 1.0
    
    t_start = time.time()
    print('start:')
    for i in range(60):
    
      # Test model on random input data.
      interpreter.set_tensor(input_details[0]['index'], img)
      interpreter.invoke()
  
      # The function `get_tensor()` returns a copy of the tensor data.
      # Use `tensor()` in order to get a pointer to the tensor. 
      output_data = interpreter.get_tensor(output_details[0]['index'])
    
    print('time:')
    print((time.time()-t_start)/60.0)
    
    #ldr = np.squeeze(output_data)
    #ldr_ = ((ldr + 1.0) / 2.0)
    #tem = ldr_ * 255.0
    #tem = tem.astype(np.uint8)
    #io.imwrite(path_p, tem)


if __name__ == "__main__":

    '''Run Custom Tensorflow Lite Model on Raspberry Pi'''
    test_tflite()

