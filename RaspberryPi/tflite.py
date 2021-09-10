import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
#import matplotlib.pyplot as plt
import cv2
import numpy as np
import imageio as io
ch = 6
w  = 2736
h  = 1824

def supervised_model(w,h,c, k1=1, k2=3, s1=1, s2=2, f1=256, f2=64, pad="same", act='relu'):
  input_ = keras.Input((w, h, c))
  out_1 = layers.Conv2D(f1, (k1, k1), strides=(s1, s1), padding=pad, activation=act)(input_)
  out_1 = layers.Conv2D(f2, (k1, k1), strides=(s1, s1), padding=pad, activation=act)(out_1)
  out_1 = layers.Conv2D(c//2, (k1, k1), strides=(s1, s1), padding=pad, activation=act)(out_1)
  
  out_2 = layers.DepthwiseConv2D((k2,k2), strides = (s2, s2), padding=pad)(input_)
  out_2 = layers.DepthwiseConv2D((k2,k2), strides = (s2, s2), padding=pad)(out_2)
  out_2 = layers.SeparableConv2D(c//2, (k2,k2), strides = (s2, s2), padding=pad)(out_2)
  
  out_2 = layers.UpSampling2D(size=(2, 2))(out_2)  
  out_2 = layers.UpSampling2D(size=(2, 2))(out_2) 
  out_2 = layers.UpSampling2D(size=(2, 2))(out_2)
  
  out = layers.Add()([out_1, out_2])
  output_ = layers.Activation('tanh')(out) 
  model = Model(inputs=input_, outputs=output_)
  op = keras.optimizers.Adam(learning_rate=0.001)
  model.compile(optimizer=op, loss='mse')
  return model
  
# tf 2.x  
def converter_tflite(model):

    print(model.summary())
    model.load_weights('../addtion_merge.h5')

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_dog_model = converter.convert()
    open("lite_model.tflite", "wb").write(tflite_dog_model)

def norm_0_to_1(img):
    img = np.float32(img)
    img_flat = img.flatten()
    max_value = np.max(img_flat)
    min_value = np.min(img_flat)
    new_img = (img - min_value) * 1 / (max_value - min_value)
    return new_img

#tf 1.15.0
def test_tflite():
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path="lite_model.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # load data
    p_over = r'./over_exposed.JPG'
    over_exp = cv2.imread(p_over)
    over_exp = over_exp[:, :, ::-1]

    p_under = r'./under_exposed.JPG'
    under_exp = cv2.imread(p_under)
    under_exp = under_exp[:, :, ::-1]

    over_exp  = cv2.resize(over_exp, (w, h))
    under_exp = cv2.resize(under_exp, (w, h))
    over_exp  = norm_0_to_1(over_exp)
    under_exp = norm_0_to_1(under_exp)
    
    img = np.concatenate([under_exp, over_exp], axis=2)
    img = np.expand_dims(img, axis=0)          
    img = img*2.0-1.0   

    # Test model on random input data.
    interpreter.set_tensor(input_details[0]['index'], img)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])

    ldr  = np.squeeze(output_data)
    ldr_ = ((ldr+1.0)/2.0)
    tem  = ldr_*255.0
    tem  = tem.astype(np.uint8)
    io.imwrite('test.png', tem)



if __name__ == "__main__":
    '''Convert to Tensorflow Lite Model'''
    #model = supervised_model(h, w, ch)
    #converter_tflite(model)
    '''Run Custom Tensorflow Lite Model on Raspberry Pi'''
    test_tflite()

