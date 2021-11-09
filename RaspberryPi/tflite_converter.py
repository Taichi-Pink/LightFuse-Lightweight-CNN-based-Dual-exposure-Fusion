import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, initializers
import numpy as np

ch = 6
w  = 336
h  = 224

def supervised_model(w,h,c, k1=1, k2=3, s1=1, s2=2, f1=256, f2=64, pad="same", act='relu'):
  input_ = keras.Input((w, h, c))
  conv_ = custconv2d()
  out_1 = conv_(input_)
  
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

class custconv2d(keras.layers.Layer):
  def __init__(self, name=None, **kwargs):
    super(custconv2d, self).__init__(name=name)
    self.w1 = self.add_weight(shape=(1,1,6,256), initializer="random_normal", trainable=True, name='w1')
    self.b1 = self.add_weight(shape=(256,), initializer="zeros", trainable=True, name='b1')
    self.w2 = self.add_weight(shape=(1,1,256,64), initializer="random_normal", trainable=True, name='w2')
    self.b2 = self.add_weight(shape=(64,), initializer="zeros", trainable=True, name='b2')
    self.w3 = self.add_weight(shape=(1,1,64,3), initializer="random_normal", trainable=True, name='w3')
    self.b3 = self.add_weight(shape=(3,), initializer="zeros", trainable=True, name='b3')
  
  def call(self, inputs):
    n, h, w, c = inputs.shape.as_list()
    stride_ = 16

    hi = []
    for hight in range(0, h-stride_+1, stride_):
      wi = []
      for width in range(0, w-stride_+1, stride_):       
          inputs_ = inputs[:, hight:hight+stride_, width:width+stride_, :]
          temp0 = tf.nn.relu(tf.matmul(inputs_, self.w1) + self.b1)
          temp1 = tf.nn.relu(tf.matmul(temp0, self.w2) + self.b2)
          temp2 = tf.nn.relu(tf.matmul(temp1, self.w3) + self.b3)
          
          wi.append(temp2)
      length = len(wi)
      fist_item = tf.concat([wi[0], wi[1]], axis=2)
      for index in range(2, length):
          fist_item = tf.concat([fist_item, wi[index]], axis=2)
      hi.append(fist_item)
    
    length = len(hi)
    fist_item0 = tf.concat([hi[0], hi[1]], axis=1)
    for index in range(2, length):
        fist_item0 = tf.concat([fist_item0, hi[index]], axis=1)
    
    return fist_item0
  
# conda env: tf 2.0
def converter_tflite(model):

    print(model.summary())
    model.load_weights('addtion_merge.h5')
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_dog_model = converter.convert()
    
    open("lite_model_cust_336x224_s16.tflite", "wb").write(tflite_dog_model)


if __name__ == "__main__":
    '''Convert to Tensorflow Lite Model'''
    model = supervised_model(h, w, ch)
    converter_tflite(model)

