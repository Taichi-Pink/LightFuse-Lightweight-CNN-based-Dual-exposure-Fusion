import tensorflow as tf
import numpy as np
import cv2, glob, os,math
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import layers, Model
from tensorflow import keras

patch_size = 256
ch         = 3

select_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
vgg16 = VGG16(include_top=False, weights='imagenet', input_shape=(patch_size, patch_size, ch))
vgg16.trainable = False
for l in vgg16.layers:
  l.trainable = False
select = [vgg16.get_layer(name).output for name in select_layers]

model_vgg = Model(inputs=vgg16.input, outputs=select)
model_vgg.trainable = False
    
def vgg_loss(y_true, y_pred):
    
    out_pred = model_vgg(y_pred) 
    out_true = model_vgg(y_true)
    loss_f = 0
    for f_g, f_l in zip(out_pred, out_true):
        loss_f += K.mean(K.abs(f_g - f_l))

    #return loss_f + 0.1*tf.math.reduce_mean(tf.square(y_true - y_pred))
    return loss_f + tf.math.reduce_mean(tf.square(y_true - y_pred))

def psnr_loss(y_true, y_pred):
    #mse = tf.math.reduce_mean(tf.square(y_true - y_pred)) 
    psnr = tf.image.psnr(y_pred, y_true, max_val=2.0)
    return 100-psnr

def ssim_loss(y_true, y_pred):
    ssim = tf.image.ssim(y_true, y_pred, max_val=2.0) 
    return -ssim

def norm_0_to_1(img):
    img = np.float32(img)
    img_flat = img.flatten()
    max_value = np.max(img_flat)
    min_value = np.min(img_flat)
    new_img = (img - min_value) * 1 / (max_value - min_value)
    return new_img

def compute_psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )

    if mse == 0:
        return 100
    PIXEL_MAX = 1.0 
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def supervised_model(w,h,c, k1=1, k2=3, s1=1, s2=2, pad="same", act='relu'):
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
    model.compile(optimizer=op, loss=vgg_loss)
    return model

class custconv2d(keras.layers.Layer):
  def __init__(self):
    super(custconv2d, self).__init__()
    self.w1 = self.add_weight(shape=(1,1,6,32), initializer="random_normal", trainable=True, name='w1')
    self.b1 = self.add_weight(shape=(32,), initializer="zeros", trainable=True, name='b1')
    self.w2 = self.add_weight(shape=(1,1,32,32), initializer="random_normal", trainable=True, name='w2')
    self.b2 = self.add_weight(shape=(32,), initializer="zeros", trainable=True, name='b2')
    self.w3 = self.add_weight(shape=(1,1,32,3), initializer="random_normal", trainable=True, name='w3')
    self.b3 = self.add_weight(shape=(3,), initializer="zeros", trainable=True, name='b3')
  
  def call(self, inputs):
    n, h, w, c = inputs.shape.as_list()
    a = tf.Variable(tf.zeros((1,896,1344,3), 'float32'))
    stride_ = 448#896 #test

    for hight in range(0, h-stride_+1, stride_):
      for width in range(0, w-stride_+1, stride_):
          inputs_ = inputs[:, hight:hight+stride_, width:width+stride_, :]
          temp0 = tf.nn.relu(tf.matmul(inputs_, self.w1) + self.b1)    
          temp1 = tf.nn.relu(tf.matmul(temp0, self.w2) + self.b2)
          a[:, hight:hight+stride_, width:width+stride_, :].assign(tf.nn.relu(tf.matmul(temp1, self.w3) + self.b3))
    return a
  
  
class custconv2d_2(keras.layers.Layer):
  def __init__(self):
    super(custconv2d_2, self).__init__()
    self.w1 = self.add_weight(shape=(1,1,6,32), initializer="random_normal", trainable=True, name='w1')
    self.b1 = self.add_weight(shape=(32,), initializer="zeros", trainable=True, name='b1')
    self.w2 = self.add_weight(shape=(1,1,32,32), initializer="random_normal", trainable=True, name='w2')
    self.b2 = self.add_weight(shape=(32,), initializer="zeros", trainable=True, name='b2')
    self.w3 = self.add_weight(shape=(1,1,32,3), initializer="random_normal", trainable=True, name='w3')
    self.b3 = self.add_weight(shape=(3,), initializer="zeros", trainable=True, name='b3')
  
  def call(self, inputs):
    n, h, w, c = inputs.shape.as_list()
    #stride_ = 64 #train
    stride_ = 128 #test
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
      
def transform(image, im_size=(256,256)):
    out = tf.cast(image, tf.float32)
    out = tf.image.resize_images(out, im_size)
    return out*2. - 1.
