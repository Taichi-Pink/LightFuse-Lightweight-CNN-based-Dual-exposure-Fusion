from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
import cv2, glob, os,math
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras import backend as K

select_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
vgg19 = VGG16(include_top=False, weights='imagenet', input_shape=(256,256,3))
vgg19.trainable = False
for l in vgg19.layers:
  l.trainable = False
select = [vgg19.get_layer(name).output for name in select_layers]

model_vgg = Model(inputs=vgg19.input, outputs=select)
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

def model_test1(w,h,c):
  input_ = keras.Input((w, h, c))
  x1 = layers.SeparableConv2D(256, (3, 3), strides=(1, 1), padding="same", activation='relu')(input_)
  x1 = layers.SeparableConv2D(64, (3, 3), strides=(1, 1), padding="same", activation='relu')(x1)
  x1 = layers.SeparableConv2D(3, (3, 3), strides=(1, 1), padding="same", activation='relu')(x1)
  output_ = layers.Activation('tanh')(x1) 
  model = Model(inputs=input_, outputs=output_)
  op = keras.optimizers.Adam(learning_rate=0.001)
  model.compile(optimizer=op, loss=vgg_loss)
  return model

def model_test(w,h,c):
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
  model = Model(inputs=input_, outputs=output_)
  op = keras.optimizers.Adam(learning_rate=0.001)
  model.compile(optimizer=op, loss=vgg_loss)
  return model
     
def channel_shuffle(x):
    num_groups = 2
    n, h, w, c = x.shape.as_list()
    x_reshaped = tf.reshape(x, [-1, h, w, num_groups, c // num_groups])
    x_transposed = tf.transpose(x_reshaped, [0, 1, 2, 4, 3])
    output = tf.reshape(x_transposed, [-1, h, w, c])
    return output
      
def transform(image, im_size=(256,256)):
    out = tf.cast(image, tf.float32)
    out = tf.image.resize_images(out, im_size)
    return out*2. - 1.

class MyLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.a = self.add_weight(name='a', 
                                    shape=(1,),
                                    initializer='uniform',
                                    trainable=True)
        self.b = self.add_weight(name='b', 
                                    shape=(1,),
                                    initializer='uniform',
                                    trainable=True)                         
        super(MyLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return self.a*x + self.b

    def compute_output_shape(self, input_shape):
        return input_shape[0]