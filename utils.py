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
