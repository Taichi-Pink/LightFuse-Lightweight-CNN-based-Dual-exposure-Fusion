import tensorflow as tf
tf.enable_eager_execution()
from tensorflow import keras
from tensorflow.keras import layers, Model
# import matplotlib.pyplot as plt
import cv2
import numpy as np
import imageio as io
ch = 6
w = 256
h = 128
def supervised_model(w, h, c, k2=3, s2=2, pad="same"):
    input_ = keras.Input((w, h, c))
    conv_ = custconv2d()
    out_1 = conv_(input_)

    out_2 = layers.DepthwiseConv2D((k2, k2), strides=(s2, s2), padding=pad)(input_)
    out_2 = layers.DepthwiseConv2D((k2, k2), strides=(s2, s2), padding=pad)(out_2)
    out_2 = layers.SeparableConv2D(c // 2, (k2, k2), strides=(s2, s2), padding=pad)(out_2)

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
    def __init__(self, **kwargs):
        super(custconv2d, self).__init__(**kwargs)
        self.w1 = self.add_weight(shape=(1, 1, 6, 256), initializer="random_normal", trainable=True, name='w1')
        self.b1 = self.add_weight(shape=(256,), initializer="zeros", trainable=True, name='b1')
        self.w2 = self.add_weight(shape=(1, 1, 256, 64), initializer="random_normal", trainable=True, name='w2')
        self.b2 = self.add_weight(shape=(64,), initializer="zeros", trainable=True, name='b2')
        self.w3 = self.add_weight(shape=(1, 1, 64, 3), initializer="random_normal", trainable=True, name='w3')
        self.b3 = self.add_weight(shape=(3,), initializer="zeros", trainable=True, name='b3')

    def call(self, inputs):
        n, h, w, c = inputs.shape.as_list()
        stride_ = 16  # train
        # stride_ = 128 #test
        hi = []
        for hight in range(0, h - stride_ + 1, stride_):
            wi = []
            for width in range(0, w - stride_ + 1, stride_):
                inputs_ = inputs[:, hight:hight + stride_, width:width + stride_, :]
                temp0 = tf.matmul(inputs_, self.w1) + self.b1
                temp1 = tf.matmul(temp0, self.w2) + self.b2
                temp2 = tf.matmul(temp1, self.w3) + self.b3
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


def norm_0_to_1(img):
    img = np.float32(img)
    img_flat = img.flatten()
    max_value = np.max(img_flat)
    min_value = np.min(img_flat)
    new_img = (img - min_value) * 1 / (max_value - min_value)
    return new_img


# tf 1.15.0
def test_tflite():
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path="lite_model_cust.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # load data
    p_over = r'./over_exp256x128.png'
    over_exp = cv2.imread(p_over)
    over_exp = over_exp[:, :, ::-1]

    p_under = r'./under_exp256x128.png'
    under_exp = cv2.imread(p_under)
    under_exp = under_exp[:, :, ::-1]

    over_exp = cv2.resize(over_exp, (w, h))
    under_exp = cv2.resize(under_exp, (w, h))
    over_exp = norm_0_to_1(over_exp)
    under_exp = norm_0_to_1(under_exp)

    img = np.concatenate([under_exp, over_exp], axis=2)
    img = np.expand_dims(img, axis=0)
    img = img * 2.0 - 1.0

    # Test model on random input data.
    interpreter.set_tensor(input_details[0]['index'], img)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])

    ldr = np.squeeze(output_data)
    ldr_ = ((ldr + 1.0) / 2.0)
    tem = ldr_ * 255.0
    tem = tem.astype(np.uint8)
    io.imwrite('test_cust.png', tem)


if __name__ == "__main__":

    '''Run Custom Tensorflow Lite Model on Raspberry Pi'''
    test_tflite()

