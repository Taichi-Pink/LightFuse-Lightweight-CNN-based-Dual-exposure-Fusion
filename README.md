# LightFuse: Lightweight CNN based Dual-exposure Fusion
## 
This is the implementation for link:LightFuse: Lightweight CNN based Dual-exposure Fusion.

Deep convolutional neural networks (DNN) aided high dynamic range (HDR) imaging has received a lot of attention recently. The quality of generated HDR images have overperformed the traditional counterparts. The generation of HDR image rely on combination of multiple exposure images with the help of DNN computation. However, DNN is prone to be computationally intensive and power-hungry. To address the challenge, we propose LightFuse, a light-weight CNN-based algorithm for dual-exposure image fusion. To our best knowledge, this is the first lightweight HDR fusion algorithm that could be used in power and resource constrained edge-computing devices. It is challenging to train a lightweight model with fewer parameters and layers while maintain comparable performance. The proposed LightFuse model consists of two sub-networks: a ```CombiningNet``` and a ```FilteringNet```. The ```goal of CombiningNet is to learn
the channel-related information, whereas FilteringNet aims in combining the spatial information```. Both CombiningNet and FilteringNet is based on ```depthwise separable convolution``` to reduce required parameters and computations. LightFuse is trained with extreme exposure images to avoid possible fail during inference phase.

## Get Started
### Prerequisites
 * Python 	       = 3.7.9
 * TensorFlow     = 1.15.0
 * Opencv-python  = 4.4.0.44
 * Scipy          = 1.5.2
 * Matplotlib     = 3.3.1
  
### Setup
 * Clone this repo:
 ```
 git clone https://github.com/Taichi-Pink/LightFuse-Lightweight-CNN-based-Dual-exposure-Fusion.git
 cd LightFuse-Lightweight-CNN-based-Dual-exposure-Fusion
 ```
 * Download data from [SICE dataset](https://github.com/csjcai/SICE).Place it under ```Dataset``` folder. Split the data in Dataset_Part1 into train (80%) and test (20%) set.

### Demo
 ```
 python test.py
 ```
### Train
  * Prepare TFRecord.
  ```
  python FuDataset.py
  ```
  * run train.py


## Citation
