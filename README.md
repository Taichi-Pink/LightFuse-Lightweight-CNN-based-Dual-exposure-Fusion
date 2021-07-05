# LightFuse: Lightweight CNN based Dual-exposure Fusion
##
<p align="middle">
<img src="exposure_fusion/github1.png" ><img src="exposure_fusion/github2.png"><img src="exposure_fusion/github3.png">
</p>

The official codebase for [LightFuse: Lightweight CNN based Dual-exposure Fusion](). Contains demo (see [demo.ipynb](demo.ipynb)) and scripts to reproduce experiments. To our best knowledge, this is the first lightweight HDR fusion algorithm that could be used in power and resource-constrained edge-computing devices. The proposed LightFuse model consists of two sub-networks: a ```GlobalNet``` and a ```DetailNet```. ```GlobalNet is to learn global illumination information on the spatial dimension, whereas DetailNet aims to enhance local details on the channel dimension```. Both GlobalNet and DetailNet are based solely on ```depthwise convolution and pointwise convolution``` to reduce required parameters and computations. LightFuse is trained with two extreme exposure LDR images to avoid problems such as large storage requirements, processing time, and power budget caused by a sequence of LDR images.

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
 <!-- * Download data from [SICE dataset](https://github.com/csjcai/SICE). Place it under ```Dataset``` folder. Split the data in Dataset_Part1 into train (80%) and test (20%) set. -->

### Test
 ```
 python test.py
 ```
### Train
  * Prepare TFRecord.
  ```
  python FuDataset.py
  ```
  * run train.py
  ```
  python train.py
  ```

<!-- ## Citation --> 
