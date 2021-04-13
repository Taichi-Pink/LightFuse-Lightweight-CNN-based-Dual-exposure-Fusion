# LightFuse: Lightweight CNN based Dual-exposure Fusion
## 
This is the implementation for link:LightFuse: Lightweight CNN based Dual-exposure Fusion.

Deep convolutional neural networks (DNN) aided high dynamic range (HDR) imaging has received a lot of attention recently. The quality of generated HDR images have overperformed the traditional counterparts. The generation of HDR image rely on combination of multiple exposure images with the help of DNN computation. However, DNN is prone to be computationally intensive and power-hungry. To address the challenge, we propose LightFuse, a light-weight CNN-based algorithm for dual-exposure image fusion. To our best knowledge, this is the first lightweight HDR fusion algorithm that could be used in power and resource constrained edge-computing devices. It is challenging to train a lightweight model with fewer parameters and layers while maintain comparable performance. The proposed LightFuse model consists of two sub-networks: a ```CombiningNet``` and a ```FilteringNet```. The ```goal of CombiningNet is to learn
the channel-related information, whereas FilteringNet aims in combining the spatial information```. Both CombiningNet and FilteringNet is based on ```depthwise separable convolution``` to reduce required parameters and computations. LightFuse is trained with extreme exposure images to avoid possible fail during inference phase.

## Get Started
### Prerequisites

## Implemet
  * dowload data from link, run sort_ev.py.
  * split the data into training and testing set.
  * run FuDataset.py; pair its lowest and highest-exposed img according to .exls.
  * run train.py
  * run test.py

## Citation
