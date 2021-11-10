Deploy LightFuse on Raspberry Pi

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
