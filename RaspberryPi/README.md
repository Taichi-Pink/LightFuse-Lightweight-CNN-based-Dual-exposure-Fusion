# Deploy LightFuse on Raspberry Pi

## Get Started
### Prerequisites
 * A Raspberry Pi 2/3/4
 * An SD Card
 
### Setup
 * Install OS on Raspberry Pi. Please follow [this tutorial](https://ziyiliu29.medium.com/install-os-on-raspberry-pi-eef50a402510).
 * Set TensorFlow environment on your Raspberry Pi. Please follow [this tutorial](https://ziyiliu29.medium.com/deploy-tensorflow-model-on-raspberry-pi-1ba31d22c848)

### Test

  * Convert pre-trained weights to a TFLite file.
  ```
  python tflite_converter.py
  ```
  * Run tflite_test.py. 
  ```
  python tflite_test.py
  ```
  * Check the saved image ```test_cust.png``` on your Raspberry Pi.

<!-- ## Citation --> 
