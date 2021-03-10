# ==============================================================================
# MIT License
#
# Copyright 2021 Institute for Automotive Engineering of RWTH Aachen University.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================

import tensorflow as tf

from .SegmentationNetwork import PCLSegmentationNetwork


class CAM(tf.keras.layers.Layer):
  """Context Aggregation Module"""

  def __init__(self, in_channels, reduction_factor=16, bn_momentum=0.999, l2=0.0001):
    super(CAM, self).__init__()
    self.in_channels = in_channels
    self.reduction_factor = reduction_factor
    self.bn_momentum = bn_momentum
    self.l2 = l2

    self.pool = tf.keras.layers.MaxPool2D(
      pool_size=7,
      strides=1,
      padding='SAME'
    )

    self.squeeze = tf.keras.layers.Conv2D(
      filters=(self.in_channels // self.reduction_factor),
      kernel_size=1,
      strides=1,
      padding='SAME',
      kernel_initializer='glorot_uniform',
      kernel_regularizer=tf.keras.regularizers.L2(l2=self.l2)
    )
    self.squeeze_bn = tf.keras.layers.BatchNormalization(momentum=self.bn_momentum)

    self.excitation = tf.keras.layers.Conv2D(
      filters=self.in_channels,
      kernel_size=1,
      strides=1,
      padding='SAME',
      kernel_initializer='glorot_uniform',
      kernel_regularizer=tf.keras.regularizers.L2(l2=self.l2)
    )
    self.excitation_bn = tf.keras.layers.BatchNormalization(momentum=self.bn_momentum)

  def call(self, inputs, training=False):
    pool = self.pool(inputs)
    squeeze = tf.nn.relu(self.squeeze_bn(self.squeeze(pool)))
    excitation = tf.nn.sigmoid(self.excitation_bn(self.excitation(squeeze)))
    return inputs * excitation

  def get_config(self):
    config = super(CAM, self).get_config()
    config.update({'in_channels': self.in_channels,
                   'reduction_factor': self.reduction_factor,
                   'bn_momentum': self.bn_momentum,
                   'l2': self.l2})
    return config


class FIRE(tf.keras.layers.Layer):
  """FIRE MODULE"""

  def __init__(self, sq1x1_planes, ex1x1_planes, ex3x3_planes, bn_momentum=0.999, l2=0.0001):
    super(FIRE, self).__init__()
    self.sq1x1_planes = sq1x1_planes
    self.ex1x1_planes = ex1x1_planes
    self.ex3x3_planes = ex3x3_planes
    self.bn_momentum = bn_momentum
    self.l2 = l2

    self.squeeze = tf.keras.layers.Conv2D(
      filters=self.sq1x1_planes,
      kernel_size=1,
      strides=1,
      padding='SAME',
      kernel_regularizer=tf.keras.regularizers.L2(l2=self.l2)
    )
    self.squeeze_bn = tf.keras.layers.BatchNormalization(momentum=self.bn_momentum)

    self.expand1x1 = tf.keras.layers.Conv2D(
      filters=self.ex1x1_planes,
      kernel_size=1,
      strides=1,
      padding='SAME',
      kernel_regularizer=tf.keras.regularizers.L2(l2=self.l2)
    )
    self.expand1x1_bn = tf.keras.layers.BatchNormalization(momentum=self.bn_momentum)

    self.expand3x3 = tf.keras.layers.Conv2D(
      filters=self.ex3x3_planes,
      kernel_size=3,
      strides=1,
      padding='SAME',
      kernel_regularizer=tf.keras.regularizers.L2(l2=self.l2)
    )
    self.expand3x3_bn = tf.keras.layers.BatchNormalization(momentum=self.bn_momentum)

  def call(self, inputs, training=False):
    squeeze = tf.nn.relu(self.squeeze_bn(self.squeeze(inputs), training))
    expand1x1 = tf.nn.relu(self.expand1x1_bn(self.expand1x1(squeeze), training))
    expand3x3 = tf.nn.relu(self.expand3x3_bn(self.expand3x3(squeeze), training))
    return tf.concat([expand1x1, expand3x3], axis=3)

  def get_config(self):
    config = super(FIRE, self).get_config()
    config.update({'sq1x1_planes': self.sq1x1_planes,
                   'ex1x1_planes': self.ex1x1_planes,
                   'ex3x3_planes': self.ex3x3_planes,
                   'bn_momentum': self.bn_momentum,
                   'l2': self.l2})
    return config


class FIREUP(tf.keras.layers.Layer):
  """FIRE MODULE WITH TRANSPOSE CONVOLUTION"""

  def __init__(self, sq1x1_planes, ex1x1_planes, ex3x3_planes, stride, bn_momentum=0.99, l2=0.0001):
    super(FIREUP, self).__init__()
    self.sq1x1_planes = sq1x1_planes
    self.ex1x1_planes = ex1x1_planes
    self.ex3x3_planes = ex3x3_planes
    self.stride = stride
    self.bn_momentum = bn_momentum
    self.l2 = l2

    self.squeeze = tf.keras.layers.Conv2D(
      filters=self.sq1x1_planes,
      kernel_size=1,
      strides=1,
      padding='SAME',
      kernel_regularizer=tf.keras.regularizers.L2(l2=self.l2)
    )
    self.squeeze_bn = tf.keras.layers.BatchNormalization(momentum=self.bn_momentum)

    if self.stride == 2:
      self.upconv = tf.keras.layers.Conv2DTranspose(
        filters=self.sq1x1_planes,
        kernel_size=[1, 4],
        strides=[1, 2],
        padding='SAME',
        kernel_regularizer=tf.keras.regularizers.L2(l2=self.l2)
      )

    self.expand1x1 = tf.keras.layers.Conv2D(
      filters=self.ex1x1_planes,
      kernel_size=1,
      strides=1,
      padding='SAME',
      kernel_regularizer=tf.keras.regularizers.L2(l2=self.l2)
    )
    self.expand1x1_bn = tf.keras.layers.BatchNormalization(momentum=self.bn_momentum)

    self.expand3x3 = tf.keras.layers.Conv2D(
      filters=self.ex3x3_planes,
      kernel_size=3,
      strides=1,
      padding='SAME',
      kernel_regularizer=tf.keras.regularizers.L2(l2=self.l2)
    )
    self.expand3x3_bn = tf.keras.layers.BatchNormalization(momentum=self.bn_momentum)

  def get_config(self):
    config = super(FIREUP, self).get_config()
    config.update({'sq1x1_planes': self.sq1x1_planes,
                   'ex1x1_planes': self.ex1x1_planes,
                   'ex3x3_planes': self.ex3x3_planes,
                   'stride': self.stride,
                   'bn_momentum': self.bn_momentum,
                   'l2': self.l2})
    return config

  def call(self, inputs, training=False):
    squeeze = tf.nn.relu(self.squeeze_bn(self.squeeze(inputs), training))
    if self.stride == 2:
      upconv = tf.nn.relu(self.upconv(squeeze))
    else:
      upconv = squeeze
    expand1x1 = tf.nn.relu(self.expand1x1_bn(self.expand1x1(upconv), training))
    expand3x3 = tf.nn.relu(self.expand3x3_bn(self.expand3x3(upconv), training))
    return tf.concat([expand1x1, expand3x3], axis=3)


class SqueezeSegV2(PCLSegmentationNetwork):
  """SqueezeSegV2 Model as custom Keras Model in TF 2.4"""

  def __init__(self, mc):
    super(SqueezeSegV2, self).__init__(mc)
    self.mc = mc
    self.ZENITH_LEVEL = mc.ZENITH_LEVEL
    self.AZIMUTH_LEVEL = mc.AZIMUTH_LEVEL
    self.NUM_FEATURES = mc.NUM_FEATURES
    self.NUM_CLASS = mc.NUM_CLASS
    self.drop_rate = mc.DROP_RATE
    self.l2 = mc.L2_WEIGHT_DECAY
    self.bn_momentum = mc.BN_MOMENTUM

    # Layers
    self.conv1 = tf.keras.layers.Conv2D(
      input_shape=[self.ZENITH_LEVEL, self.AZIMUTH_LEVEL, self.mc.NUM_FEATURES],
      filters=64,
      kernel_size=3,
      strides=[1, 2],
      padding='SAME',
      kernel_regularizer=tf.keras.regularizers.L2(l2=self.l2)
    )
    self.bn1 = tf.keras.layers.BatchNormalization(momentum=self.bn_momentum)
    self.cam1 = CAM(in_channels=64, bn_momentum=self.bn_momentum, l2=self.l2)

    self.conv1_skip = tf.keras.layers.Conv2D(
      input_shape=[self.ZENITH_LEVEL, self.AZIMUTH_LEVEL, self.NUM_FEATURES],
      filters=64,
      kernel_size=1,
      strides=1,
      padding='SAME',
      kernel_regularizer=tf.keras.regularizers.L2(l2=self.l2)
    )
    self.bn1_skip = tf.keras.layers.BatchNormalization(momentum=self.bn_momentum)

    self.fire2 = FIRE(sq1x1_planes=16, ex1x1_planes=64, ex3x3_planes=64, bn_momentum=self.bn_momentum, l2=self.l2)
    self.cam2 = CAM(in_channels=128, bn_momentum=self.bn_momentum, l2=self.l2)
    self.fire3 = FIRE(sq1x1_planes=16, ex1x1_planes=64, ex3x3_planes=64, bn_momentum=self.bn_momentum, l2=self.l2)
    self.cam3 = CAM(in_channels=128, bn_momentum=self.bn_momentum, l2=self.l2)

    self.fire4 = FIRE(sq1x1_planes=32, ex1x1_planes=128, ex3x3_planes=128, bn_momentum=self.bn_momentum, l2=self.l2)
    self.fire5 = FIRE(sq1x1_planes=32, ex1x1_planes=128, ex3x3_planes=128, bn_momentum=self.bn_momentum, l2=self.l2)

    self.fire6 = FIRE(sq1x1_planes=48, ex1x1_planes=192, ex3x3_planes=192, bn_momentum=self.bn_momentum, l2=self.l2)
    self.fire7 = FIRE(sq1x1_planes=48, ex1x1_planes=192, ex3x3_planes=192, bn_momentum=self.bn_momentum, l2=self.l2)
    self.fire8 = FIRE(sq1x1_planes=64, ex1x1_planes=256, ex3x3_planes=256, bn_momentum=self.bn_momentum, l2=self.l2)
    self.fire9 = FIRE(sq1x1_planes=64, ex1x1_planes=256, ex3x3_planes=256, bn_momentum=self.bn_momentum, l2=self.l2)

    # Decoder
    self.fire10 = FIREUP(sq1x1_planes=64, ex1x1_planes=128, ex3x3_planes=128, stride=2, bn_momentum=self.bn_momentum,
                         l2=self.l2)
    self.fire11 = FIREUP(sq1x1_planes=32, ex1x1_planes=64, ex3x3_planes=64, stride=2, bn_momentum=self.bn_momentum,
                         l2=self.l2)
    self.fire12 = FIREUP(sq1x1_planes=16, ex1x1_planes=32, ex3x3_planes=32, stride=2, bn_momentum=self.bn_momentum,
                         l2=self.l2)
    self.fire13 = FIREUP(sq1x1_planes=16, ex1x1_planes=32, ex3x3_planes=32, stride=2, bn_momentum=self.bn_momentum,
                         l2=self.l2)

    self.conv14 = tf.keras.layers.Conv2D(
      filters=self.NUM_CLASS,
      kernel_size=3,
      strides=1,
      padding='SAME',
      kernel_regularizer=tf.keras.regularizers.L2(l2=self.l2)
    )
    self.dropout = tf.keras.layers.Dropout(self.drop_rate)

  def call(self, inputs, training=False, mask=None):
    lidar_input, lidar_mask = inputs[0], inputs[1]

    # Encoder
    x = tf.nn.relu(self.bn1(self.conv1(lidar_input), training))

    cam1_output = self.cam1(x, training)

    conv1_skip = self.bn1_skip(self.conv1_skip(lidar_input), training)

    x = tf.nn.max_pool2d(cam1_output, ksize=3, strides=[1, 2], padding='SAME')
    x = self.fire2(x, training)
    x = self.cam2(x, training)
    x = self.fire3(x, training)
    cam3_output = self.cam3(x, training)

    x = tf.nn.max_pool2d(cam3_output, ksize=3, strides=[1, 2], padding='SAME')
    x = self.fire4(x, training)
    fire5_output = self.fire5(x, training)

    x = tf.nn.max_pool2d(fire5_output, ksize=3, strides=[1, 2], padding='SAME')
    x = self.fire6(x, training)
    x = self.fire7(x, training)
    x = self.fire8(x, training)
    fire9_output = self.fire9(x, training)

    # Decoder
    x = self.fire10(fire9_output, training)
    x = tf.add(x, fire5_output)
    x = self.fire11(x, training)
    x = tf.add(x, cam3_output)
    x = self.fire12(x, training)
    x = tf.add(x, cam1_output)
    x = self.fire13(x, training)
    x = tf.add(x, conv1_skip)

    x = self.dropout(x, training)

    logits = self.conv14(x)

    return self.segmentation_head(logits, lidar_mask)

  def get_config(self):
    config = super(SqueezeSegV2, self).get_config()
    config.update({"mc": self.mc})
    return config
