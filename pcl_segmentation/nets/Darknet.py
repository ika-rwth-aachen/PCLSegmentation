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

class BasicBlock(tf.keras.layers.Layer):
  """Basic Block of the Darknet Architecture"""

  def __init__(self, inplanes, planes, bn_momentum=0.9):
    super(BasicBlock, self).__init__()
    self.conv1 = tf.keras.layers.Conv2D(
      filters=planes[0],
      kernel_size=1,
      strides=1,
      padding='VALID',
      use_bias=False
    )
    self.bn1 = tf.keras.layers.BatchNormalization(momentum=bn_momentum)
    self.leaky_relu1 = tf.keras.layers.LeakyReLU(0.1)

    self.conv2 = tf.keras.layers.Conv2D(
      filters=planes[1],
      kernel_size=3,
      strides=1,
      padding='SAME',
      use_bias=False
    )
    self.bn2 = tf.keras.layers.BatchNormalization(momentum=bn_momentum)
    self.leaky_relu2 = tf.keras.layers.LeakyReLU(0.1)

  def call(self, inputs, training=False):
    residual = inputs

    x = self.conv1(inputs)
    x = self.bn1(x)
    x = self.leaky_relu1(x)

    x = self.conv2(x)
    x = self.bn2(x)
    x = self.leaky_relu2(x)

    x += residual
    return x


class EncoderLayer(tf.keras.layers.Layer):
  """Basic Encoder Layer of the Darknet Architecture"""

  def __init__(self, block, planes, num_blocks, stride, bn_momentum=0.9):
    super(EncoderLayer, self).__init__()
    self.num_blocks = num_blocks
    # downsample
    self.conv1 = tf.keras.layers.Conv2D(
      filters=planes[1],
      kernel_size=3,
      strides=[1, stride],
      dilation_rate=1,
      padding='SAME',
      use_bias=False
    )
    self.bn1 = tf.keras.layers.BatchNormalization(momentum=bn_momentum)
    self.leaky_relu1 = tf.keras.layers.LeakyReLU(0.1)

    inplanes = planes[1]
    for i in range(0, self.num_blocks):
      setattr(self,
              "residual_{}".format(i),
              block(inplanes=inplanes,
                    planes=planes,
                    bn_momentum=bn_momentum)
              )

  def call(self, inputs, training=False):
    # downsample
    x = self.conv1(inputs)
    x = self.bn1(x)
    x = self.leaky_relu1(x)
    for i in range(0, self.num_blocks):
      x = getattr(self, "residual_{}".format(i))(x)
    return x


class DecoderLayer(tf.keras.layers.Layer):
  """Basic Decoder Layer of the Darknet Architecture"""
  def __init__(self, block, planes, stride, bn_momentum=0.9):
    super(DecoderLayer, self).__init__()
    self.stride = stride
    if self.stride == 2:
      # upsample
      self.upconv1 = tf.keras.layers.Conv2DTranspose(
        filters=planes[1],
        kernel_size=[1, 4],
        strides=[1, 2],
        padding="SAME"
      )
    else:
      # keep constant
      self.conv1 = tf.keras.layers.Conv2D(
        filters=planes[1],
        kernel_size=3,
        padding="SAME"
      )
    self.bn1 = tf.keras.layers.BatchNormalization(momentum=bn_momentum)
    self.leaky_relu1 = tf.keras.layers.LeakyReLU(0.1)
    self.block = block(planes[1], planes, bn_momentum)

  def call(self, inputs, training=False):
    if self.stride == 2:
      x = self.upconv1(inputs)
    else:
      x = self.conv1(inputs)
    x = self.bn1(x)
    x = self.leaky_relu1(x)
    x = self.block(x)
    return x


# number of layers per model
model_blocks = {
  21: [1, 1, 2, 2, 1],
  53: [1, 2, 8, 8, 4],
}

class Darknet(PCLSegmentationNetwork):
  """Implements the Darknet Segmentation Model"""
  def __init__(self, mc):
    super(Darknet, self).__init__(mc)
    self.mc = mc
    self.drop_rate = mc.DROP_RATE
    self.bn_momentum = mc.BN_MOMENTUM
    self.output_stride = mc.OUTPUT_STRIDE  # Output stride only horizontally
    self.num_layers = mc.NUM_LAYERS
    self.last_channels_encoder = 1024

    # stride play
    self.encoder_strides = [2, 2, 2, 2, 2]

    # check current stride
    current_os = 1
    for s in self.encoder_strides:
      current_os *= s
    print("Encoder Original OS: ", current_os)

    # make the new stride
    if self.output_stride > current_os:
      print("Can't do OS, ", self.output_stride,
            " because it is bigger than original ", current_os)
    else:
      # redo strides according to needed stride
      for i, stride in enumerate(reversed(self.encoder_strides), 0):
        if int(current_os) != self.output_stride:
          if stride == 2:
            current_os /= 2
            self.encoder_strides[-1 - i] = 1
          if int(current_os) == self.output_stride:
            break
      print("Encoder New OS: ", int(current_os))
      print("Encoder Strides: ", self.encoder_strides)

    # generate layers depending on darknet type
    self.num_blocks = model_blocks[self.num_layers]

    # input layer
    self.conv1 = tf.keras.layers.Conv2D(
      input_shape=[self.mc.ZENITH_LEVEL, self.mc.AZIMUTH_LEVEL, self.mc.NUM_FEATURES],
      filters=32,
      kernel_size=3,
      strides=1,
      padding='SAME',
      use_bias=False
    )
    self.bn1 = tf.keras.layers.BatchNormalization(momentum=self.bn_momentum)
    self.leaky_relu1 = tf.keras.layers.LeakyReLU(0.1)

    self.enc1 = EncoderLayer(block=BasicBlock, planes=[32, 64], num_blocks=self.num_blocks[0],
                             stride=self.encoder_strides[0], bn_momentum=self.bn_momentum)

    self.enc2 = EncoderLayer(block=BasicBlock, planes=[64, 128], num_blocks=self.num_blocks[1],
                             stride=self.encoder_strides[1], bn_momentum=self.bn_momentum)

    self.enc3 = EncoderLayer(block=BasicBlock, planes=[128, 256], num_blocks=self.num_blocks[2],
                             stride=self.encoder_strides[2], bn_momentum=self.bn_momentum)

    self.enc4 = EncoderLayer(block=BasicBlock, planes=[256, 512], num_blocks=self.num_blocks[3],
                             stride=self.encoder_strides[3], bn_momentum=self.bn_momentum)

    self.enc5 = EncoderLayer(block=BasicBlock, planes=[512, self.last_channels_encoder], num_blocks=self.num_blocks[4],
                             stride=self.encoder_strides[4], bn_momentum=self.bn_momentum)

    self.dropout = tf.keras.layers.Dropout(self.drop_rate)

    # Decoder
    self.decoder_strides = [2, 2, 2, 2, 2]
    # check current stride
    current_os = 1
    for s in self.decoder_strides:
      current_os *= s
    print("Decoder original OS: ", int(current_os))
    # redo strides according to needed stride
    for i, stride in enumerate(self.decoder_strides):
      if int(current_os) != self.output_stride:
        if stride == 2:
          current_os /= 2
          self.decoder_strides[i] = 1
        if int(current_os) == self.output_stride:
          break
    print("Decoder new OS: ", int(current_os))
    print("Decoder strides: ", self.decoder_strides)

    self.dec5 = DecoderLayer(BasicBlock,
                             planes=[self.last_channels_encoder, 512],
                             bn_momentum=self.bn_momentum,
                             stride=self.decoder_strides[0])
    self.dec4 = DecoderLayer(BasicBlock,
                             planes=[512, 256],
                             bn_momentum=self.bn_momentum,
                             stride=self.decoder_strides[1])
    self.dec3 = DecoderLayer(BasicBlock,
                             planes=[256, 128],
                             bn_momentum=self.bn_momentum,
                             stride=self.decoder_strides[2])
    self.dec2 = DecoderLayer(BasicBlock,
                             planes=[128, 64],
                             bn_momentum=self.bn_momentum,
                             stride=self.decoder_strides[3])
    self.dec1 = DecoderLayer(BasicBlock,
                             planes=[64, 32],
                             bn_momentum=self.bn_momentum,
                             stride=self.decoder_strides[4])

    # Head
    self.head = tf.keras.layers.Conv2D(
      filters=mc.NUM_CLASS,
      kernel_size=3,
      strides=1,
      padding="SAME"
    )


  def run_enc_block(self, x, layer, skips, os):
    y = layer(x)
    if y.shape[1] < x.shape[1] or y.shape[2] < x.shape[2]:
      skips[os] = x
      os *= 2
    x = y
    return x, skips, os

  def run_dec_block(self, x, layer, skips, os):
    y = layer(x)  # up
    if y.shape[2] > x.shape[2]:
      os //= 2  # match skip
      y = y + skips[os]  # add skip
    x = y
    return x, skips, os

  def call(self, inputs, training=False, mask=None):
    lidar_input, lidar_mask = inputs[0], inputs[1]

    # run cnn
    # store for skip connections
    skips = {}
    os = 1

    # first layer
    x, skips, os = self.run_enc_block(lidar_input, self.conv1, skips, os)
    x = self.bn1(x)
    x = self.leaky_relu1(x)

    # all encoder blocks with intermediate dropouts
    x, skips, os = self.run_enc_block(x, self.enc1, skips, os)
    x = self.dropout(x, training)
    x, skips, os = self.run_enc_block(x, self.enc2, skips, os)
    x = self.dropout(x, training)
    x, skips, os = self.run_enc_block(x, self.enc3, skips, os)
    x = self.dropout(x, training)
    x, skips, os = self.run_enc_block(x, self.enc4, skips, os)
    x = self.dropout(x, training)
    x, skips, os = self.run_enc_block(x, self.enc5, skips, os)
    x = self.dropout(x, training)

    # run decoder layers
    x, skips, os = self.run_dec_block(x, self.dec5, skips, os)
    x, skips, os = self.run_dec_block(x, self.dec4, skips, os)
    x, skips, os = self.run_dec_block(x, self.dec3, skips, os)
    x, skips, os = self.run_dec_block(x, self.dec2, skips, os)
    x, skips, os = self.run_dec_block(x, self.dec1, skips, os)

    x = self.dropout(x, training)
    logits = self.head(x)

    return self.segmentation_head(logits, lidar_mask)

