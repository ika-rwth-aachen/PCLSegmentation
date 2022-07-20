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

"""Model configuration for SqueezeSeg"""
import numpy as np
from easydict import EasyDict

def rgb(bgr):
  return [bgr[2], bgr[1], bgr[0]]

def SqueezeSegV2KittiConfig():
  mc                    = EasyDict()

  mc.CLASSES            = ["None",
                           "car",
                           "bicycle",
                           "motorcycle",
                           "truck",
                           "other-vehicle",
                           "person",
                           "bicyclist",
                           "motorcyclist",
                           "road",
                           "parking",
                           "sidewalk",
                           "other-ground",
                           "building",
                           "fence",
                           "vegetation",
                           "trunk",
                           "terrain",
                           "pole",
                           "traffic-sign"]

  mc.NUM_CLASS          = len(mc.CLASSES)
  mc.CLS_2_ID           = dict(zip(mc.CLASSES, range(len(mc.CLASSES))))
  mc.CLS_LOSS_WEIGHT    = np.ones(mc.NUM_CLASS)

  color_map             = {0:  rgb([0, 0, 0]),
                           1:  rgb([245, 150, 100]),
                           2:  rgb([245, 230, 100]),
                           3:  rgb([150, 60, 30]),
                           4:  rgb([180, 30, 80]),
                           5:  rgb([255, 0, 0]),
                           6:  rgb([30, 30, 255]),
                           7:  rgb([200, 40, 255]),
                           8:  rgb([90, 30, 150]),
                           9:  rgb([255, 0, 255]),
                           10: rgb([255, 150, 255]),
                           11: rgb([75, 0, 75]),
                           12: rgb([75, 0, 175]),
                           13: rgb([0, 200, 255]),
                           14: rgb([50, 120, 255]),
                           15: rgb([0, 175, 0]),
                           16: rgb([0, 60, 135]),
                           17: rgb([80, 240, 150]),
                           18: rgb([150, 240, 255]),
                           19: rgb([0, 0, 255])
                           }

  mc.CLS_COLOR_MAP = np.zeros((mc.NUM_CLASS, 3), dtype=np.float32)
  for key, value in color_map.items():
    mc.CLS_COLOR_MAP[key] = np.array(value, np.float32) / 255.0

  # Input Shape
  mc.BATCH_SIZE         = 64
  mc.AZIMUTH_LEVEL      = 1024
  mc.ZENITH_LEVEL       = 64
  mc.NUM_FEATURES       = 6

  # Loss
  mc.USE_FOCAL_LOSS     = False  # either use focal loss or sparse categorical cross entropy
  mc.FOCAL_GAMMA        = 2.0
  mc.CLS_LOSS_COEF      = 15.0
  mc.DENOM_EPSILON      = 1e-12   # small value used in denominator to prevent division by 0

  # Gradient Decent
  mc.LEARNING_RATE      = 0.001
  mc.LR_DECAY_STEPS     = 500
  mc.LR_DECAY_FACTOR    = 0.99
  mc.MAX_GRAD_NORM      = 100.0

  # Network
  mc.L2_WEIGHT_DECAY    = 0.05
  mc.DROP_RATE          = 0.1
  mc.BN_MOMENTUM        = 0.9
  mc.REDUCTION          = 16

  # Dataset
  mc.DATA_AUGMENTATION  = True
  mc.RANDOM_FLIPPING    = True
  mc.SHIFT_UP_DOWN      = 0
  mc.SHIFT_LEFT_RIGHT   = 70

  # x, y, z, intensity, distance
  mc.INPUT_MEAN           = np.array([[[-0.047, 0.365, -0.855, 0.2198, 8.3568]]])
  mc.INPUT_STD            = np.array([[[10.154, 7.627, 0.8651, 0.1764, 9.6474]]])

  return mc
