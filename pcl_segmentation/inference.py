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

import os.path
import glob

import tqdm
from PIL import Image
import argparse

from configs import *
from utils.util import *


def inference(arg):
  config = SqueezeSegV2Config()

  model = tf.keras.models.load_model(arg.path_to_model)

  if not os.path.exists(arg.output_dir):
    os.makedirs(arg.output_dir)

  for f in tqdm.tqdm(list(glob.iglob(arg.input_path))):
    print("Process: {0}".format(f))

    sample = np.load(f).astype(np.float32, copy=False)

    # Get x, y, z, intensity, depth
    lidar = sample[:, :, :5]

    # Compute binary mask: True where the depth is bigger then 0, false in any other case
    mask = lidar[:, :, 4] > 0

    # Normalize input data using the mean and standard deviation
    lidar = (lidar - config.INPUT_MEAN) / config.INPUT_STD

    # Set lidar on all channels to zero where the mask is False. Ie. where no points are present
    lidar[~mask] = 0.0

    # Append mask to lidar input
    lidar = np.append(lidar, np.expand_dims(mask, -1), axis=2)

    # Get segmentation map from sample
    label = sample[:, :, 5]

    # set label to None class where no points are present
    label[~mask] = config.CLASSES.index("None")

    # add batch dim
    lidar = np.expand_dims(lidar, axis=0)
    mask = np.expand_dims(mask, axis=0)

    # execute model
    probabilities, predictions = model([lidar, mask])

    # to numpy and remove batch dimension
    predictions = predictions.numpy()[0]

    # save the data
    file_name = f.strip('.npy').split('/')[-1]
    np.save(
      os.path.join(arg.output_dir, 'pred_' + file_name + '.npy'),
      predictions
    )

    depth_map = Image.fromarray(
      (255 * normalize(lidar[0][:, :, 3])).astype(np.uint8))
    label_map = Image.fromarray(
      (255 * config.CLS_COLOR_MAP[predictions]).astype(np.uint8))

    blend_map = Image.blend(
      depth_map.convert('RGBA'),
      label_map.convert('RGBA'),
      alpha=1.0
    )

    blend_map.save(
      os.path.join(arg.output_dir, 'plot_' + file_name + '.png'))

    # save the label plot
    label_map = Image.fromarray(
      (255 * config.CLS_COLOR_MAP[label.astype(np.int32)]).astype(np.uint8)
    )
    blend_map = Image.blend(
      depth_map.convert('RGBA'),
      label_map.convert('RGBA'),
      alpha=1.0
    )
    blend_map.save(
      os.path.join(arg.output_dir, 'plot_gt_' + file_name + '.png')
    )


if __name__ == '__main__':
  physical_devices = tf.config.experimental.list_physical_devices('GPU')
  if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

  parser = argparse.ArgumentParser(description='Parse Flags for the inference script!')
  parser.add_argument('-d', '--input_path', type=str,
                      help='Input LiDAR scans to be detected. Must be a glob pattern input such as'
                           '`./data/samples/*.npy` !')
  parser.add_argument('-m', '--model', type=str,
                      help='Model name either `squeezesegv2`, `darknet53`, `darknet21`')
  parser.add_argument('-t', '--output_dir', type=str,
                      help="Directory where to write the model predictions and visualizations")
  parser.add_argument('-p', '--path_to_model', type=str,
                      help='Path to the model')
  args = parser.parse_args()

  inference(args)
