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

import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import cv2
import time
import argparse

sys.path.append(".")
from configs import SqueezeSegV2Config


class RunningStd(object):
  def __init__(self):
    self.s0, self.s1, self.s2 = 0.0, 0.0, 0.0

  def include(self, array):
    self.s0 += array.size
    self.s1 += np.sum(array)
    self.s2 += np.sum(np.square(array))

  @property
  def std(self):
    return np.sqrt((self.s0 * self.s2 - self.s1 * self.s1) / (self.s0 * (self.s0 - 1)))


def main(args):
  config = SqueezeSegV2Config()  # only for viz

  input_files = sorted(glob.glob(os.path.join(args.input_dir, "*.npy")))

  if args.output_dir:
    outfile_path = os.path.join(args.output_dir, "train.txt")
    output_mask_path = os.path.join(args.output_dir, "mask.npy")
  else:
    outfile_path, output_mask_path = None, None

  target_width = 240
  target_height = 32

  print("Number of Files:", len(input_files))

  # for means
  all_x_means = []
  all_y_means = []
  all_z_means = []
  all_i_means = []
  all_d_means = []

  all_depths = []

  # for mask
  running_mask = np.zeros((target_height, target_width))

  # for std
  ov_x = RunningStd()
  ov_y = RunningStd()
  ov_z = RunningStd()
  ov_i = RunningStd()
  ov_d = RunningStd()

  class_wise_count = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0}

  with open(outfile_path, 'w') as outfile:
    for idx, f in tqdm.tqdm(enumerate(input_files), total=len(input_files)):
      lidar = np.load(f)

      all_x_means.append(np.mean(lidar[:, :, 0]))
      ov_x.include(lidar[:, :, 0])

      all_y_means.append(np.mean(lidar[:, :, 1]))
      ov_y.include(lidar[:, :, 1])

      all_z_means.append(np.mean(lidar[:, :, 2]))
      ov_z.include(lidar[:, :, 2])

      all_i_means.append(np.mean(lidar[:, :, 3]))
      ov_i.include(lidar[:, :, 3])

      all_d_means.append(np.mean(lidar[:, :, 4]))
      ov_d.include(lidar[:, :, 4])

      if len(all_depths) < 5000:
        all_depths.append(lidar[:, :, 4])

      running_mask += np.reshape((lidar[:, :, 4] > 0), [target_height, target_width])

      unique, count = np.unique(lidar[:, :, 5], return_counts=True)
      for train_id, c in zip(unique, count):
        try:
          class_wise_count[int(train_id)] += c
        except KeyError:
          print("WARNING INVALID LABEL ID FOUND: ", int(train_id))
          print("FILE: ", f)
          print("SKIPPING CURRENT SAMPLE")
          continue

      if idx % 25 == 0 and args.do_visualisation:
        label_cloud = ((255 * config.CLS_COLOR_MAP[lidar[:, :, 5].astype(np.uint8)]).astype(np.uint8))
        label_cloud = cv2.cvtColor(label_cloud, cv2.COLOR_RGB2BGR)
        label_cloud = cv2.resize(label_cloud, (0, 0), fx=3, fy=3)
        cv2.imshow("Spherical Label Image", label_cloud)
        cv2.waitKey(1)
        time.sleep(0.05)

      if outfile_path:
        outfile.write(f.split("/")[-1].split(".")[0] + os.linesep)

  print("x_mean = ", np.mean(all_x_means))
  print("x_std  = ", ov_x.std)

  print("y_mean = ", np.mean(all_y_means))
  print("y_std  = ", ov_y.std)

  print("z_mean = ", np.mean(all_z_means))
  print("z_std  = ", ov_z.std)

  print("intensities_mean = ", np.mean(all_i_means))
  print("intensities_std  = ", ov_i.std)

  print("distances_mean = ", np.mean(all_d_means))
  print("distances_std  = ", ov_d.std)

  mask = np.mean(all_depths, axis=0)

  plt.imshow(mask, cmap='gray')
  plt.show()

  # to perform prediction
  lidar_mask = np.reshape(
    (mask > 0),
    [target_height, target_width, 1]
  )
  print("lidar mask shape: ", np.shape(lidar_mask))
  plt.imshow(lidar_mask[:, :, 0], cmap='gray')
  plt.show()

  if args.output_dir:
    np.save(output_mask_path, lidar_mask)

  print("number of labels for each class:")
  print(class_wise_count)

  all_pixels = sum(list(class_wise_count.values()))

  for k, v in class_wise_count.items():
    print("Class ID: ", k)
    print("Ratio: ", v / all_pixels)
    print("")


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Parse flags for the manually annotated PCD to *.NPY conversion")
  parser.add_argument('-i', '--input_dir', type=str, required=True, help='Input directory which contains the *.npy '
                                                                         'files. A glob patter `*.npy` is applied to '
                                                                         'this path')
  parser.add_argument('-o', '--output_dir', type=str, help='Output path to which `train.txt` and also'
                                                           '`mask.npy` is applied. If not specified, no files'
                                                           'will be written and a test run is performed.')
  parser.add_argument('-v', '--do_visualisation', type=bool, default=True,
                      help='Visualize the data during the computation')

  args = parser.parse_args()
  main(args)
