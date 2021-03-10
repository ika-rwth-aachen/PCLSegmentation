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
import glob
import numpy as np
import cv2
import argparse

from configs import SqueezeSegV2Config

ignore_id = 10

cityscapes_trainId_to_trainId = {
  # Road
  1: 0,

  # Sidewalk
  2: 1,

  # Building
  15: 2,  # Building
  16: 2,  # Wall
  19: 2,  # Bridge
  20: 2,  # Tunnel

  # Pole
  21: 3,  # Pole
  23: 3,  # Traffic Sign
  24: 3,  # Traffic Light

  # Vegetation
  25: 4,  # Vegetation
  26: 4,  # Terrain

  # Person
  5: 5,

  # Two Wheeler
  6: 6,  # Rider
  12: 6,  # Bicycle
  11: 6,  # Motorcycle

  # Car
  7: 7,  # Car
  14: 7,  # Trailer

  # Truck
  8: 8,

  # Bus
  9: 9,  # Bus
  13: 9,  # Caravan

  # None
  27: ignore_id,  # Sky class
  0: ignore_id,
  255: ignore_id
}


def generateTrainIdMap(length=256):
  """
  Create a numpy array of length 'length' which maps the index to the trainID
  """
  map = np.ones((length)) * ignore_id
  for k, v in cityscapes_trainId_to_trainId.items():
    map[k] = v
  return map


def normalize(x):
  return (x - x.min()) / (x.max() - x.min())


def pcl_xyz_i_r_d_l_to_information_map(pcl,
                                       H=32, W=240, C=7,
                                       leftPhi=np.radians(24.32),
                                       rightPhi=np.radians(22.23)):
  """
    Project velodyne points into front view depth map.
    :param pcl: velodyne points in shape [:,5]
    :param H: the row num of depth map, could be 64(default), 32, 16
    :param W: the col num of depth map
    :param C: the channel size of depth map
        3 cartesian coordinates (x; y; z),
        an intensity measurement and
        range r = sqrt(x^2 + y^2 + z^2)
    :param dtheta: the delta theta of H, in radian
    :param dphi: the delta phi of W, in radian
    :param use_ring: Use the ring index as height or projection of theta
    :return: `depth_map`: the projected depth map of shape[H,W,C]
    """

  x = pcl[:, 0]
  y = pcl[:, 1]
  z = pcl[:, 2]
  i = pcl[:, 3]
  r = pcl[:, 4].astype(int)
  d = pcl[:, 5]
  l = pcl[:, 6].astype(int)

  # projection on to a 2D plane
  deltaPhi_ = (rightPhi + leftPhi) / W
  phi = np.arctan2(y, x)
  phi_ = ((leftPhi - phi) / deltaPhi_).astype(int)

  # mask
  mask = np.zeros_like(l)
  mask[d > 0] = 1

  # points that exceed the boundaries must be removed
  delete_ids = np.where(np.logical_or(phi_ < 0, phi_ >= W))

  x = np.delete(x, delete_ids)
  y = np.delete(y, delete_ids)
  z = np.delete(z, delete_ids)
  i = np.delete(i, delete_ids)
  r = np.delete(r, delete_ids)
  d = np.delete(d, delete_ids)
  l = np.delete(l, delete_ids)
  phi_ = np.delete(phi_, delete_ids)
  mask = np.delete(mask, delete_ids)

  information_map = np.zeros((H, W, C))

  information_map[(H - 1) - r, phi_, 0] = x
  information_map[(H - 1) - r, phi_, 1] = y
  information_map[(H - 1) - r, phi_, 2] = z
  information_map[(H - 1) - r, phi_, 3] = i
  information_map[(H - 1) - r, phi_, 4] = d
  information_map[(H - 1) - r, phi_, 5] = l
  information_map[(H - 1) - r, phi_, 6] = mask

  return information_map


def main(args):
  trainIdMap = generateTrainIdMap()
  mc = SqueezeSegV2Config()  # only for visualisations

  # initialize emtpy dict to count the class distribution
  class_distribution = {}
  for k, v in cityscapes_trainId_to_trainId.items():
    class_distribution[v] = 0

  # retrieve *.npy files
  files = glob.glob(os.path.join(args.input_dir, "*.pcd"))

  for file in files:
    with open(file, "r") as pcd_file:
      lines = [line.strip().split(" ") for line in pcd_file.readlines()]

    print("Parsing file: ", file)

    pcl = []
    is_data = False
    for line in lines:
      if line[0] == 'DATA':  # skip the header
        is_data = True
        continue
      if is_data:
        x = float(line[0])  # x
        y = float(line[1])  # y
        z = float(line[2])  # z
        i = int(line[3])  # intensity
        r = float(line[4])  # ring
        l = int(line[5])  # label id
        o = int(line[6])  # object id
        d = np.sqrt(x ** 2 + y ** 2 + z ** 2)

        # not labled points
        if o == -1 and l == 0:
          l = ignore_id

        # only points in the camera frame
        if x > 0:
          point = np.array((x, y, z, i, r, d, l))
          pcl.append(point)

    # spherical projection
    information_map = pcl_xyz_i_r_d_l_to_information_map(np.asarray(pcl))

    # apply trainId Mapping
    information_map[:, :, 5] = trainIdMap[information_map[:, :, 5].astype(int)]

    # get binary mask
    binary_mask = information_map[:, :, 6]
    binary_mask_copy = binary_mask.copy()

    # apply mask
    information_map[binary_mask == 0] = 0
    information_map[:, :, 5][binary_mask == 0] = ignore_id  # set label to ignore_id

    # for class distribution statistics
    unique, counts = np.unique(information_map[:, :, 5], return_counts=True)
    for id, count in zip(unique, counts):
      class_distribution[id] += count

    if args.vis_dir:
      # x cloud
      cloud = (normalize(information_map[:, :, 0].copy()) * 255).astype(np.uint8)
      cloud = cv2.applyColorMap(cloud, cv2.COLORMAP_JET)
      cloud = cv2.resize(cloud, (0, 0), fx=3, fy=3)
      cv2.imshow("spherical x image", cloud)
      cv2.imwrite(args.vis_dir + "/" + "x.png", cloud)
      cv2.waitKey(1)

      # y cloud
      cloud = (normalize(information_map[:, :, 1].copy()) * 255).astype(np.uint8)
      cloud = cv2.applyColorMap(cloud, cv2.COLORMAP_JET)
      cloud = cv2.resize(cloud, (0, 0), fx=3, fy=3)
      cv2.imshow("spherical y image", cloud)
      cv2.imwrite(args.vis_dir + "/" + "y.png", cloud)
      cv2.waitKey(1)

      # y cloud
      cloud = (normalize(information_map[:, :, 2].copy()) * 255).astype(np.uint8)
      cloud = cv2.applyColorMap(cloud, cv2.COLORMAP_JET)
      cloud = cv2.resize(cloud, (0, 0), fx=3, fy=3)
      cv2.imshow("spherical z image", cloud)
      cv2.imwrite(args.vis_dir + "/" + "z.png", cloud)
      cv2.waitKey(1)

      # intensity cloud
      cloud = (normalize(information_map[:, :, 3].copy()) * 255).astype(np.uint8)
      cloud = cv2.applyColorMap(cloud, cv2.COLORMAP_JET)
      cloud = cv2.resize(cloud, (0, 0), fx=3, fy=3)
      cv2.imshow("spherical intensity image", cloud)
      cv2.imwrite(args.vis_dir + "/" + "i.png", cloud)
      cv2.waitKey(1)

      # depth and normalize
      cloud = (normalize(information_map[:, :, 4].copy()) * 255).astype(np.uint8)
      cloud = cv2.applyColorMap(cloud, cv2.COLORMAP_JET)
      cloud = cv2.resize(cloud, (0, 0), fx=3, fy=3)
      cv2.imshow("spherical depth image", cloud)
      cv2.imwrite(args.vis_dir + "/" + "d.png", cloud)
      cv2.waitKey(1)

      # label cloud
      label_cloud = ((255 * mc.CLS_COLOR_MAP[information_map[:, :, 5].astype(np.uint8)]).astype(np.uint8))
      label_cloud = cv2.cvtColor(label_cloud, cv2.COLOR_RGB2BGR)
      label_cloud = cv2.resize(label_cloud, (0, 0), fx=3, fy=3)
      cv2.imshow("spherical label image", label_cloud)
      cv2.imwrite(args.vis_dir + "/" + "l.png", label_cloud)
      cv2.waitKey(1)

      # mask
      mask = (normalize(binary_mask_copy) * 255).astype(np.uint8)
      # mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
      mask = cv2.resize(mask, (0, 0), fx=3, fy=3)
      cv2.imshow("spherical mask image", mask)
      cv2.imwrite(args.vis_dir + "/" + "m.png", mask)
      cv2.waitKey(1)

    # write numpy tensor to file .npy
    if args.output_dir:
      filename = os.path.join(args.output_dir, os.path.basename(file).split(".")[0] + ".npy")
      print("Write tensor matrix to {0}".format(filename))
      print(np.unique(information_map[:, :, 5]))
      np.save(filename, information_map[:, :, :6])


  print("Absolute Class Distribution")
  print(class_distribution)
  sum = np.sum(list(class_distribution.values()))

  for k, v in class_distribution.items():
    class_distribution[k] = v / sum

  print("Normalized Class Distribution")
  print(class_distribution)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Parse flags for the manually annotated PCD to *.NPY conversion")
  parser.add_argument('-o', '--output_dir', type=str, help="Output dir for the *.npy files. If flag not defined, then"
                                                           "the numpy files are not stored.")
  parser.add_argument('-v', '--vis_dir', type=str, help="Output dir for the visualizations. If flag not defined, then"
                                                        "visualizations are not stored.")
  parser.add_argument('-i', '--input_dir', required=True, type=str, help="Input dir which contains *.pcd files, created"
                                                                         "by an labeling tool. These are the raw labled"
                                                                         "point clouds")
  args = parser.parse_args()
  main(args)
