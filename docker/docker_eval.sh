#!/bin/sh

DIR="$(cd -P "$(dirname "$0")" && pwd)"

docker run \
--gpus all \
--name='pcl_segmentation' \
--rm \
--tty \
--user "$(id -u):$(id -g)" \
--volume $DIR/../:/src \
pcl_segmentation \
python3 src/pcl_segmentation/eval.py \
--data_path="/src/sample_dataset" \
--eval_dir="/src/output/eval" \
--path_to_model="/src/output/model" \
--image_set="val" \
--model=squeezesegv2