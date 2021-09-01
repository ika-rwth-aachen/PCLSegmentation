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
python3 src/pcl_segmentation/train.py \
--data_path="src/sample_dataset" \
--train_dir="src/output" \
--epochs=5 \
--model=squeezesegv2