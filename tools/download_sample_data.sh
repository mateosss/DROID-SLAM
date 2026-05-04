#!/bin/bash

mkdir -p data && cd data

wget https://www.eth3d.net/data/slam/datasets/sfm_bench_mono.zip
unzip sfm_bench_mono.zip
rm sfm_bench_mono.zip

wget https://vision.in.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_cabinet.tgz
tar -zxvf rgbd_dataset_freiburg3_cabinet.tgz
rm rgbd_dataset_freiburg3_cabinet.tgz

wget https://syncandshare.lrz.de/dl/fiBwLLQWbnAwg9r6Qn6uzb/MH_03_medium.zip
unzip MH_03_medium.zip
rm MH_03_medium.zip

cd ..
