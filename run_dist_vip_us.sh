#!/bin/bash
rm -rf triplet.zip
zip -r triplet.zip src

spark-submit \
--py-files triplet.zip \
src/vip_train_tripletloss_spark.py \
--transfer_learning \
--epochs 1000 \
--model VIPUS \
--local_data_path /home/mlp/training_data/vip_dresses \
--num_executor 10 \
--transfer_learning \
--classes_per_batch 300 \
--learning_rate 0.002 \
--image_size 224 \
--embedding_size 2048 \
--alpha 0.4 \
--pretrained_ckpt hdfs://bipcluster/user/vincent.wang/tensorflow_model/upperbody_clothes/v0.2.1/model.ckpt
