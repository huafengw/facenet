#!/bin/bash
rm -rf triplet.zip
zip -r triplet.zip src

spark-submit \
--py-files triplet.zip \
src/vip_train_tripletloss_spark.py \
--model FACENET \
--local_data_path /home/mlp/training_data/vip_dresses \
--num_executor 10 \
--transfer_learning \
--classes_per_batch 300 \
--learning_rate 0.002 \
--image_size 299 \
--alpha 0.2 \
--embedding_size 128