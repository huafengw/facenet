#!/bin/bash
rm -rf triplet.zip
zip -r triplet.zip src

spark-submit \
--py-files triplet.zip \
src/vip_train_tripletloss_spark.py \
--num_executor 10 \
--transfer_learning \
--classes_per_batch 300 \
--learning_rate 0.002