from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pyspark.context import SparkContext
from pyspark.conf import SparkConf
from tensorflowonspark import TFCluster, TFNode
from datetime import datetime
import os

def main_fun(argv, ctx):
  from src import distributed_vip_train_tripletloss
  import sys

  job_name = ctx.job_name
  assert job_name in ['ps', 'worker'], 'job_name must be ps or worker'
  print("argv:", argv)
  sys.argv = argv

  cluster_spec, server = TFNode.start_cluster_server(ctx, num_gpus=1)
  if job_name == 'ps':
    server.join()
  else:
    distributed_vip_train_tripletloss.train(server, cluster_spec, argv, ctx)


if __name__ == '__main__':
  # parse arguments needed by the Spark driver
  import argparse
  from src import transform_pretrained
  parser = argparse.ArgumentParser()
  parser.add_argument("--epochs", help="number of epochs", type=int, default=200)
  parser.add_argument("--start_from_scratch", help="Start training from pretrained inception model", action="store_true")
  parser.add_argument("--input_data", help="HDFS path to input dataset")
  parser.add_argument('--num_worker', default=2, type=int, help='The worker num')
  parser.add_argument("--tensorboard", help="launch tensorboard process", action="store_true")
  parser.add_argument("--pretrained_ckpt", help="The pretrained inception model", default='hdfs://hdfs-server/home/mlp/vincent/facenet')

  parser.add_argument('--workspace', type=str,
        help='Directory where to write event logs and checkpoints on hdfs.', default='hdfs://hdfs-server/home/mlp/vincent/facenet')
  parser.add_argument('--weight_decay', type=float,
        help='L2 weight regularization.', default=0.0)
  parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=60)
  parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=299)
  parser.add_argument('--people_per_batch', type=int,
        help='Number of people per batch.', default=45)
  parser.add_argument('--images_per_person', type=int,
        help='Number of images per person.', default=3)
  parser.add_argument('--epoch_size', type=int,
        help='Number of batches per epoch.', default=100)
  parser.add_argument('--alpha', type=float,
        help='Positive to negative triplet distance margin.', default=0.2)
  parser.add_argument('--embedding_size', type=int,
        help='Dimensionality of the embedding.', default=128)
  parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],
        help='The optimization algorithm to use', default='ADAGRAD')
  parser.add_argument('--learning_rate', type=float,
        help='Initial learning rate. If set to a negative value a learning rate ' +
        'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.0002)
  parser.add_argument('--learning_rate_decay_epochs', type=int,
        help='Number of epochs between learning rate decay.', default=100)
  parser.add_argument('--learning_rate_decay_factor', type=float,
        help='Learning rate decay factor.', default=1.0)
  parser.add_argument('--moving_average_decay', type=float,
        help='Exponential decay for tracking of training parameters.', default=0.9999)
  parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)

  args = parser.parse_args()

  checkpoint_dir = args.workspace + "/models"
  if args.start_from_scratch:
    print("Transforming the pretrained inception model...")
    transform_pretrained.transform(args.pretrained_ckpt, args.image_size, checkpoint_dir, args.embedding_size)

  conf = SparkConf() \
    .setAppName("triplet_distributed_train") \
    .set("spark.dynamicAllocation.enabled", "false") \
    .set("spark.shuffle.service.enabled", "false") \
    .setExecutorEnv("JAVA_HOME", os.environ["JAVA_HOME"]) \
    .setExecutorEnv("HADOOP_HDFS_HOME", os.environ["HADOOP_HOME"]) \
    .setExecutorEnv("LD_LIBRARY_PATH", os.environ["JAVA_HOME"] + "/jre/lib/amd64/server:" + os.environ["HADOOP_HOME"] + "/lib/native" ) \
    .set("hostbalance_shuffle","true")

  print("{0} ===== Start".format(datetime.now().isoformat()))
  sc = SparkContext(conf = conf)
  num_executors = int(args.num_worker)
  num_ps = 1

  cluster = TFCluster.run(sc, main_fun, args, num_executors, num_ps, args.tensorboard, TFCluster.InputMode.TENSORFLOW)
  cluster.shutdown()
  print("{0} ===== Stop".format(datetime.now().isoformat()))

