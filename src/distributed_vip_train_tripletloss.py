# MIT License
# 
# Copyright (c) 2016 David Sandberg
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import tensorflow.contrib.slim as slim
from src import transform_pretrained
from src import inception_preprocessing
from src.inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope
import os.path
import time
import tensorflow as tf
import numpy as np
import itertools
import random
from src import facenet

from tensorflow.python.ops import data_flow_ops

from six.moves import xrange


def read_pairs(pairs_filename):
  pairs = []
  with open(pairs_filename, 'r') as f:
    for line in f.readlines():
      pair = line.strip().split(",")
      pairs.append(pair)
  return np.array(pairs)


def get_paths(lfw_dir, pairs):
  nrof_skipped_pairs = 0
  path_list = []
  issame_list = []
  for pair in pairs:
    if len(pair) == 3:
      path0 = os.path.join(lfw_dir, pair[0], pair[1])
      path1 = os.path.join(lfw_dir, pair[0], pair[2])
      issame = True
    elif len(pair) == 4:
      path0 = os.path.join(lfw_dir, pair[0], pair[1])
      path1 = os.path.join(lfw_dir, pair[2], pair[3])
      issame = False
    if os.path.exists(path0) and os.path.exists(path1):  # Only add the pair if both paths exist
      path_list += (path0, path1)
      issame_list.append(issame)
    else:
      nrof_skipped_pairs += 1
  if nrof_skipped_pairs > 0:
    print('Skipped %d image pairs' % nrof_skipped_pairs)

  return path_list, issame_list


def _evaluate(embeddings, actual_issame, nrof_folds=10):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy = facenet.calculate_roc(thresholds, embeddings1, embeddings2,
                                               np.asarray(actual_issame), nrof_folds=nrof_folds)
    thresholds = np.arange(0, 4, 0.001)
    val, val_std, far = facenet.calculate_val(thresholds, embeddings1, embeddings2,
                                              np.asarray(actual_issame), 1e-3, nrof_folds=nrof_folds)
    return tpr, fpr, accuracy, val, val_std, far


def train(server, cluster_spec, args, ctx):
  import tarfile
  def download_images_to_local(path):
    assert tf.gfile.Exists(path)
    file_name = path.split("/")[-1]
    local_path = "/tmp/" + file_name
    if not os.path.exists(local_path):
      tf.gfile.Copy(path, "file://" + local_path)
    tar = tarfile.open(local_path, "r:")
    tar.extractall("/tmp")
    os.remove(local_path)

  if not os.path.exists("/tmp/dress"):
    download_images_to_local(args.input_data)

  task_index = ctx.task_index
  num_workers = len(cluster_spec['worker'])
  is_chief = task_index == 0

  data_dir = "/tmp/dress/train"
  val_dir = "/tmp/dress/val"
  val_pairs = "/tmp/dress/pairs.txt"
  subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
  log_dir = args.workspace + "/logs/" + subdir

  checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir else (args.workspace + "/models/" + subdir)

  if task_index == 0:
    if not tf.gfile.Exists(args.workspace):
      tf.gfile.MakeDirs(args.workspace)
    if not tf.gfile.Exists(checkpoint_dir):
      tf.gfile.MakeDirs(checkpoint_dir)
    if not tf.gfile.Exists(log_dir):
        tf.gfile.MakeDirs(log_dir)

  if is_chief and args.transfer_learning:
    print("Transforming the pretrained inception model...")
    transform_pretrained.transform(args, args.pretrained_ckpt, args.image_size, checkpoint_dir, args.embedding_size)
    print("Transform finished")
    tf.reset_default_graph()

  seed = random.SystemRandom().randint(0, 10240)
  print("Random seed: " + str(seed))
  np.random.seed(seed=seed)
  train_set = facenet.get_dataset(data_dir)

  print('Model directory: %s' % checkpoint_dir)
  print('Log directory: %s' % log_dir)

  # Read the file containing the pairs used for testing
  pairs = read_pairs(val_pairs)
  # Get the paths for the corresponding images
  val_image_paths, actual_issame = get_paths(val_dir, pairs)

  with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % task_index, cluster=cluster_spec)):
    tf.set_random_seed(seed)

    # Placeholder for the learning rate
    learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')
    batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
    phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
    image_paths_placeholder = tf.placeholder(tf.string, shape=(None, 3), name='image_paths')
    labels_placeholder = tf.placeholder(tf.int64, shape=(None, 3), name='labels')

    input_queue = data_flow_ops.FIFOQueue(capacity=10000,
                                          dtypes=[tf.string, tf.int64],
                                          shapes=[(3,), (3,)],
                                          shared_name=None, name=None)
    enqueue_op = input_queue.enqueue_many([image_paths_placeholder, labels_placeholder])

    nrof_preprocess_threads = 4
    images_and_labels = []
    for _ in range(nrof_preprocess_threads):
      filenames, label = input_queue.dequeue()
      images = []
      for filename in tf.unstack(filenames):
        file_contents = tf.read_file(filename)
        image = tf.image.decode_image(file_contents, channels=3)
        processed_image = inception_preprocessing.preprocess_image(image, args.image_size, args.image_size, is_training=False)
        # if args.random_crop:
        #     image = tf.random_crop(image, [args.image_size, args.image_size, 3])
        # else:
        #     image = tf.image.resize_image_with_crop_or_pad(image, args.image_size, args.image_size)
        if args.random_flip:
          processed_image = tf.image.random_flip_left_right(processed_image)

        images.append(processed_image)
      images_and_labels.append([images, label])

    image_batch, labels_batch = tf.train.batch_join(
      images_and_labels, batch_size=batch_size_placeholder,
      shapes=[(args.image_size, args.image_size, 3), ()], enqueue_many=True,
      capacity=4 * nrof_preprocess_threads * args.batch_size,
      allow_smaller_final_batch=True)
    image_batch = tf.identity(image_batch, 'image_batch')
    image_batch = tf.identity(image_batch, 'input')
    labels_batch = tf.identity(labels_batch, 'label_batch')

    with slim.arg_scope(inception_resnet_v2_arg_scope(weight_decay=args.weight_decay)):
      prelogits, _ = inception_resnet_v2(image_batch, num_classes=args.embedding_size,
                                         is_training=phase_train_placeholder)

    global_step = tf.train.get_or_create_global_step()

    embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
    # Split embeddings into anchor, positive and negative and calculate triplet loss
    anchor, positive, negative = tf.unstack(tf.reshape(embeddings, [-1, 3, args.embedding_size]), 3, 1)
    triplet_loss = facenet.triplet_loss(anchor, positive, negative, args.alpha)

    learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step,
                                               args.learning_rate_decay_epochs * args.epoch_size,
                                               args.learning_rate_decay_factor, staircase=True)
    # Calculate the total losses
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = tf.add_n([triplet_loss] + regularization_losses, name='total_loss')

    tf.summary.scalar('learning_rate', learning_rate)
    tf.summary.scalar('triplet_loss', triplet_loss)
    tf.summary.scalar('total_losses', total_loss)

    # Build a Graph that trains the model with one batch of examples and updates the model parameters
    train_layers = ['Logits', 'Conv2d_7b_1x1', 'Block8', 'Repeat_2', 'Mixed_7a']
    var_list = []
    for v in tf.global_variables():
      splits = v.name.split("/")
      if len(splits) > 2 and splits[1] in train_layers:
        var_list.append(v)
    train_op, opt = facenet.train(total_loss, global_step, args.optimizer,
                             learning_rate, args.moving_average_decay, var_list, sync_replicas=args.sync_replicas, replicas_to_aggregate=num_workers)

    summary_op = tf.summary.merge_all()
    
    hooks = [tf.train.StopAtStepHook(int(args.epochs) * int(args.epoch_size))]
    if args.sync_replicas:
      hooks += [opt.make_session_run_hook(is_chief)]

    sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False,
                                 device_filters=['/job:ps', '/job:worker/task:%d' % task_index])
    #sess_config.operation_timeout_in_ms=80000
    
    with tf.train.MonitoredTrainingSession(master=server.target,
                                           is_chief=is_chief,
                                           config=sess_config,
                                           checkpoint_dir=checkpoint_dir,
                                           hooks=hooks,
                                           save_summaries_steps=None,
                                           save_summaries_secs=None,
                                           stop_grace_period_secs=30,
                                           save_checkpoint_secs=600) as sess:
      # Training and validation loop
      summary_writer = tf.summary.FileWriter(log_dir, sess.graph) if is_chief else None

      while not sess.should_stop():
        # Train for one epoch
        step = _train(args, sess, train_set, image_paths_placeholder, labels_placeholder, labels_batch,
               batch_size_placeholder, learning_rate_placeholder, phase_train_placeholder, enqueue_op, input_queue,
               global_step, embeddings, total_loss, train_op, args.embedding_size, triplet_loss, summary_op, summary_writer)
        if is_chief:
          # checkpoint_path = os.path.join(checkpoint_dir, 'model-%s.ckpt' % "test")
          # saver.save(sess._sess._sess._sess._sess, checkpoint_path, global_step=step, write_meta_graph=False)
          evaluate(sess, val_image_paths, embeddings, labels_batch, image_paths_placeholder, labels_placeholder,
                   batch_size_placeholder, learning_rate_placeholder, phase_train_placeholder, enqueue_op,
                   actual_issame, args.batch_size, args.lfw_nrof_folds, step, summary_writer, args.embedding_size)

  return checkpoint_dir


def _train(args, sess, dataset, image_paths_placeholder, labels_placeholder, labels_batch,
           batch_size_placeholder, learning_rate_placeholder, phase_train_placeholder, enqueue_op, input_queue,
           global_step, embeddings, loss, train_op, embedding_size, triplet_loss, summary_op, summary_writer):
  batch_number = 0

  lr = args.learning_rate
  while batch_number < args.epoch_size:
    # Sample classes randomly from the dataset
    image_paths, num_per_class = sample_classes(dataset, args.classes_per_batch, args.images_per_class)
    #print('Running forward pass on sampled images: ', flush=True)
    start_time = time.time()
    nrof_examples = args.classes_per_batch * args.images_per_class
    labels_array = np.reshape(np.arange(nrof_examples), (-1, 3))
    image_paths_array = np.reshape(np.expand_dims(np.array(image_paths), 1), (-1, 3))
    sess.run(enqueue_op, feed_dict = {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})
    emb_array = np.zeros((nrof_examples, embedding_size))
    nrof_batches = int(np.ceil(nrof_examples / args.batch_size))
    for i in range(nrof_batches):
      batch_size = min(nrof_examples - i * args.batch_size, args.batch_size)
      emb, lab = sess.run([embeddings, labels_batch], feed_dict={batch_size_placeholder: batch_size,
                                                                 learning_rate_placeholder: lr,
                                                                 phase_train_placeholder: True})
      emb_array[lab, :] = emb
    #print('%.3f' % (time.time() - start_time))

    # Select triplets based on the embeddings
    print('Selecting suitable triplets for training')
    triplets, nrof_random_negs, nrof_triplets = select_triplets(emb_array, num_per_class,
                                                                image_paths, args.classes_per_batch, args.alpha)
    selection_time = time.time() - start_time
    print('(nrof_random_negs, nrof_triplets) = (%d, %d): time=%.3f seconds' %
          (nrof_random_negs, nrof_triplets, selection_time))

    # Perform training on the selected triplets
    nrof_batches = int(np.ceil(nrof_triplets * 3 / args.batch_size))
    triplet_paths = list(itertools.chain(*triplets))
    labels_array = np.reshape(np.arange(len(triplet_paths)), (-1, 3))
    triplet_paths_array = np.reshape(np.expand_dims(np.array(triplet_paths), 1), (-1, 3))
    sess.run(enqueue_op, {image_paths_placeholder: triplet_paths_array, labels_placeholder: labels_array})
    nrof_examples = len(triplet_paths)
    train_time = 0
    i = 0
    emb_array = np.zeros((nrof_examples, embedding_size))
    loss_array = np.zeros((nrof_triplets,))
    summary = tf.Summary()
    while i < nrof_batches:
      start_time = time.time()
      batch_size = min(nrof_examples - i * args.batch_size, args.batch_size)
      feed_dict = {batch_size_placeholder: batch_size, learning_rate_placeholder: lr, phase_train_placeholder: True}
      err, triplet_err, _, step, emb, lab = sess.run(
        [loss, triplet_loss, train_op, global_step, embeddings, labels_batch], feed_dict=feed_dict)
      emb_array[lab, :] = emb
      loss_array[i] = err
      duration = time.time() - start_time
      epoch = step // args.epoch_size
      #print('Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f' %
      #      (epoch, batch_number + 1, args.epoch_size, duration, err))
      batch_number += 1
      i += 1
      train_time += duration
      summary.value.add(tag='loss', simple_value=err)
      summary.value.add(tag='triplet_loss', simple_value=triplet_err)
    
    if summary_writer:
      summary_writer.add_summary(summary, step)
    
  return step


def evaluate(sess, image_paths, embeddings, labels_batch, image_paths_placeholder, labels_placeholder,
             batch_size_placeholder, learning_rate_placeholder, phase_train_placeholder, enqueue_op, actual_issame,
             batch_size, nrof_folds, step, summary_writer, embedding_size):
  start_time = time.time()
  # Run forward pass to calculate embeddings
  print('Running forward pass on LFW images: ', end='')

  nrof_images = len(actual_issame) * 2
  assert (len(image_paths) == nrof_images)
  labels_array = np.reshape(np.arange(nrof_images), (-1, 3))
  image_paths_array = np.reshape(np.expand_dims(np.array(image_paths), 1), (-1, 3))
  sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})
  emb_array = np.zeros((nrof_images, embedding_size))
  nrof_batches = int(np.ceil(nrof_images / batch_size))
  label_check_array = np.zeros((nrof_images,))
  summary = tf.Summary()
  for i in xrange(nrof_batches):
    batch_size = min(nrof_images - i * batch_size, batch_size)
    emb, lab = sess.run([embeddings, labels_batch], feed_dict={batch_size_placeholder: batch_size,
                                                               learning_rate_placeholder: 0.0,
                                                               phase_train_placeholder: False})
    emb_array[lab, :] = emb
    label_check_array[lab] = 1
  print('%.3f' % (time.time() - start_time))

  assert (np.all(label_check_array == 1))

  _, _, accuracy, val, val_std, far = _evaluate(emb_array, actual_issame, nrof_folds=nrof_folds)

  print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
  print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
  lfw_time = time.time() - start_time
  # Add validation loss and accuracy to summary
  # pylint: disable=maybe-no-member
  summary.value.add(tag='lfw/accuracy', simple_value=np.mean(accuracy))
  summary.value.add(tag='lfw/val_rate', simple_value=val)
  summary.value.add(tag='time/lfw', simple_value=lfw_time)  
  summary_writer.add_summary(summary, step)


def select_triplets(embeddings, nrof_images_per_class, image_paths, classes_per_batch, alpha):
  """ Select the triplets for training
  """
  trip_idx = 0
  emb_start_idx = 0
  num_trips = 0
  triplets = []

  # VGG Face: Choosing good triplets is crucial and should strike a balance between
  #  selecting informative (i.e. challenging) examples and swamping training with examples that
  #  are too hard. This is achieve by extending each pair (a, p) to a triplet (a, p, n) by sampling
  #  the image n at random, but only between the ones that violate the triplet loss margin. The
  #  latter is a form of hard-negative mining, but it is not as aggressive (and much cheaper) than
  #  choosing the maximally violating example, as often done in structured output learning.

  for i in xrange(classes_per_batch):
    nrof_images = int(nrof_images_per_class[i])
    for j in xrange(1, nrof_images):
      a_idx = emb_start_idx + j - 1
      neg_dists_sqr = np.sum(np.square(embeddings[a_idx] - embeddings), 1)
      for pair in xrange(j, nrof_images):  # For every possible positive pair.
        p_idx = emb_start_idx + pair
        pos_dist_sqr = np.sum(np.square(embeddings[a_idx] - embeddings[p_idx]))
        neg_dists_sqr[emb_start_idx:emb_start_idx + nrof_images] = np.NaN
        # all_neg = np.where(np.logical_and(neg_dists_sqr-pos_dist_sqr<alpha, pos_dist_sqr<neg_dists_sqr))[0]  # FaceNet selection
        all_neg = np.where(neg_dists_sqr - pos_dist_sqr < alpha)[0]  # VGG Face selecction
        nrof_random_negs = all_neg.shape[0]
        if nrof_random_negs > 0:
          rnd_idx = np.random.randint(nrof_random_negs)
          n_idx = all_neg[rnd_idx]
          triplets.append((image_paths[a_idx], image_paths[p_idx], image_paths[n_idx]))
          # print('Triplet %d: (%d, %d, %d), pos_dist=%2.6f, neg_dist=%2.6f (%d, %d, %d, %d, %d)' %
          #    (trip_idx, a_idx, p_idx, n_idx, pos_dist_sqr, neg_dists_sqr[n_idx], nrof_random_negs, rnd_idx, i, j, emb_start_idx))
          trip_idx += 1

        num_trips += 1

    emb_start_idx += nrof_images

  np.random.shuffle(triplets)
  return triplets, num_trips, len(triplets)


def sample_classes(dataset, classes_per_batch, images_per_class):
  nrof_images = classes_per_batch * images_per_class

  # Sample classes from the dataset
  nrof_classes = len(dataset)
  class_indices = np.arange(nrof_classes)
  np.random.shuffle(class_indices)

  i = 0
  image_paths = []
  num_per_class = []
  sampled_class_indices = []
  # Sample images from these classes until we have enough
  while len(image_paths) < nrof_images:
    class_index = class_indices[i]
    nrof_images_in_class = len(dataset[class_index])
    image_indices = np.arange(nrof_images_in_class)
    np.random.shuffle(image_indices)
    nrof_images_from_class = min(nrof_images_in_class, images_per_class, nrof_images - len(image_paths))
    idx = image_indices[0:nrof_images_from_class]
    image_paths_for_class = [dataset[class_index].image_paths[j] for j in idx]
    sampled_class_indices += [class_index] * nrof_images_from_class
    image_paths += image_paths_for_class
    num_per_class.append(nrof_images_from_class)
    i += 1

  return image_paths, num_per_class


def get_learning_rate_from_file(filename, epoch):
  with open(filename, 'r') as f:
    for line in f.readlines():
      line = line.split('#', 1)[0]
      if line:
        par = line.strip().split(':')
        e = int(par[0])
        lr = float(par[1])
        if e <= epoch:
          learning_rate = lr
        else:
          return learning_rate

