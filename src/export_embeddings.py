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

import inception_preprocessing
import os.path
import sys
import tensorflow as tf
import numpy as np
import argparse
import pickle


def preprocess(filepath, image_size):
  file_contents = tf.read_file(filepath)
  image = tf.image.decode_image(file_contents, channels=3)
  return inception_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)


def main(args):
  output_file = os.path.join(args.output_dir, 'result.pkl')
  print('Model checkpoint: %s' % args.ckpt_path)
  print('Test data directory: %s' % args.test_images_dir)
  print('Output file: %s' % output_file)

  image_paths = list(map(lambda name: os.path.join(args.test_images_dir, name), os.listdir(args.test_images_dir)))
  with tf.Graph().as_default():
    with tf.Session() as sess:
      ckpt_dir = args.ckpt_path.rsplit('/', 1)[0]
      meta_file = args.ckpt_path + ".meta"
      if not tf.gfile.Exists(meta_file):
        meta_files = tf.gfile.ListDirectory(ckpt_dir)
        meta_files = [s for s in meta_files if s.endswith('.meta')]
        assert len(meta_files) > 0
        meta_file = ckpt_dir + '/' + meta_files[0]

      saver = tf.train.import_meta_graph(meta_file, clear_devices=True)
      saver.restore(tf.get_default_session(), args.ckpt_path)
      # Placeholder for the learning rate

      images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
      embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
      phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

      nrof_images = len(image_paths)
      print('Number of images: ', nrof_images)
      batch_size = args.batch_size
      if nrof_images % batch_size == 0:
        nrof_batches = nrof_images // batch_size
      else:
        nrof_batches = (nrof_images // batch_size) + 1
      print('Number of batches: ', nrof_batches)
      embedding_size = embeddings.get_shape()[1]
      emb_array = np.zeros((nrof_images, embedding_size))

      dataset = tf.data.Dataset.from_tensor_slices(image_paths)
      dataset = dataset.map(lambda path: preprocess(path, args.image_size))
      batched_dataset = dataset.batch(batch_size)
      iterator = batched_dataset.make_one_shot_iterator()
      next_element = iterator.get_next()

      for i in range(nrof_batches):
        if i == nrof_batches - 1:
          n = nrof_images
        else:
          n = i * batch_size + batch_size
        images = sess.run(next_element)
        feed_dict = {images_placeholder: images, phase_train_placeholder: False}
        embed = sess.run(embeddings, feed_dict=feed_dict)
        emb_array[i * batch_size:n, :] = embed
      all_features = dict()
      for i in range(nrof_images):
        all_features[image_paths[i]] = emb_array[i].tolist()
      with open(output_file, 'wb') as fd:
        pickle.dump(all_features, fd)


def print_results(output_file):
  with open(output_file, 'rb') as fd:
    path_to_embeddings = pickle.load(fd)
    print(path_to_embeddings)


def parse_arguments(argv):
  parser = argparse.ArgumentParser()

  parser.add_argument('--ckpt_path', type=str,
                      help='The checkpoint file path.')
  parser.add_argument('--output_dir', type=str,
                      help='Output dir of embeddings.')
  parser.add_argument('--batch_size', type=int,
                      help='Number of images to process in a batch.', default=60)
  parser.add_argument('--image_size', type=int,
                      help='Image size (height, width) in pixels.', default=299)
  parser.add_argument('--embedding_size', type=int,
                      help='Dimensionality of the embedding.', default=128)
  parser.add_argument('--test_images_dir', type=str,
                      help='Path to the data directory containing aligned face patches.', default='')
  return parser.parse_args(argv)


if __name__ == '__main__':
  main(parse_arguments(sys.argv[1:]))
