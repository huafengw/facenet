import tensorflow as tf
import tensorflow.contrib.slim as slim
from inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope
import os

def transform(pretrained_ckpt, image_size, output_dir, bottleneck_size):
  with tf.Graph().as_default():
    image_batch = tf.placeholder(tf.float32, shape=(None, image_size, image_size, 3))
    with slim.arg_scope(inception_resnet_v2_arg_scope()):
      prelogits, _ = inception_resnet_v2(image_batch, num_classes=bottleneck_size)

    exclude = ['InceptionResnetV2/Logits', 'InceptionResnetV2/AuxLogits']
    variables_to_restore = slim.get_variables_to_restore(exclude=exclude)
    saver = tf.train.Saver(variables_to_restore)

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      saver.restore(sess, pretrained_ckpt)

      checkpoint_path = os.path.join(output_dir, 'model-%s.ckpt' % bottleneck_size)
      saver = tf.train.Saver()
      saver.save(sess, checkpoint_path)

if __name__ == '__main__':
  transform("/Users/huafengw/Downloads/inception_resnet_v2_2016_08_30.ckpt", 299, "/tmp", 128)