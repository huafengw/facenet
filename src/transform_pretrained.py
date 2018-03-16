import tensorflow as tf
import tensorflow.contrib.slim as slim
from src import inception_resnet_v2
from src import facenet
from src import model_factory
import os

def transform(args, pretrained_ckpt, image_size, output_dir, bottleneck_size):
  with tf.Graph().as_default():
    model = model_factory.InceptionImageNetModel()
    learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')
    phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
    image_batch = tf.placeholder(tf.float32, shape=(None, image_size, image_size, 3))

    with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope(weight_decay=args.weight_decay)):
      prelogits, _ = inception_resnet_v2.inception_resnet_v2(image_batch, num_classes=bottleneck_size, is_training=phase_train_placeholder)

    exclude = ['InceptionResnetV2/Logits', 'InceptionResnetV2/AuxLogits']
    variables_to_restore = slim.get_variables_to_restore(exclude=exclude)
    loader = tf.train.Saver(variables_to_restore)

    global_step = tf.train.get_or_create_global_step()

    embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
    anchor, positive, negative = tf.unstack(tf.reshape(embeddings, [-1, 3, args.embedding_size]), 3, 1)
    triplet_loss = facenet.triplet_loss(anchor, positive, negative, args.alpha)

    learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step,
                                               args.learning_rate_decay_epochs * args.epoch_size,
                                               args.learning_rate_decay_factor, staircase=True)
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = tf.add_n([triplet_loss] + regularization_losses, name='total_loss')

    var_list = model.filter_variables_to_train(tf.global_variables())
    facenet.train(total_loss, global_step, args.optimizer, learning_rate, args.moving_average_decay, var_list)

    saver = tf.train.Saver()
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())

      loader.restore(sess, pretrained_ckpt)

      checkpoint_path = os.path.join(output_dir, 'model-%s.ckpt' % bottleneck_size)
      saver.save(sess, checkpoint_path, write_meta_graph=False)
      metagraph_filename = os.path.join(output_dir, 'model-%s.meta' % bottleneck_size)
      saver.export_meta_graph(metagraph_filename)

if __name__ == '__main__':
  pass