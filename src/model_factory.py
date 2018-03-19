import tensorflow.contrib.slim as slim
from src import vgg_preprocessing
from src import resnet_v1
from src import inception_preprocessing
from src import inception_resnet_v2
from src import facenet
import tensorflow as tf
import os

class Model(object):

  def preprecess_function(self):
    pass

  def inference(self, image_batch, embedding_size, phase_train_placeholder):
    pass

  def filter_variables_to_train(self, variables):
    pass

  def tweak_pretrained_model(self, args, pretrained_ckpt, image_size, checkpoint_dir, embedding_size):
    pass

  def arg_scorp_function(self):
    pass


class VipUSModel(Model):

  def preprecess_function(self):
    def preprocessing_fn(image, output_height, output_width, is_training):
        return vgg_preprocessing.preprocess_image(
            image, output_height, output_width, is_training=is_training, bgr=True)
    return preprocessing_fn


  def arg_scorp_function(self):
    return resnet_v1.resnet_arg_scope


  def inference(self, image_batch, embedding_size, phase_train_placeholder):
    return resnet_v1.resnet_v1_101_triplet(image_batch, embedding_size=embedding_size, is_training=phase_train_placeholder)


  def tweak_pretrained_model(self, args, pretrained_ckpt, image_size, checkpoint_dir, embedding_size):
    parent_dir = pretrained_ckpt.rsplit('/', 1)[0]
    files = tf.gfile.ListDirectory(parent_dir)
    for file in files:
      tf.gfile.Copy(parent_dir + '/' + file, checkpoint_dir + "/" + file)


  def filter_variables_to_train(self, variables):
    train_layers = ['logits', 'mutli_task']
    var_list = []
    for v in variables:
      splits = v.name.split("/")
      if len(splits) > 2 and splits[1] in train_layers:
        var_list.append(v)
    return var_list


class InceptionImageNetModel(Model):

  def preprecess_function(self):
    return inception_preprocessing.preprocess_image

  def arg_scorp_function(self):
    return inception_resnet_v2.inception_resnet_v2_arg_scope

  def inference(self, image_batch, embedding_size, phase_train_placeholder):
    return inception_resnet_v2.inception_resnet_v2(image_batch, num_classes=embedding_size, is_training=phase_train_placeholder)

  def tweak_pretrained_model(self, args, pretrained_ckpt, image_size, checkpoint_dir, embedding_size):
    print("Transforming the pretrained inception model...")
    self.transform(args, pretrained_ckpt, image_size, checkpoint_dir, embedding_size)
    print("Transform finished")
    tf.reset_default_graph()

  def filter_variables_to_train(self, variables):
    train_layers = ['Logits', 'Conv2d_7b_1x1', 'Block8', 'Repeat_2', 'Mixed_7a']
    var_list = []
    for v in variables:
      splits = v.name.split("/")
      if len(splits) > 2 and splits[1] in train_layers:
        var_list.append(v)
    return var_list


  def transform(self, args, pretrained_ckpt, image_size, output_dir, bottleneck_size):
    with tf.Graph().as_default():
      learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')
      phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
      image_batch = tf.placeholder(tf.float32, shape=(None, image_size, image_size, 3))

      with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope(weight_decay=args.weight_decay)):
        prelogits, _ = inception_resnet_v2.inception_resnet_v2(image_batch, num_classes=bottleneck_size,
                                                               is_training=phase_train_placeholder)

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

      var_list = self.filter_variables_to_train(tf.global_variables())
      facenet.train(total_loss, global_step, args.optimizer, learning_rate, args.moving_average_decay, var_list)

      saver = tf.train.Saver()
      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        loader.restore(sess, pretrained_ckpt)

        checkpoint_path = os.path.join(output_dir, 'model-%s.ckpt' % bottleneck_size)
        saver.save(sess, checkpoint_path, write_meta_graph=False)
        metagraph_filename = os.path.join(output_dir, 'model-%s.meta' % bottleneck_size)
        saver.export_meta_graph(metagraph_filename)


def getModel(model_name):
  if model_name == 'FACENET':
    return InceptionImageNetModel()
  elif model_name == 'VIPUS':
    return VipUSModel()
  else:
    return None

