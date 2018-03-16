from src import vgg_preprocessing
from src import resnet_v1
from src import inception_preprocessing
from src import inception_resnet_v2
from src import transform_pretrained
import tensorflow as tf

class Model(object):

  def preprecess_function(self):
    pass

  def inference_function(self):
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


  def inference_function(self):
    return resnet_v1.resnet_v1_101_triplet

  def tweak_pretrained_model(self, args, pretrained_ckpt, image_size, checkpoint_dir, embedding_size):
    pass

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

  def inference_function(self):
    return inception_resnet_v2.inception_resnet_v2

  def tweak_pretrained_model(self, args, pretrained_ckpt, image_size, checkpoint_dir, embedding_size):
    print("Transforming the pretrained inception model...")
    transform_pretrained.transform(args, pretrained_ckpt, image_size, checkpoint_dir, embedding_size)
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


def getModel(model_name):
  if model_name == 'FACENET':
    return InceptionImageNetModel()
  elif model_name == 'VIPUS':
    return VipUSModel()
  else:
    return None

