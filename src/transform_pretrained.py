import tensorflow as tf
import tensorflow.contrib.slim as slim
from src import inception_resnet_v2
from src import facenet
from tensorflow.python.ops import data_flow_ops
from src import inception_preprocessing
import os

def transform(args, pretrained_ckpt, image_size, output_dir, bottleneck_size):
  with tf.Graph().as_default():
    learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')
    batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
    phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
    image_paths_placeholder = tf.placeholder(tf.string, shape=(None,3), name='image_paths')
    labels_placeholder = tf.placeholder(tf.int64, shape=(None,3), name='labels')
    input_queue = data_flow_ops.FIFOQueue(capacity=100000,
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

    with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope(weight_decay=args.weight_decay)):
      prelogits, _ = inception_resnet_v2.inception_resnet_v2(image_batch, num_classes=bottleneck_size, is_training=phase_train_placeholder)

    exclude = ['InceptionResnetV2/Logits', 'InceptionResnetV2/AuxLogits']
    variables_to_restore = slim.get_variables_to_restore(exclude=exclude)
    loader = tf.train.Saver(variables_to_restore)

    global_step = tf.train.get_or_create_global_step()

    embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
    # Split embeddings into anchor, positive and negative and calculate triplet loss
    anchor, positive, negative = tf.unstack(tf.reshape(embeddings, [-1, 3, args.embedding_size]), 3, 1)
    triplet_loss = facenet.triplet_loss(anchor, positive, negative, args.alpha)

    learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step,
                                               args.learning_rate_decay_epochs * args.epoch_size,
                                               args.learning_rate_decay_factor, staircase=True)
    tf.summary.scalar('learning_rate', learning_rate)
    # Calculate the total losses
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = tf.add_n([triplet_loss] + regularization_losses, name='total_loss')

    # Build a Graph that trains the model with one batch of examples and updates the model parameters
    train_layers = ['Logits', 'Conv2d_7b_1x1', 'Block8', 'Repeat_2', 'Mixed_7a']
    var_list = []
    for v in tf.global_variables():
      splits = v.name.split("/")
      if len(splits) > 2 and splits[1] in train_layers:
        var_list.append(v)
    facenet.train(total_loss, global_step, args.optimizer,
                             learning_rate, args.moving_average_decay, var_list)

    saver = tf.train.Saver(max_to_keep=3)
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())

      loader.restore(sess, pretrained_ckpt)

      checkpoint_path = os.path.join(output_dir, 'model-%s.ckpt' % bottleneck_size)
      saver.save(sess, checkpoint_path, write_meta_graph=False)
      metagraph_filename = os.path.join(output_dir, 'model-%s.meta' % bottleneck_size)
      saver.export_meta_graph(metagraph_filename)

if __name__ == '__main__':
  transform("/Users/huafengw/Downloads/inception_resnet_v2_2016_08_30.ckpt", 299, "/tmp", 128)