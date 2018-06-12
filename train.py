import os
import sys
import tensorflow as tf
import resnet_model
import resnet_run_loop
import preprocess_image as pi

_NUM_CLASSES = 23
_NUM_LANDMARK = 8
_NUM_IMAGES = {
    'train': 0,
    'test': 0
}

def get_filenames(is_training, data_dir):
    if is_training:
        return [os.path.join(data_dir, 'train.tfrecord')]
    else:
        return [os.path.join(data_dir, 'test.tfrecord')]


def parse_record(raw_record, is_training, no_lmk):
    feature_map = {
        'image/imgdata': tf.FixedLenFeature([], dtype=tf.string),
        'image/object/class/label': tf.FixedLenFeature([], dtype=tf.int64),
        'image/object/bbox/xmin': tf.FixedLenFeature([], dtype=tf.int64),
        'image/object/bbox/ymin': tf.FixedLenFeature([], dtype=tf.int64),
        'image/object/bbox/xmax': tf.FixedLenFeature([], dtype=tf.int64),
        'image/object/bbox/ymax': tf.FixedLenFeature([], dtype=tf.int64),
        'image/object/lmk/lx1': tf.FixedLenFeature([], dtype=tf.int64),
        'image/object/lmk/ly1': tf.FixedLenFeature([], dtype=tf.int64),
        'image/object/lmk/lx2': tf.FixedLenFeature([], dtype=tf.int64),
        'image/object/lmk/ly2': tf.FixedLenFeature([], dtype=tf.int64),
        'image/object/lmk/lx3': tf.FixedLenFeature([], dtype=tf.int64),
        'image/object/lmk/ly3': tf.FixedLenFeature([], dtype=tf.int64),
        'image/object/lmk/lx4': tf.FixedLenFeature([], dtype=tf.int64),
        'image/object/lmk/ly4': tf.FixedLenFeature([], dtype=tf.int64),
        'image/object/lmk/lx5': tf.FixedLenFeature([], dtype=tf.int64),
        'image/object/lmk/ly5': tf.FixedLenFeature([], dtype=tf.int64),
        'image/object/lmk/lx6': tf.FixedLenFeature([], dtype=tf.int64),
        'image/object/lmk/ly6': tf.FixedLenFeature([], dtype=tf.int64),
        'image/object/lmk/lx7': tf.FixedLenFeature([], dtype=tf.int64),
        'image/object/lmk/ly7': tf.FixedLenFeature([], dtype=tf.int64),
        'image/object/lmk/lx8': tf.FixedLenFeature([], dtype=tf.int64),
        'image/object/lmk/ly8': tf.FixedLenFeature([], dtype=tf.int64)
    }
    features = tf.parse_single_example(raw_record, feature_map)
    bbox = {
        'ymin': features['image/object/bbox/ymin'],
        'xmin': features['image/object/bbox/xmin'],
        'ymax': features['image/object/bbox/ymax'],
        'xmax': features['image/object/bbox/xmax']
    }

    landmarks = [
        {
            'height': features['image/object/lmk/ly%d' % (i + 1)],
            'width': features['image/object/lmk/lx%d' % (i + 1)]
        } for i in range(_NUM_LANDMARK)
    ]
    image_buffer = features['image/imgdata']
    label = tf.one_hot(features['image/object/class/label'], _NUM_CLASSES)

    image = pi.preprocess(image_buffer, is_training, no_lmk, bbox, landmarks)

    return image, label


def input_fn(is_training, no_lmk, data_dir, batch_size, num_epochs=1, num_parallel_calls=1, multi_gpu=False):
    filenames = get_filenames(is_training, data_dir)
    dataset = tf.data.TFRecordDataset(
        filenames, num_parallel_reads=num_parallel_calls)

    num_images = is_training and _NUM_IMAGES['train'] or _NUM_IMAGES['test']

    return resnet_run_loop.process_record_dataset(dataset, is_training, no_lmk, batch_size, num_images, parse_record, num_epochs, num_parallel_calls, examples_per_epoch=num_images, multi_gpu=multi_gpu)


class Model(resnet_model.Model):
    def __init__(self, resnet_size, data_format=None, num_classes=_NUM_CLASSES, version=resnet_model.DEFAULT_VERSION):
        if resnet_size < 50:
            bottleneck = False
            final_size = 512
        else:
            bottleneck = True
            final_size = 2048
        super(Model, self).__init__(
            resnet_size=resnet_size,
            bottleneck=bottleneck,
            num_classes=num_classes,
            num_filters=64,
            kernel_size=7,
            conv_stride=2,
            first_pool_size=3,
            first_pool_stride=2,
            second_pool_size=7,
            second_pool_stride=1,
            block_sizes=_get_block_sizes(resnet_size),
            block_strides=[1, 2, 2, 2],
            final_size=final_size,
            version=version,
            data_format=data_format)


def _get_block_sizes(resnet_size):
    choices = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
        200: [3, 24, 36, 3]
    }
    try:
        return choices[resnet_size]
    except KeyError:
        err = ('Could not find layers for selected Resnet size.\n'
            'Size received: {}; sizes allowed: {}.'.format(resnet_size, choices.keys()))
        raise ValueError(err)


def model_fn(features, labels, mode, params):
    learning_rate_fn = resnet_run_loop.learning_rate_with_decay(
        batch_size=params['batch_size'], batch_denom=256,
        num_images=_NUM_IMAGES['train'], boundary_epochs=[30, 60, 80, 90],
        decay_rates=[1, 0.1, 0.01, 0.001, 1e-4])
    return resnet_run_loop.resnet_model_fn(features, labels, mode, Model,
                                           resnet_size=params['resnet_size'],
                                           weight_decay=1e-4,
                                           learning_rate_fn=learning_rate_fn,
                                           momentum=0.9,
                                           data_format=params['data_format'],
                                           version=params['version'],
                                           loss_filter_fn=None,
                                           multi_gpu=params['multi_gpu'])


def main(argv):
    parser = resnet_run_loop.ResnetArgParser(
        resnet_size_choices=[18, 34, 50, 101, 152, 200])

    parser.set_defaults(
        train_epochs=100,
        data_dir='./data'
    )

    flags = parser.parse_args(args=argv[1:])

    flags.model_dir = './no-lmk-model' if flags.no_lmk else './lmk-model'

    _NUM_IMAGES['train'] = sum(1 for _ in tf.python_io.tf_record_iterator(get_filenames(True, flags.data_dir)[0]))
    _NUM_IMAGES['test'] = sum(1 for _ in tf.python_io.tf_record_iterator(get_filenames(False, flags.data_dir)[0]))

    # batch_size=32
    # no-lmk = False
    # data_dir = './data',
    # model_dir = './lmk-model'
    # resnet_size = 50
    # version = 2
    # train_epochs = 100
    # epochs_between_evals = 1
    # max_train_steps = None

    resnet_run_loop.resnet_main(flags, model_fn, input_fn)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main(argv=sys.argv)
