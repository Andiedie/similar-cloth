import os
import sys
import tensorflow as tf
import resnet_model
import resnet_run_loop

_IMAGE_SIZE = 256
_LOCAL_SIZE = 32
_NUM_CHANNELS = 3
_NUM_CLASSES = 23
_NUM_LANDMARK = 8
_NUM_IMAGES = {
    'train': 0,
    'test': 0
}
_SHUFFLE_BUFFER = 1500


def get_filenames(is_training, data_dir):
    if is_training:
        return [os.path.join(data_dir, 'train.tfrecord')]
    else:
        return [os.path.join(data_dir, 'test.tfrecord')]


def aspect_preserving_resize(image, size):
    shape = tf.shape(image)
    height, width = shape[0], shape[1]
    height, width = tf.cast(height, tf.float32), tf.cast(width, tf.float32)
    bigger_dim = tf.maximum(height, width)
    scale_ratio = tf.cast(size, tf.float32) / bigger_dim
    new_height = tf.cast(height * scale_ratio, tf.int32)
    new_width = tf.cast(width * scale_ratio, tf.int32)
    image = tf.image.resize_images(image, [new_height, new_width])
    image = tf.image.resize_image_with_crop_or_pad(image, size, size)
    return image


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

    origin_image = tf.reshape(tf.image.decode_jpeg(
        image_buffer), [_IMAGE_SIZE, _IMAGE_SIZE, _NUM_CHANNELS])

    bbox_ymin = tf.maximum(tf.constant(0, dtype=tf.int64), bbox['ymin'])
    bbox_xmin = tf.maximum(tf.constant(0, dtype=tf.int64), bbox['xmin'])
    bbox_ymax = tf.minimum(tf.constant(_IMAGE_SIZE, dtype=tf.int64), bbox['ymax'])
    bbox_xmax = tf.minimum(tf.constant(_IMAGE_SIZE, dtype=tf.int64), bbox['xmax'])
    crop_window = tf.cast(
        [bbox_ymin, bbox_xmin, bbox_ymax - bbox_ymin, bbox_xmax - bbox_xmin], dtype=tf.int32)

    cropped_image = tf.image.decode_and_crop_jpeg(image_buffer, crop_window, _NUM_CHANNELS)

    cropped_image = aspect_preserving_resize(cropped_image, _IMAGE_SIZE)
    if (not no_lmk):
        return cropped_image, label

    images = []
    for i in range(_NUM_LANDMARK):
        try:
            lmk_ymin = tf.maximum(tf.constant(
                0, dtype=tf.int64), landmarks[i]['height'] - int(_LOCAL_SIZE / 2))
            lmk_xmin = tf.maximum(tf.constant(
                0, dtype=tf.int64), landmarks[i]['width'] - int(_LOCAL_SIZE / 2))
            lmk_ymax = tf.minimum(tf.constant(
                _IMAGE_SIZE, dtype=tf.int64), lmk_ymin + _LOCAL_SIZE)
            lmk_xmax = tf.minimum(tf.constant(
                _IMAGE_SIZE, dtype=tf.int64), lmk_xmin + _LOCAL_SIZE)
            landmark_local = tf.image.crop_to_bounding_box(
                origin_image, lmk_ymin, lmk_xmin, lmk_ymax - lmk_ymin, lmk_xmax - lmk_xmin)
            landmark_local = tf.image.resize_images(landmark_local, [_IMAGE_SIZE, _IMAGE_SIZE])
        except ValueError:
            landmark_local = tf.zeros(
                [_IMAGE_SIZE, _IMAGE_SIZE, _NUM_CHANNELS], tf.float32)
        images.append(landmark_local)

    image = tf.concat([cropped_image] + images, -1)
    return image, label


def input_fn(is_training, no_lmk, data_dir, batch_size, num_epochs=1, num_parallel_calls=1, multi_gpu=False):
    filenames = get_filenames(is_training, data_dir)
    dataset = tf.data.TFRecordDataset(
        filenames, num_parallel_reads=num_parallel_calls)

    num_images = is_training and _NUM_IMAGES['train'] or _NUM_IMAGES['test']

    return resnet_run_loop.process_record_dataset(dataset, is_training, no_lmk, batch_size, num_images, parse_record, num_epochs, num_parallel_calls, examples_per_epoch=num_images, multi_gpu=multi_gpu)


def get_synth_input_fn():
  return resnet_run_loop.get_synth_input_fn(_IMAGE_SIZE, _IMAGE_SIZE, _NUM_CHANNELS, _NUM_CLASSES)


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
    # use_synthetic_data = False

    input_function = flags.use_synthetic_data and get_synth_input_fn() or input_fn

    resnet_run_loop.resnet_main(flags, model_fn, input_function)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main(argv=sys.argv)
