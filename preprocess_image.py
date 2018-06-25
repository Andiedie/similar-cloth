import tensorflow as tf
import random

_BBOX_BOUNCE_RATE = 0.2
_IMAGE_SIZE = 256
_LOCAL_SIZE = 32
_NUM_CHANNELS = 3
_NUM_LANDMARK = 8


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


def rand(num):
    return tf.random_uniform(
        [1],
        minval=0,
        maxval=num+1,
        dtype=tf.int64
    )[0] - num


def bbox_bounce(bbox, is_training):
    zero_bound = tf.constant(0, dtype=tf.int64)
    max_bound = tf.constant(_IMAGE_SIZE, dtype=tf.int64)
    rate = tf.constant(_BBOX_BOUNCE_RATE, dtype=tf.float32)

    bbox_ymin = tf.maximum(zero_bound, bbox['ymin'])
    bbox_xmin = tf.maximum(zero_bound, bbox['xmin'])
    bbox_ymax = tf.minimum(max_bound, bbox['ymax'])
    bbox_xmax = tf.minimum(max_bound, bbox['xmax'])
    bbox_height = bbox_ymax - bbox_ymin
    bbox_width = bbox_xmax - bbox_xmin

    if (not is_training):
        return bbox_ymin, bbox_xmin, bbox_height, bbox_width

    vertical_bounce = tf.cast(
        tf.cast(bbox_height, dtype=tf.float32) * rate, dtype=tf.int64)
    horizontal_bounce = tf.cast(
        tf.cast(bbox_width, dtype=tf.float32) * rate, dtype=tf.int64)

    bbox_ymin = tf.maximum(zero_bound, bbox_ymin + rand(vertical_bounce))
    bbox_xmin = tf.maximum(zero_bound, bbox_xmin + rand(horizontal_bounce))
    bbox_ymax = tf.minimum(max_bound, bbox_ymax + rand(vertical_bounce))
    bbox_xmax = tf.minimum(max_bound, bbox_xmax + rand(horizontal_bounce))
    bbox_height = bbox_ymax - bbox_ymin
    bbox_width = bbox_xmax - bbox_xmin

    return bbox_ymin, bbox_xmin, bbox_height, bbox_width


def preprocess(buffer, is_training, no_lmk, bbox, landmarks):
    random_seed = random.randint(1, 10)
    origin_image = tf.reshape(tf.image.decode_jpeg(buffer), [
                              _IMAGE_SIZE, _IMAGE_SIZE, _NUM_CHANNELS])

    bbox_ymin, bbox_xmin, bbox_height, bbox_width = bbox_bounce(
        bbox, is_training)

    cropped_image = tf.image.crop_to_bounding_box(
        origin_image, bbox_ymin, bbox_xmin, bbox_height, bbox_width)
    cropped_image = aspect_preserving_resize(cropped_image, _IMAGE_SIZE)
    if (is_training):
        cropped_image = tf.image.random_flip_left_right(
            cropped_image, seed=random_seed)

    if no_lmk:
        return cropped_image

    images = []
    for i in range(_NUM_LANDMARK):
        try:
            lmk_ymin = landmarks[i]['height'] - int(_LOCAL_SIZE / 2)
            lmk_xmin = landmarks[i]['width'] - int(_LOCAL_SIZE / 2)
            lmk_box = {
                'ymin': lmk_ymin,
                'xmin': lmk_xmin,
                'ymax': lmk_ymin + _LOCAL_SIZE,
                'xmax': lmk_xmin + _LOCAL_SIZE
            }
            lmk_ymin, lmk_xmin, lmk_height, lmk_width = bbox_bounce(
                lmk_box, is_training)
            landmark_local = tf.image.crop_to_bounding_box(
                origin_image, lmk_ymin, lmk_xmin, lmk_height, lmk_width)
            landmark_local = aspect_preserving_resize(
                landmark_local, _IMAGE_SIZE)
            if (is_training):
                landmark_local = tf.image.random_flip_left_right(
                    landmark_local, seed=random_seed)
        except ValueError:
            landmark_local = tf.zeros(
                [_IMAGE_SIZE, _IMAGE_SIZE, _NUM_CHANNELS], tf.float32)
        images.append(landmark_local)

    image = tf.concat([cropped_image] + images, -1)
    return image
