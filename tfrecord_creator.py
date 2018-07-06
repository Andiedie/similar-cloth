import os
import logging
import argparse
import numpy as np
import dataset_util
import tensorflow as tf

LOG_FORMAT = "[%(asctime)s] [%(levelname)s] - %(message)s"
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)

parser = argparse.ArgumentParser()
parser.add_argument('--predict', default=False, type=bool, help='write predict.tfrecord')

_TRAIN_RATE = 0.85


def make_example(filename, label, ymin, xmin, ymax, xmax):
    filename = os.path.join('./data', filename)
    with tf.gfile.GFile(filename, 'rb') as fid:
        image = fid.read()
    example = tf.train.SequenceExample()
    # pylint: disable=E1101
    example.context.feature['image'].bytes_list.value.append(image)
    example.context.feature['ymin'].int64_list.value.append(ymin)
    example.context.feature['xmin'].int64_list.value.append(xmin)
    example.context.feature['ymax'].int64_list.value.append(ymax)
    example.context.feature['xmax'].int64_list.value.append(xmax)
    labels = example.feature_lists.feature_list['label']
    # pylint: enable=E1101
    for i in label:
        labels.feature.add().int64_list.value.append(i)
    return example


def main(argv):
    args = parser.parse_args(argv[1:])

    print('reading bbox...')
    bbox_raw = np.loadtxt('./data/Anno/list_bbox_inshop.txt', skiprows=2, dtype='str')

    print('reading attributes...')
    attr = np.loadtxt('./data/Anno/list_attr_items.txt', skiprows=2, dtype=str)
    attr = np.unique(attr, axis=0)
    labels = attr[:,1:].astype(np.int)
    labels = np.vectorize(lambda x: 0 if x == -1 else 1)(labels)

    if not args.predict:
        print('shuffling...')
        np.random.shuffle(bbox_raw)
    
    print('writting...')
    if args.predict is None:
        train_writer = tf.python_io.TFRecordWriter('./data/train.tfrecord')
        test_writer = tf.python_io.TFRecordWriter('./data/test.tfrecord')
    else:
        pred_writer = tf.python_io.TFRecordWriter('./data/predict.tfrecord')

    total_num = len(bbox_raw)
    for i in range(total_num):
        if args.predict:
            writer = pred_writer
        elif i < total_num * _TRAIN_RATE:
            writer = train_writer
        else:
            writer = test_writer

        bbox = bbox_raw[i]
        filename = bbox[0]
        foundIndex = filename.find('id_') + 3
        foundIndex = int(filename[foundIndex:foundIndex+8]) - 1
        label = labels[foundIndex]
        xmin = int(bbox[3])
        ymin = int(bbox[4])
        xmax = int(bbox[5])
        ymax = int(bbox[6])

        example = make_example(filename, label, ymin, xmin, ymax, xmax)

        writer.write(example.SerializeToString())
        if (i % 1000 == 0):
            print(i, 'done')

    if not args.predict:
        train_writer.close()
        test_writer.close()
        train_num = sum(1 for _ in tf.python_io.tf_record_iterator(
            './data/train.tfrecord'))
        test_num = sum(1 for _ in tf.python_io.tf_record_iterator(
            './data/test.tfrecord'))
        print('train data number:', train_num)
        print('test data number:', test_num)
    else:
        pred_writer.close()
        pred_num = sum(1 for _ in tf.python_io.tf_record_iterator(
            './data/predict.tfrecord'))
        print('predict data number:', pred_num)
    


if __name__ == '__main__':
    tf.app.run()
