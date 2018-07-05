import tensorflow as tf
import os
import dataset_util
import numpy as np
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--sample', default=0, type=int, help='sample size')
parser.add_argument('--predict', default=None, nargs='+', type=str, help='specific image to predict, can be all or random')

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
    bbox_raw = np.loadtxt(
        './data/Anno/list_bbox_inshop.txt', skiprows=2, dtype='str')
    if args.predict is None:
        print('shuffling...')
        np.random.shuffle(bbox_raw)

    print('reading attributes...')
    attr = np.loadtxt('./data/Anno/list_attr_items.txt', skiprows=2, dtype=str)
    attr = np.unique(attr, axis=0)
    labels = attr[:,1:].astype(np.int)
    labels = np.vectorize(lambda x: 0 if x == -1 else 1)(labels)

    print('writting...')
    if args.predict is None:
        train_writer = tf.python_io.TFRecordWriter('./data/train.tfrecord')
        test_writer = tf.python_io.TFRecordWriter('./data/test.tfrecord')
    else:
        pred_writer = tf.python_io.TFRecordWriter('./data/predict.tfrecord')
        if args.predict[0] == 'random':
            args.predict = [random.choice(bbox_raw)[0]]
            print('random', args.predict)

    total_num = len(bbox_raw)
    for i in range(total_num):
        bbox = bbox_raw[i]
        filename = bbox[0]
        foundIndex = filename.find('id_') + 3
        foundIndex = int(filename[foundIndex:foundIndex+8]) - 1
        label = labels[foundIndex]
        xmin = int(bbox[3])
        ymin = int(bbox[4])
        xmax = int(bbox[5])
        ymax = int(bbox[6])
        if args.predict is not None:
            if args.predict[0] != 'all' and filename not in args.predict:
                continue

        example = make_example(filename, label, ymin, xmin, ymax, xmax)

        if (args.sample > 0):
            if (i < args.sample):
                train_writer.write(example.SerializeToString())
                test_writer.write(example.SerializeToString())
            else:
                break
            continue

        if args.predict is not None:
            pred_writer.write(example.SerializeToString())
        elif (i < total_num * _TRAIN_RATE):
            train_writer.write(example.SerializeToString())
        else:
            test_writer.write(example.SerializeToString())
        if (i % 1000 == 0):
            print(i, 'done')

    if args.predict is None:
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
