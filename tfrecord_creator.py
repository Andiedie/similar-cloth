import tensorflow as tf
import os
import dataset_util
import numpy as np
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--sample', default=0, type=int, help='sample size')

_TRAIN_RATE = 0.8
rank = [
    "MEN/Denim",
    "MEN/Jackets_Vests",
    "MEN/Pants",
    "MEN/Shirts_Polos",
    "MEN/Shorts",
    "MEN/Suiting",
    "MEN/Sweaters",
    "MEN/Sweatshirts_Hoodies",
    'MEN/Tees_Tanks',
    "WOMEN/Blouses_Shirts",
    "WOMEN/Cardigans",
    "WOMEN/Denim",
    "WOMEN/Dresses",
    "WOMEN/Graphic_Tees",
    "WOMEN/Jackets_Coats",
    "WOMEN/Leggings",
    "WOMEN/Pants",
    "WOMEN/Rompers_Jumpsuits",
    "WOMEN/Shorts",
    "WOMEN/Skirts",
    "WOMEN/Sweaters",
    "WOMEN/Sweatshirts_Hoodies",
    "WOMEN/Tees_Tanks"
]

def make_example(filename, cloth_type, minx, miny, maxx, maxy, lv1, lx1, ly1, lv2, lx2, ly2, lv3, lx3, ly3, lv4, lx4, ly4, lv5, lx5, ly5, lv6, lx6, ly6, lv7, lx7, ly7, lv8, lx8, ly8):
    label = filename[4:filename.find('/id')]
    label = rank.index(label)
    filename = os.path.join('./data', filename)
    with tf.gfile.GFile(filename, 'rb') as fid:
        image = fid.read()
    return tf.train.Example(features=tf.train.Features(feature={
        'image/object/class/label': dataset_util.int64_feature(label),
        'image/imgdata': dataset_util.bytes_feature(image),

        'image/object/bbox/xmin': dataset_util.int64_feature(minx),
        'image/object/bbox/xmax': dataset_util.int64_feature(maxx),
        'image/object/bbox/ymin': dataset_util.int64_feature(miny),
        'image/object/bbox/ymax': dataset_util.int64_feature(maxy),

        'image/object/lmk/lv1': dataset_util.int64_feature(lv1),
        'image/object/lmk/lx1': dataset_util.int64_feature(lx1),
        'image/object/lmk/ly1': dataset_util.int64_feature(ly1),

        'image/object/lmk/lv2': dataset_util.int64_feature(lv2),
        'image/object/lmk/lx2': dataset_util.int64_feature(lx2),
        'image/object/lmk/ly2': dataset_util.int64_feature(ly2),

        'image/object/lmk/lv3': dataset_util.int64_feature(lv3),
        'image/object/lmk/lx3': dataset_util.int64_feature(lx3),
        'image/object/lmk/ly3': dataset_util.int64_feature(ly3),

        'image/object/lmk/lv4': dataset_util.int64_feature(lv4),
        'image/object/lmk/lx4': dataset_util.int64_feature(lx4),
        'image/object/lmk/ly4': dataset_util.int64_feature(ly4),

        'image/object/lmk/lv5': dataset_util.int64_feature(lv5),
        'image/object/lmk/lx5': dataset_util.int64_feature(lx5),
        'image/object/lmk/ly5': dataset_util.int64_feature(ly5),

        'image/object/lmk/lv6': dataset_util.int64_feature(lv6),
        'image/object/lmk/lx6': dataset_util.int64_feature(lx6),
        'image/object/lmk/ly6': dataset_util.int64_feature(ly6),

        'image/object/lmk/lv7': dataset_util.int64_feature(lv7),
        'image/object/lmk/lx7': dataset_util.int64_feature(lx7),
        'image/object/lmk/ly7': dataset_util.int64_feature(ly7),

        'image/object/lmk/lv8': dataset_util.int64_feature(lv8),
        'image/object/lmk/lx8': dataset_util.int64_feature(lx8),
        'image/object/lmk/ly8': dataset_util.int64_feature(ly8)
    }))

def main(argv):
    args = parser.parse_args(argv[1:])
    landmark_raw = []
    print('reading landmarks...')
    with open('./data/Anno/list_landmarks_inshop.txt') as f:
        next(f)
        next(f)
        for line in f:
            line = line.strip()
            items = list(filter(None, line.split(' ')))
            landmark_raw.append(items)
    print('reading bbox...')
    bbox_raw = np.loadtxt(
        './data/Anno/list_bbox_inshop.txt', skiprows=2, dtype='str')
    print('shuffling...')
    combined = list(zip(landmark_raw, bbox_raw))
    random.shuffle(combined)
    landmark_raw[:], bbox_raw[:] = zip(*combined)
    print('writting...')
    train_writer = tf.python_io.TFRecordWriter('./data/train.tfrecord')
    test_writer = tf.python_io.TFRecordWriter('./data/test.tfrecord')
    total_len = len(landmark_raw)
    for i in range(total_len):
        landmark = landmark_raw[i]
        bbox = bbox_raw[i]
        filename = landmark[0]
        cloth_type = int(landmark[1])
        minx = int(bbox[3])
        miny = int(bbox[4])
        maxx = int(bbox[5])
        maxy = int(bbox[6])
        if(cloth_type == 1):
            lv1 = int(landmark[3])
            lx1 = int(landmark[4])
            ly1 = int(landmark[5])

            lv2 = int(landmark[6])
            lx2 = int(landmark[7])
            ly2 = int(landmark[8])

            lv3 = int(landmark[9])
            lx3 = int(landmark[10])
            ly3 = int(landmark[11])

            lv4 = int(landmark[12])
            lx4 = int(landmark[13])
            ly4 = int(landmark[14])

            lv5 = -1
            lx5 = -1
            ly5 = -1

            lv6 = -1
            lx6 = -1
            ly6 = -1

            lv7 = int(landmark[15])
            lx7 = int(landmark[16])
            ly7 = int(landmark[17])

            lv8 = int(landmark[18])
            lx8 = int(landmark[19])
            ly8 = int(landmark[20])

        if(cloth_type == 2):
            lv1 = -1
            lx1 = -1
            ly1 = -1

            lv2 = -1
            lx2 = -1
            ly2 = -1

            lv3 = -1
            lx3 = -1
            ly3 = -1

            lv4 = -1
            lx4 = -1
            ly4 = -1

            lv5 = int(landmark[3])
            lx5 = int(landmark[4])
            ly5 = int(landmark[5])

            lv6 = int(landmark[6])
            lx6 = int(landmark[7])
            ly6 = int(landmark[8])

            lv7 = int(landmark[9])
            lx7 = int(landmark[10])
            ly7 = int(landmark[11])

            lv8 = int(landmark[12])
            lx8 = int(landmark[13])
            ly8 = int(landmark[14])
    
        if(cloth_type == 3):
            lv1 = int(landmark[3])
            lx1 = int(landmark[4])
            ly1 = int(landmark[5])

            lv2 = int(landmark[6])
            lx2 = int(landmark[7])
            ly2 = int(landmark[8])

            lv3 = int(landmark[9])
            lx3 = int(landmark[10])
            ly3 = int(landmark[11])

            lv4 = int(landmark[12])
            lx4 = int(landmark[13])
            ly4 = int(landmark[14])

            lv5 = int(landmark[15])
            lx5 = int(landmark[16])
            ly5 = int(landmark[17])

            lv6 = int(landmark[18])
            lx6 = int(landmark[19])
            ly6 = int(landmark[20])

            lv7 = int(landmark[21])
            lx7 = int(landmark[22])
            ly7 = int(landmark[23])

            lv8 = int(landmark[24])
            lx8 = int(landmark[25])
            ly8 = int(landmark[26])
        
        example = make_example(filename, cloth_type, minx, miny, maxx, maxy, lv1, lx1, ly1, lv2, lx2, ly2, lv3, lx3, ly3, lv4, lx4, ly4, lv5, lx5, ly5, lv6, lx6, ly6, lv7, lx7, ly7, lv8, lx8, ly8)

        if (args.sample > 0):
            if (i < args.sample):
                train_writer.write(example.SerializeToString())
                test_writer.write(example.SerializeToString())
            else:
                break
            continue

        if (i < total_len * _TRAIN_RATE):
            train_writer.write(example.SerializeToString())
        else:
            test_writer.write(example.SerializeToString())
        if (i % 1000 == 0):
            print(i, 'done')
    train_writer.close()
    test_writer.close()

if __name__ == '__main__':
    tf.app.run()
