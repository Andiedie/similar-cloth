import logging
LOG_FORMAT = "[%(asctime)s] [%(levelname)s] - %(message)s"
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)

logging.info('loading dependencies')
import os
import numpy as np
from PIL import Image
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

__dirname = os.path.dirname(__file__)
model_path = os.path.join(__dirname, './model')

import sys
sys.path.append(os.path.join(__dirname))

logging.info('loading database')
import database
filename = np.load(os.path.join(__dirname, './filename.npy'))

logging.info('loading graph')
import preprocess_image as pi
import main

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=True)
session_config = tf.ConfigProto(gpu_options=gpu_options)
run_config = tf.estimator.RunConfig().replace(session_config=session_config)
clf = tf.estimator.Estimator(
    model_fn=main.model_fn, model_dir=model_path, config=run_config,
    params={
        'resnet_size': 50,
        'data_format': None,
        'batch_size': 32,
        'multi_gpu': False,
        'version': 2,
    })

next_image = None
def gen():
    yield Image.new('RGB', (pi._IMAGE_SIZE, pi._IMAGE_SIZE), (0, 0, 0))
    while True:
        yield next_image


def input_fn(generator):
    return tf.data.Dataset.from_generator(generator, (tf.float32), (256, 256, 3)).batch(1)


result = clf.predict(lambda: input_fn(gen))

logging.info('loading session')
next(result)
logging.info('ready')

def process_image(image_path, ymin, xmin, ymax, xmax):
    global next_image
    im = Image.open(image_path)
    # crop to bbox
    im = im.crop((xmin, ymin, xmax, ymax))
    # aspect preserve resize
    width, height = im.size
    bigger = max(width, height)
    ratio = pi._IMAGE_SIZE / bigger
    width = int(ratio * width)
    height = int(ratio * height)
    im = im.resize((width, height))
    # pad to given size
    temp = Image.new('RGB', (pi._IMAGE_SIZE, pi._IMAGE_SIZE), (0, 0, 0))
    width_gap = (pi._IMAGE_SIZE - width) // 2
    height_gap = (pi._IMAGE_SIZE - height) // 2
    temp.paste(im, (width_gap, height_gap))
    next_image = temp


def similar_cloth(image_path, ymin, xmin, ymax, xmax, top=5, method='cosine'):
    """
    Get clothes similar to the given one from the database

    Args:
        image_path: The path of the input cloth image
        ymin: The ordinate of the upper left point of the bounding box
        xmin: The abscissa of the upper left point of the bounding box
        ymin: The ordinate of the lower right point of the bounding box
        ymin: The abscissa of the lower right point of the bounding box
        top: Number of the similar clothes, default to 5
        method: method to calculate distance, default to 'cosine'
    Returns:
        list of filenames of the most similar cloths, like
        [
            'img/WOMEN/Blouses_Shirts/id_00000001/02_1_front.jpg',
            'img/WOMEN/Blouses_Shirts/id_00000001/02_2_side.jpg'
        ]
    """
    process_image(image_path, ymin, xmin, ymax, xmax)
    vector = next(result)['logits']
    top = database.topN(vector, n=top, method=method)
    return filename[top]
