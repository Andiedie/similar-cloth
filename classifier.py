import os
import main
import pickle
import database
import tensorflow as tf
import preprocess_image as pi

tf.logging.set_verbosity(tf.logging.ERROR)

__dirname = os.path.dirname(__file__)
model_path = os.path.join(__dirname, './model')
filename = pickle.load(open(os.path.join(__dirname, './filename.pickle'), 'rb'))

# 只在预测时占用50%的显存
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
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


def _input_fn(image_path, ymin, xmin, ymax, xmax):
    with open(image_path, 'rb') as f:
        image_buffer = f.read()
    image = pi.preprocess(image_buffer, False, {
        'ymin': ymin,
        'xmin': xmin,
        'ymax': ymax,
        'xmax': xmax
    })
    dataset = tf.data.Dataset.from_tensor_slices([image]).batch(1)
    return dataset


def _input_fn_v2(cropped):
    image = pi.aspect_preserving_resize(cropped, pi._IMAGE_SIZE)
    dataset = tf.data.Dataset.from_tensor_slices([image]).batch(1)
    return dataset


def similar_cloth(image_path, ymin, xmin, ymax, xmax, top=5, method='cos'):
    """
    Get clothes similar to the given one from the database

    Args:
        image_path: The path of the input cloth image
        ymin: The ordinate of the upper left point of the bounding box
        xmin: The abscissa of the upper left point of the bounding box
        ymin: The ordinate of the lower right point of the bounding box
        ymin: The abscissa of the lower right point of the bounding box
        top: Number of the similar clothes, default to 5
        method: method to calculate distance, default to 'cos'
    Returns:
        list of filenames of the most similar cloths, like
        [
            'img/WOMEN/Blouses_Shirts/id_00000001/02_1_front.jpg',
            'img/WOMEN/Blouses_Shirts/id_00000001/02_2_side.jpg'
        ]
    """
    result = clf.predict(lambda: _input_fn(
        image_path, ymin, xmin, ymax, xmax))
    vector = list(result)[0]['logits']
    top = database.topN(
        vector, n=top, method=method)
    return filename[top]


def similar_cloth_v2(cropped, top=5, method='cos'):
    """
    Get clothes similar to the given one from the database

    The key difference of this 'v2' function compared to the the normal one is
    this function accept cropped image rather than image path and bounding box

    Args:
        cropped: 3-D Tensor of shape [height, width, channels], represents
            the cropped image of the input cloth
        top: Number of the similar clothes, default to 5
        method: method to calculate distance, default to 'cos'
    Returns:
        list of filenames of the most similar cloths, like
        [
            'img/WOMEN/Blouses_Shirts/id_00000001/02_1_front.jpg',
            'img/WOMEN/Blouses_Shirts/id_00000001/02_2_side.jpg'
        ]
    """
    result = clf.predict(lambda: _input_fn_v2(cropped))
    vector = list(result)[0]['logits']
    top = database.topN(vector, n=top, method=method)
    return filename[top]
