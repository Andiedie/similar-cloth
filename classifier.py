import tensorflow as tf
import os
import main
import preprocess_image as pi

tf.logging.set_verbosity(tf.logging.ERROR)

__dirname = os.path.dirname(__file__)
model_path = os.path.join(__dirname, './no-lmk-model')

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
session_config = tf.ConfigProto(gpu_options=gpu_options)
run_config = tf.estimator.RunConfig().replace(session_config=session_config)
classifier = tf.estimator.Estimator(
    model_fn=main.model_fn, model_dir=model_path, config=run_config,
    params={
        'resnet_size': 50,
        'data_format': None,
        'batch_size': 32,
        'multi_gpu': False,
        'version': 2,
    })


def input_fn(image_path, ymin, xmin, ymax, xmax):
    with open(image_path, 'rb') as f:
        image_buffer = f.read()
    image = pi.preprocess(image_buffer, False, False, {
        'ymin': ymin,
        'xmin': xmin,
        'ymax': ymax,
        'xmax': xmax
    }, None)
    dataset = tf.data.Dataset.from_tensor_slices([image]).batch(1)
    return dataset

def classify(image_path, ymin, xmin, ymax, xmax):
    result = classifier.predict(lambda: input_fn(
        image_path, ymin, xmin, ymax, xmax))
    print(list(result))

