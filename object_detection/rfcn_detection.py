# coding: utf-8
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

global rfcn_model_instance

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

if tf.__version__ < "1.4.0":
    raise ImportError("Please upgrade your tensorflow installation to v1.4.* or later!")


from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util


MODEL_NAME = "object_detection/rfcn_resnet101_consumer_350000"
PATH_TO_CKPT = MODEL_NAME + "/frozen_inference_graph.pb"
PATH_TO_LABELS = os.path.join("object_detection/data", "clothes_label_map.pbtxt")
NUM_CLASSES = 3
IMAGE_SIZE = (12, 8)

class RFCNDection(object):
    def __init__(self):
        self.detection_graph = self.get_detection_graph()
        self.category_index = self.get_category_index()
        with self.detection_graph.as_default():
            self.sess = tf.Session()

    def get_detection_graph(self):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, "rb") as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name="")
        
        return detection_graph
    
    def get_category_index(self):
        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)

        return category_index
    
    def load_image_into_numpy_array(self, image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
                (im_height, im_width, 3)).astype(np.uint8)
    
    def run_inference_for_single_image(self, image):
        with self.detection_graph.as_default():
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                    "num_detections", "detection_boxes", "detection_scores",
                    "detection_classes", "detection_masks"
            ]:
                tensor_name = key + ":0"
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                            tensor_name)
            if "detection_masks" in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict["detection_boxes"], [0])
                detection_masks = tf.squeeze(tensor_dict["detection_masks"], [0])

                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict["num_detections"][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                        detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                        tf.greater(detection_masks_reframed, 0.8), tf.uint8)
                
                # Follow the convention by adding back the batch dimension
                tensor_dict["detection_masks"] = tf.expand_dims(
                        detection_masks_reframed, 0)
            
            image_tensor = tf.get_default_graph().get_tensor_by_name("image_tensor:0")

            # Run inference
            output_dict = self.sess.run(tensor_dict, \
                    feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict["num_detections"] = int(output_dict["num_detections"][0])
            output_dict["detection_classes"] = output_dict["detection_classes"][0].astype(np.uint8)
            output_dict["detection_boxes"] = output_dict["detection_boxes"][0]
            output_dict["detection_scores"] = output_dict["detection_scores"][0]
            
            if "detection_masks" in output_dict:
                output_dict["detection_masks"] = output_dict["detection_masks"][0]
        return output_dict
    
    # API
    def detection_image(self, image_path):
        image = Image.open(image_path)
        image_np = self.load_image_into_numpy_array(image)
        image_np_expanded = np.expand_dims(image_np, axis=0)
        output_dict = self.run_inference_for_single_image(image_np)

        image_np, bboxs = vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                output_dict["detection_boxes"],
                output_dict["detection_classes"],
                output_dict["detection_scores"],
                self.category_index,
                instance_masks = output_dict.get("detection_masks"),
                use_normalized_coordinates=True,
                line_thickness=1)

        img = Image.fromarray(image_np).convert("RGB")

        #dirname, filename = os.path.split(image_path)
        #output_path = os.path.join(os.path.dirname(__file__), "output_img_real")
        #image_path = os.path.join(output_path, filename)
        #img.save(image_path)

        return bboxs
    
    def __del__(self):
        if self.sess:        
            self.sess.close()


# 这里生成模型实例供 server 导入并调用
print("生成 RFCN Model 实例.................")
rfcn_model_instance = RFCNDection()
print("RFCN Model 实例生成完成...............")
