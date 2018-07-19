# :dress: Smilar Cloth
This is the junior practical training project of Sun Yat-sen University.

Given the clothes picture and bounding box, our task is to find similar clothes.

## Prerequisite
- [Tensorflow](https://www.tensorflow.org/) 1.8.0
- [Pillow](https://github.com/python-pillow/Pillow) 5.0.0
- [NumPy](http://www.numpy.org/) 1.14.2
- [SciPy](https://www.scipy.org/) 1.0.0

## Dataset
We use [In-shop Clothes Retrieval Benchmark](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html) as the dataset, which is one of the subsets of [DeepFashion](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html).

Download and unzip the dataset. Put them in the `data` directory and make the structure look like this:

```
data
├── Anno
│   ├── list_attr_cloth.txt
│   ├── list_attr_items.txt
│   ├── list_bbox_inshop.txt
│   ├── list_color_cloth.txt
│   ├── list_description_inshop.json
│   ├── list_item_inshop.txt
│   └── list_landmarks_inshop.txt
├── Eval
│   └── list_eval_partition.txt
├── img
│   ├── MEN
│   └── WOMEN
└── README.txt
```

## Prepare
Generate TFRecord files.
```
python tfrecord_creator.py
```

## Train
Train the model.
```
python main.py
```

Visualize the training.
```
tensorboard --logdir=./model
```

## Build Database
Build the similar cloth database.
```
python main.py --build_database=True
```

## Get Similar Cloth
```python
import classifier as clf

cloth_path = '1.jpg'
bbox_ymin = 40
bbox_xmin = 40
bbox_ymax = 190
bbox_xmax = 230

similar_cloths = clf.similar_cloth(
  cloth_path,
  bbox_ymin,
  bbox_xmin,
  bbox_ymax,
  bbox_xmax,
)
```

## Preview
![1530955068048](https://user-images.githubusercontent.com/21376471/42438113-b10962fc-8391-11e8-8a9c-c6de4978259e.png)

![1530955184146](https://user-images.githubusercontent.com/21376471/42438116-b166e4ae-8391-11e8-9568-e476926168c8.png)
