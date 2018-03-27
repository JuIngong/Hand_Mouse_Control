import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tensorflow as tf

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import scipy.io
from utils import dataset_util

import io
import hashlib

LABEL_MAP_PATH = 'hand_label_map.pbtxt'
label_map_dict = label_map_util.get_label_map_dict(LABEL_MAP_PATH)



def coords_from_mat(mat_filepath):
    mat = scipy.io.loadmat(mat_filepath)
    coords = []
    i = 0
    for e in mat['boxes'][0]:
        coords.append(list())
        c = 0
        for d in e[0][0]:
            if c > 3:
                break
            coords[i].append((d[0][0], d[0][1]))
            c += 1
        i += 1
    return coords

def create_tf_example(name, img_dir, ann_dir):

    IMG_FILENAME = '%s.jpg' % name
    ANN_FILENAME = '%s.mat' % name
    IMG_FULL_PATH = os.path.join(img_dir, IMG_FILENAME)
    ANN_FULL_PATH = os.path.join(ann_dir, ANN_FILENAME)

    with tf.gfile.GFile(IMG_FULL_PATH, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()

    label = 'hand'
    width, height = image.size

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    difficult_obj = []

    coords = coords_from_mat(ANN_FULL_PATH)

    for coord in coords:

        x_max, x_min, y_max, y_min = 0, float('inf'), 0, float('inf')
        for y,x in coord:
            x_max, x_min = max(x, x_max), min(x, x_min)
            y_max, y_min = max(y, y_max), min(y, y_min)

        xmin.append(max(float(x_min) / width, 0.0))
        ymin.append(max(float(y_min) / height, 0.0))
        xmax.append(min(float(x_max) / width, 1.0))
        ymax.append(min(float(y_max) / height, 1.0))
        classes_text.append(label.encode('utf8'))
        classes.append(label_map_dict[label])
        truncated.append(0)
        poses.append('Frontal'.encode('utf8'))
        difficult_obj.append(0)

    return tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(
              IMG_FILENAME.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(
              IMG_FILENAME.encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
        'image/object/truncated': dataset_util.int64_list_feature(truncated),
        'image/object/view': dataset_util.bytes_list_feature(poses),
      }))


def create_tf_record(img_dir, ann_dir, output_filename):
    writer = tf.python_io.TFRecordWriter(output_filename)
    for f in os.listdir(img_dir):
        if '.jpg' in f:
            img_name = f.split('.')[0]
            tf_example = create_tf_example(img_name, img_dir, ann_dir)
            writer.write(tf_example.SerializeToString())
    writer.close()
    print('%s written.' % output_filename)


if __name__ == '__main__':
    create_tf_record('./test_dataset/test_data/images', './test_dataset/test_data/annotations', 'hands_test.record')
    create_tf_record('./training_dataset/training_data/images', './training_dataset/training_data/annotations', 'hands_train.record')
    create_tf_record('./validation_dataset/validation_data/images', './validation_dataset/validation_data/annotations', 'hands_val.record')
