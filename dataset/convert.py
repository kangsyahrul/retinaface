""" Convert Dataset to TF Records Format """

from absl import app, flags, logging

import os
import math
import tqdm
import random
import numpy as np
import tensorflow as tf


FLAGS = flags.FLAGS
flags.DEFINE_string('dataset_path', './dataset/wider_face', 'Path to dataset')


def read_annotation(label_path, images_dir):
    images_path = []
    annotations = []

    f = open(label_path, 'r')
    image_path = None
    labels = []
    for line in f.readlines():
        line = line.rstrip()

        if line.startswith("#"):
            # Add to list
            if image_path is not None:
                images_path.append(os.path.join(images_dir, image_path))
                annotations.append(labels)

            # Image path
            image_path = line.split(" ")[1]

        else:
            line = line.split(' ')
            label = [float(x) for x in line]
            labels.append(label)

    images_path.append(os.path.join(images_dir, image_path))
    annotations.append(labels)

    return images_path, annotations


def get_labels(annotations):
    if len(annotations) == 0:
        return np.zeros((0, 15))

    labels = []
    for idx, annotation in enumerate(annotations):
        if len(annotation) == 4:
            annotation.extend([
                -1, -1, -1,
                -1, -1, -1,
                -1, -1, -1,
                -1, -1, -1,
                -1, -1, -1,
                -1
            ])

        label = [0] * 15

        # bbox
        label[ 0] = annotation[ 0]  # x1
        label[ 1] = annotation[ 1]  # y1
        label[ 2] = annotation[ 0] + annotation[2]  # x2
        label[ 3] = annotation[ 1] + annotation[3]  # y2

        # landmarks
        label[ 4] = annotation[ 4]  # l0_x
        label[ 5] = annotation[ 5]  # l0_y
        label[ 6] = annotation[ 7]  # l1_x
        label[ 7] = annotation[ 8]  # l1_y
        label[ 8] = annotation[10]  # l2_x
        label[ 9] = annotation[11]  # l2_y
        label[10] = annotation[13]  # l3_x
        label[11] = annotation[14]  # l3_y
        label[12] = annotation[16]  # l4_x
        label[13] = annotation[17]  # l4_y

        if label[4] < 0:
            label[14] = -1   # w/o landmark
        else:
            label[14] = 1    # w landmark

        labels.append(label)
    return np.array(labels)


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def make_example(img_name, img_path, labels):
    # Create a dictionary with features that may be relevant.
    feature = {
        # Image
        'image/img_name': _bytes_feature([img_name]),
        'image/encoded': _bytes_feature([open(img_path, 'rb').read()]),

        # Bounding boxes
        'image/object/bbox/xmin': _float_feature(labels[:, 0]),
        'image/object/bbox/ymin': _float_feature(labels[:, 1]),
        'image/object/bbox/xmax': _float_feature(labels[:, 2]),
        'image/object/bbox/ymax': _float_feature(labels[:, 3]),

        # Landmarks
        'image/object/landmark0/x': _float_feature(labels[:, 4]),
        'image/object/landmark0/y': _float_feature(labels[:, 5]),
        'image/object/landmark1/x': _float_feature(labels[:, 6]),
        'image/object/landmark1/y': _float_feature(labels[:, 7]),
        'image/object/landmark2/x': _float_feature(labels[:, 8]),
        'image/object/landmark2/y': _float_feature(labels[:, 9]),
        'image/object/landmark3/x': _float_feature(labels[:, 10]),
        'image/object/landmark3/y': _float_feature(labels[:, 11]),
        'image/object/landmark4/x': _float_feature(labels[:, 12]),
        'image/object/landmark4/y': _float_feature(labels[:, 13]),

        # Landmarks Validation
        'image/object/landmark/valid': _float_feature(labels[:, 14]),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def main(_):
    # Check dataset folder
    dataset_path = FLAGS.dataset_path
    print(dataset_path)

    # Check if dataset and annotation is already there
    for cat in ["train", "val"]:
        label_path = os.path.join(dataset_path, f"{cat}/label.txt")
        images_dir = os.path.join(dataset_path, f"{cat}/images")

        if not (os.path.exists(label_path) and os.path.exists(images_dir)):
            raise Exception(f"Annotation or dataset not found in {os.path.join(dataset_path, cat)}")

    # Read label.txt
    for cat in ["train", "val"]:
        label_path = os.path.join(dataset_path, f"{cat}/label.txt")
        images_dir = os.path.join(dataset_path, f"{cat}/images")
        record_dir = os.path.join(dataset_path, f"{cat}/records")

        # Check if records is exists
        if os.path.exists(record_dir):
            logging.info(f'{record_dir} already exists. Exit...')
            continue
        else:
            os.mkdir(record_dir)

        logging.info(f'Reading {cat} dataset...')
        images_path, annotations = read_annotation(label_path, images_dir)

        print(f"Images Sample: {images_path[0]}")
        print(f"Images Total: {len(images_path)}")

        print(f"Annotations Sample: {annotations[0][0]}")
        print(f"Annotations Total: {len(annotations)}")

        # Shuffle
        samples = list(zip(images_path, annotations))
        random.shuffle(samples)

        # Write the record with sharded
        start = 0
        end = 0
        n_images_shard = 2000
        n_shards = math.ceil(len(samples) / n_images_shard)

        logging.info(f'Start creating TF Records for: {cat}')
        logging.info(f'Images Nums: {len(samples)}')
        logging.info(f'Images Nums per Shard: {n_images_shard}')
        logging.info(f'Shard Nums: {n_shards}')

        # tqdm is an amazing package that if you don't know yet you must check it
        for shard in tqdm.tqdm(range(n_shards)):
            # The original tfrecords_path is "{}_{}_{}.records" so the first parameter is the name of the dataset,
            # the second is "train" or "val" or "test" and the last one the pattern.
            records_shard_path = '{}_{}.tfrecords'.format(cat, '%.5d-of-%.5d' % (shard, n_shards - 1))
            records_shard_path = os.path.join(record_dir, records_shard_path)

            end = start + n_images_shard if len(samples) > (start + n_images_shard) else -1
            datas = samples[start: end]
            start = end

            with tf.io.TFRecordWriter(records_shard_path) as writer:
                for img_path, annotations in tqdm.tqdm(datas, leave=False):
                    img_name = os.path.basename(img_path).replace('.jpg', '')
                    labels = get_labels(annotations)
                    example = make_example(
                        img_name=str.encode(img_name),
                        img_path=str.encode(img_path),
                        labels=labels,
                    )

                    writer.write(example.SerializeToString())


if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass
