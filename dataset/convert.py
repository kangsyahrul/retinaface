""" Convert Dataset to TF Records Format """

from absl import app, flags

import os


FLAGS = flags.FLAGS
flags.DEFINE_string('dataset_path', './dataset/wider_face', 'Path to dataset')


def main(_):
    # Check dataset folder
    dataset_path = FLAGS.dataset_path
    print(dataset_path)


if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass
