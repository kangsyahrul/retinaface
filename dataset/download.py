""" Download Wider Face dataset to this project """
from absl import app, flags

import os.path
import shutil
import glob
import gdown
import zipfile


DOWNLOAD_DIR = './download'

URL_ANNOTATION = 'https://drive.google.com/file/d/1P8s8PvvQUmO64vkOAdjf_qMJUHF-Y7ua/view?usp=sharing'
URL_DATASET_TRAIN = 'https://drive.google.com/file/d/1-4yWWQCwead_x5KAqyq3wStV1F16P43Z/view?usp=sharing'
URL_DATASET_VAL = 'https://drive.google.com/file/d/1-77BmZJ5aA3vOF0HyTreakLeo6ImD5_v/view?usp=sharing'


def extract_file(file_name, file_path):
    print(f"Extracting {file_name}... ")
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(DOWNLOAD_DIR)


def download_file(share_url, file_name, file_path):
    download_id = share_url.split("/")
    download_url = f'https://drive.google.com/uc?id={download_id[5]}'

    print(f"Downloading {file_name}... ")
    gdown.download(download_url, file_path, quiet=False)


def download_annotation(share_url, file_name):
    # Check extracted file is exists
    label_path = glob.glob("./dataset/wider_face/*/label.txt")
    if len(label_path) == 3:
        print(f"Annotation file has been downloaded and extracted to {label_path}")
        return

    file_path = os.path.join(DOWNLOAD_DIR, file_name)
    annotation_dir = os.path.join(DOWNLOAD_DIR, file_name.split(".")[0])

    # Download file if not exists
    if not os.path.exists(file_path):
        download_file(share_url, file_name, file_path)

    # Extract file
    if not os.path.exists(annotation_dir):
        extract_file(file_name, file_path)

    # Move files
    for extracted_path in glob.glob("./download/annotation/*/label.txt"):
        destination_path = extracted_path.replace("download/annotation", "dataset/wider_face")
        destination_dir = "/".join(destination_path.split("/")[:-1])
        os.mkdir(destination_dir)
        shutil.move(extracted_path, destination_path)


def main(_):
    # Check extracted file if exists


    # Crate download folder if not exists
    if not os.path.exists("./download"):
        # Create download folder
        os.mkdir("./download")

    # Download file if not exits
    download_annotation(URL_ANNOTATION, 'annotation.zip')


if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass
