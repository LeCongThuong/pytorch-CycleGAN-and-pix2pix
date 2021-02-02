from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import argparse
import os


def read_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--image_dir', type=str, required=True, help='image directory')
    parse.add_argument('--dest_dir', type=str, required=True, help='directory to save image')
    parse.add_argument('--kernel_size', type=int, default=3, help='size of element structuring kernel')
    parse.add_argument('--image_ext', type=str, default='.png', help='image extension')
    args = parse.parse_args()
    return args


def process_args(args):
    """
    Args:
        args:

    Returns:
    """
    if not os.path.exists(args.image_dir):
        raise Exception('Image directory do not exist')
    os.makedirs(args.dest_dir, exist_ok=True)
    return args


def preprocess_2d_data(image_path, kernel_size=3):
    """
    Binarize image using otsu method and then make better image using closing method (Morphological Transformations)
    Args:
        image_path: path to image
        kernel_size: kernel size for element structuring

    Returns:
        image: preprocessed binary image

    """

    cv_image = cv2.imread(image_path, 1)
    cv_image = cv_image[:, :, :1].copy()
    _, binary_image = cv2.threshold(cv_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    constrast_binary_image = 255 - binary_image
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closing = cv2.morphologyEx(constrast_binary_image, cv2.MORPH_CLOSE, kernel)
    good_binary_image = 255 - closing
    return good_binary_image


def get_2d_images(image_dir, img_extension='.png'):
    """
    Get all images from directory
    Args:
        image_dir: directory that contains images
        img_extension: image extension
    Returns:
        list of images
    """
    return glob.glob(image_dir + f'/*{img_extension}')


def save_image(binary_image, image_name, dest_dir):
    """
    Args:
        binary_image:
        image_name:
        dest_dir:

    Returns:

    """
    image_path = os.path.join(dest_dir, image_name)
    print(image_path)
    cv2.imwrite(image_path, binary_image)


if __name__ == '__main__':
    args = read_args()
    args = process_args(args)

    image_path_list = get_2d_images(args.image_dir)

    for image_path in image_path_list:
        binary_image = preprocess_2d_data(image_path)
        print(binary_image.shape)
        image_dir, image_name = os.path.split(image_path)
        save_image(binary_image, image_name, args.dest_dir)