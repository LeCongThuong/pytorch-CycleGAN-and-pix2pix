from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import glob
from collections import defaultdict, Counter
import shutil
import json
import os


def percent_character_area_over_background(image_path: str) -> int:
    """

    Args:
        image_path:  path to images

    Returns:
        percent area of character over background: percent_ch_over_bg
    """
    pil_img = Image.open(image_path)
    np_img = np.asarray(pil_img)
    img_height, img_width = np_img.shape[0], np_img.shape[1]
    num_ch_pixel = np.count_nonzero(np_img[:, :int(img_width / 2), 0])
    num_pixels = img_height * img_width / 2
    return 100 - int(num_ch_pixel / num_pixels * 100)


def make_statistics_coverage(img_dir: str, threshold_lower: int, threshold_higher: int):
    image_path_list = glob.glob(img_dir + '/*')
    stat_dict = defaultdict(int)
    count = 0

    small_area_image_paths_list = []
    large_area_image_paths_list = []

    for image_path in image_path_list:
        percent = percent_character_area_over_background(image_path)
        if percent <= threshold_lower:
            small_area_image_paths_list.append(image_path)
        if percent >= threshold_higher:
            large_area_image_paths_list.append(image_path)

        stat_dict[percent] += 1
        count += 1
        print(f'{count}', end='\r')
    return stat_dict, small_area_image_paths_list, large_area_image_paths_list


def save_and_make_plot(stat_dict, small_area_image_paths_list, large_area_image_paths_list):
    freq_dict = dict(stat_dict)
    with open('/pytorch-CycleGAN-and-pix2pix/label_data_tools/statistics_about_invalid_images/result.json', 'w') as fp:
        json.dump(freq_dict, fp)

    with open('/pytorch-CycleGAN-and-pix2pix/label_data_tools/statistics_about_invalid_images/small_area_image_paths_file.txt', 'w') as f:
        for item in small_area_image_paths_list:
            f.write("%s\n" % item)

    with open('/pytorch-CycleGAN-and-pix2pix/label_data_tools/statistics_about_invalid_images/large_area_image_paths_file.txt', 'w') as f:
        for item in large_area_image_paths_list:
            f.write("%s\n" % item)

    fig, ax = plt.subplots(figsize=(12, 9))

    ax.bar(list(stat_dict.keys()), list(stat_dict.values()))
    plt.savefig('/home/love_you/Documents/Study/deep_learning/mocban/pytorch-CycleGAN-and-pix2pix/label_data_tools/image.jpeg')
    plt.show()


def move_invalid_files(file_list_path, dest_dir):
    os.mkdir(dest_dir)
    with open(file_list_path, 'r') as f:
        lines = f.read().splitlines()
    for line in lines:
        shutil.move(line, dest_dir)


def make_statistics_num_images_per_id(img_dir):
    stat_default_dict = defaultdict(int)
    image_path_list = glob.glob(img_dir + '/*')
    for image_path in image_path_list:
        img_id = image_path.split(os.path.sep)[-1].split('_')[0]
        stat_default_dict[img_id] += 1
    stat_dict = dict(stat_default_dict)
    image_id_has_one_image = []
    for image_id in stat_dict:
        if stat_dict[image_id] == 1:
            image_id_has_one_image.append(image_id)
    with open('/pytorch-CycleGAN-and-pix2pix/label_data_tools/statistics_about_invalid_images/image_id_has_one_image_file.txt', 'w') as f:
        for item in image_id_has_one_image:
            f.write("%s\n" % item)

    return stat_default_dict


if __name__ == '__main__':
    image_dir = "/media/love_you/DOCUMENTS/Study/mocban_all/augment_train/"
    # stat_dict, small_area_image_paths_list, large_area_image_paths_list = make_statistics_coverage(image_dir, 3, 70)
    # save_and_make_plot(stat_dict, small_area_image_paths_list, large_area_image_paths_list)
    # move_invalid_files('/home/love_you/Documents/Study/deep_learning/mocban/pytorch-CycleGAN-and-pix2pix/label_data_tools/small_area_image_paths_file.txt', "/media/love_you/DOCUMENTS/Study/mocban_all/invalid_augment_train/low_area")
    make_statistics_num_images_per_id(image_dir)

