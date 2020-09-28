import numpy as np
import matplotlib.pyplot as plt
import glob
from IPython.display import display
from functools import partial
from ipywidgets import interactive, widgets
import PIL
from PIL import Image
import shutil
from pathlib import Path
from _collections import defaultdict
import random


class LabelUtil:
    def __init__(self, image_list_file, annos_file_path, checkpoint_id_file_path, problem_file_path):
        self.images_list = create_file_list(image_list_file)
        self.annos_file_path = annos_file_path
        self.checkpoint_id_file_path = checkpoint_id_file_path
        self.problem_file_path = problem_file_path
        self._temp_image_id = 0
        self._temp_threshold = [0, 0, 0, 0, 0]  # [global_thres, tl_thres, tr_thres, bl_thes, br_thres]
        self._control_position = 0
        self.temp_state_image = None
        self.current_image_id = -1

    def create_iteractive_board(self, resume=False):
        if resume:
            checkpoint_image_id = self.load_checkpoint() + 1
            self.current_image_id = checkpoint_image_id - 1
        else:
            checkpoint_image_id = 0
            self.current_image_id = -1

        def on_change_image_id(v):
            self._temp_image_id = v['new']

        def on_change_threshold(v, position, is_global=False):
            if is_global:
                self._temp_threshold = [v['new']] * 5
            else:
                self._temp_threshold[position] = v['new']
            self._control_position = position

        image_id_widgets = widgets.IntText(value=checkpoint_image_id, description="Image ID")
        has_problem_widgets = widgets.Checkbox(value=False, description="Is Invalid")
        threshold_widgets_global = widgets.FloatSlider(value=0.3, step=0.01, max=1, min=0, readout_format='.3f',
                                                       description="Global")
        threshold_widgets_tl = widgets.FloatSlider(value=0.3, step=0.01, max=1, min=0, readout_format='.3f',
                                                   description="Top Left")
        threshold_widgets_tr = widgets.FloatSlider(value=0.3, step=0.01, max=1, min=0, readout_format='.3f',
                                                   description="Top Right")
        threshold_widgets_bl = widgets.FloatSlider(value=0.3, step=0.01, max=1, min=0, readout_format='.3f',
                                                   description="Bottom Left")
        threshold_widgets_br = widgets.FloatSlider(value=0.3, step=0.01, max=1, min=0, readout_format='.3f',
                                                   description="Bottom Left")
        h1 = widgets.HBox(children=[threshold_widgets_tl, threshold_widgets_tr, has_problem_widgets])
        h2 = widgets.HBox(children=[threshold_widgets_bl, threshold_widgets_br])
        h3 = widgets.HBox(children=[threshold_widgets_global, image_id_widgets])
        h4 = widgets.VBox(children=[h1, h2, h3])
        image_id_widgets.observe(on_change_image_id, names='value')
        threshold_widgets_global.observe(partial(on_change_threshold, position=0, is_global=True), names='value')
        threshold_widgets_tl.observe(partial(on_change_threshold, position=1), names='value')
        threshold_widgets_tr.observe(partial(on_change_threshold, position=2), names='value')
        threshold_widgets_bl.observe(partial(on_change_threshold, position=3), names='value')
        threshold_widgets_br.observe(partial(on_change_threshold, position=4), names='value')

        output = widgets.interactive_output(self.filter_with_many_threshold,
                                            {"image_id": image_id_widgets, "threshold_global": threshold_widgets_global,
                                             "threshold_tl": threshold_widgets_tl, "threshold_tr": threshold_widgets_tr,
                                             "threshold_bl": threshold_widgets_bl, "threshold_br": threshold_widgets_br}
                                            )

        display(widgets.VBox([h4, output]))
        self.save_checkpoint(image_id_widgets, has_problem_widgets)

    def load_checkpoint(self):
        image_id_list = [int(line.rstrip('\n')) for line in open(self.checkpoint_id_file_path)]
        return max(image_id_list)

    def filter_with_many_threshold(self, image_id, threshold_global, threshold_tl, threshold_tr, threshold_bl,
                                   threshold_br):

        image_path = self.images_list[image_id]
        pil_image = Image.open(image_path)
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 9))
        axes[0].axis('off')
        np_image = np.asarray(pil_image, dtype=np.float32) / 255.
        np_gray_image = np_image[:, :, 0]
        axes[0].imshow(np_image)
        if image_id != self.current_image_id:
            self.temp_state_image = np_image[:, :, 0]
            self._control_position = 0
            self.current_image_id = image_id
        axes[1].axis('off')

        h, w = self.temp_state_image.shape[0], self.temp_state_image.shape[1]
        if self._control_position == 0:
            self.temp_state_image = np.where(np_gray_image < threshold_global, 0, 1)
        elif self._control_position == 1:
            self.temp_state_image[0: int(h // 2), 0: int(w // 2)] = np.where(np_gray_image[0: int(h // 2), 0: int(w // 2)] < threshold_tl, 0, 1)
        elif self._control_position == 2:
            self.temp_state_image[0: int(h // 2), int(w // 2): w] = np.where(np_gray_image[0: int(h // 2), int(w // 2): w] < threshold_tr, 0, 1)
        elif self._control_position == 3:
            self.temp_state_image[int(h // 2): h, 0: int(w // 2)] = np.where(np_gray_image[int(h // 2): h, 0: int(w // 2)] < threshold_bl, 0, 1)
        else:
            self.temp_state_image[int(h // 2): h, int(w // 2): w] = np.where(np_gray_image[int(h // 2): h, int(w // 2): w] < threshold_br, 0, 1)
        axes[1].imshow(self.temp_state_image, cmap='gray')

    def _on_button_clicked(self, b, output_widget, image_id_widgets, has_problem_widgets):
        image_path = self.images_list[self._temp_image_id]
        with open(self.annos_file_path, 'a') as f:
            annos = image_path + ' ' + " ".join([str(element) for element in self._temp_threshold[1:]]) + '\n'
            f.write(annos)
        if has_problem_widgets.value:
            with open(self.problem_file_path, 'a') as f:
                f.write(image_path + '\n')
            has_problem_widgets.value = False
        with open(self.checkpoint_id_file_path, 'a') as f:
            f.write(str(self._temp_image_id) + '\n')
        image_id_widgets.value += 1

        with output_widget:
            print(annos)

    def save_checkpoint(self, image_id_widgets, has_problem_widgets):
        button = widgets.Button(description="Save")
        output_widget = widgets.Output()
        display(button, output_widget)
        button.on_click(partial(self._on_button_clicked, output_widget=output_widget,
                                image_id_widgets=image_id_widgets, has_problem_widgets=has_problem_widgets))


def generate_theshold_list(threshold_range_list):
    """
    generate list of threshold
    Args:
        threshold_range_list: list of ranges for choosing threshold
    Returns:
        list: list of threshold
    """
    threshold_list = []
    for threshold_range in threshold_range_list:
        threshold_list.append(round(np.random.uniform(threshold_range[0], threshold_range[1]), 2))
    return threshold_list


def generate_images(img_dir: str, saved_dir: str, threshold_range_list=[[0.37, 0.4], [0.4, 0.45], [0.45, 0.48], [0.48, 0.51], [0.51, 0.55], [0.55, 0.58]],
                    scaled_method=PIL.Image.LANCZOS, target_size=(256, 256)):
    """

    Args:
        img_dir: dir contains original images
        saved_dir: where to save processed images
        scaled_method: method to scale down images
        target_size: target size
        threshold_range_list: list of ranges for choosing threshold

    Returns:
        None

    """
    threshold_list = generate_theshold_list(threshold_range_list)
    image_dir_path = Path(img_dir)
    image_path_list = image_dir_path.rglob("*")
    for image_path in image_path_list:
        pil_image = Image.open(image_path)
        for threshold in threshold_list:
            processed_image = augment_image(pil_image, threshold=threshold, scaled_method=scaled_method,
                                            target_size=target_size)
            image_name = image_path.stem
            new_image_name = "_".join(image_name.split('_')[:2]) + f"_thres{threshold}.png"
            new_image_path = Path(saved_dir) / new_image_name
            processed_image.save(new_image_path)


def augment_image(img, threshold: int, scaled_method=PIL.Image.LANCZOS, target_size=(256, 256)):
    """

    Args:
        img: PIL image : depth map image
        threshold: pixel_value < threshold: pixel_value = 0, pixel_value = 1
        scaled_method: methods for scaling down an image
        target_size: size of resized image
    Returns:
        image :PIL image: a image contains original image and augmented image
    """
    resized_image = img.resize(target_size, resample=scaled_method)
    np_img = np.asarray(resized_image)
    np_processed_image = np.where(np_img > int(threshold * 255), 255, 0).astype(np.uint8)
    result_image = np.concatenate([np_processed_image, np_img], 1)
    pil_processed_image = Image.fromarray(result_image)
    return pil_processed_image


def list_all_files_and_save_list(image_dir, image_list_file):
    """

    Args:
        image_dir: image dir
        image_list_file: file that content all image path in image_dir

   Returns:
       None
    """
    file_list = glob.glob(image_dir + "/*")
    with open(image_list_file, 'w') as f:
        for item in file_list:
            f.write("%s\n" % item)


def list_image_id_in_dir(image_dir, saved_file_path):
    image_path_dir = Path(image_dir)
    image_id_list = [image_path.name.split("_")[0] for image_path in image_path_dir.rglob("*")]
    print(len(image_id_list))
    with open(saved_file_path, 'w') as f:
        for item in image_id_list:
            f.write("%s\n" % item)


def create_file_list(image_list_file):
    """

    Args:
        image_list_file: A file contains all image path

    Returns:
        image_path_list :list: contains image path
    """
    image_path_list = [line.rstrip('\n') for line in open(image_list_file)]
    return image_path_list


def create_2d_images(annos_file_path, problem_file_path, saved_dir="/"):
    annos_content = [line.rstrip('\n') for line in open(annos_file_path)]
    problem_file_list = [line.rstrip('\n') for line in open(problem_file_path)]
    image_path_list = []
    threshold_list = []
    for line in annos_content:
        image_path = line.split(' ')[0]
        threshold = line.split(' ')[1]
        if image_path not in problem_file_list:
            image_path_list.append(image_path)
            threshold_list.append(threshold)

    print("Num images: ", len(image_path_list))
    print("Num error images: ", len(problem_file_list))
    for i in range(len(image_path_list)):
        image_path = image_path_list[i]
        threshold = float(threshold_list[i])
        image_name = image_path.split('/')[-1]
        pil_img = Image.open(image_path)
        np_img = np.asarray(pil_img)
        np_processed_image = np.where(np_img > int(threshold * 255), 255, 0).astype(np.uint8)
        pil_processed_image = Image.fromarray(np_processed_image)
        pil_processed_image.save(saved_dir + '/' + image_name)
        
        
def create_valid_images_list(annos_file_path, problem_file_path):
    annos_content = [line.rstrip('\n') for line in open(annos_file_path)]
    problem_file_list = [line.rstrip('\n') for line in open(problem_file_path)]
    image_path_list = []
    for line in annos_content:
        image_path = line.split(' ')[0]
        if image_path not in problem_file_list:
            image_path_list.append(image_path)
    return image_path_list


def move_files(image_path_list, des_dir):
    for image_path in image_path_list:
        shutil.copy(image_path, des_dir)


def make_statistics_about_threshold(anno_file_path: str):
    """
    Make a plots about threshold
    Args:
        anno_file_path: file that contains image_path and threhold

    Returns:

    """
    threshold_bin = defaultdict(int)
    annos_content = [line.rstrip('\n') for line in open(anno_file_path)]
    for anno in annos_content:
        threshold = int(float(anno.split(" ")[-1]) * 100)
        threshold_bin[threshold] += 1
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.bar(list(threshold_bin.keys()), list(threshold_bin.values()))
    plt.show()


def list_statistic_unique_files(image_dir: str, unique_file_path: str):
    """
    list all unique characters and save to file
    Args:
        image_dir: contains 3 sub dirs: train, val, test
        unique_file_path: file contains list of unique file path

    Returns:

    """
    train_dir_path = Path(image_dir) / "depthmap_train"
    val_dir_path = Path(image_dir) / "depthmap_val"
    test_dir_path = Path(image_dir) / "depthmap_test"
    all_image_ids = []
    train_image_ids = [image.name.split('_')[0] for image in train_dir_path.rglob("*")]
    val_image_ids = [image.name.split('_')[0] for image in val_dir_path.rglob("*")]
    test_image_ids = [image.name.split('_')[0] for image in test_dir_path.rglob("*")]

    print(len(train_image_ids))
    print(len(val_image_ids))
    print(len(test_image_ids))

    all_image_ids.extend(train_image_ids)
    all_image_ids.extend(val_image_ids)
    all_image_ids.extend(test_image_ids)
    unique_image_ids = set(all_image_ids)
    print(len(all_image_ids))
    print(len(unique_image_ids))
    with open(unique_file_path, 'w') as f:
        for item in unique_image_ids:
            f.write("%s\n" % item)


def create_dataset(unique_id_list_path: str, image_dir: str, train_dir: str, test_dir: str, test_image_num=500):
    unique_id_list = [line.rstrip('\n') for line in open(unique_id_list_path)]
    Path(train_dir).mkdir(parents=True, exist_ok=True)
    Path(test_dir).mkdir(parents=True, exist_ok=True)
    random.shuffle(unique_id_list)
    test_image_id_list = unique_id_list[:test_image_num]
    train_image_id_list = unique_id_list[test_image_num:]

    train_dir_path = Path(image_dir) / "depthmap_train"
    val_dir_path = Path(image_dir) / "depthmap_val"
    test_dir_path = Path(image_dir) / "depthmap_test"
    all_image_path = []
    train_image_path = [str(image) for image in train_dir_path.rglob("*")]
    val_image_path = [str(image) for image in val_dir_path.rglob("*")]
    test_image_path = [str(image) for image in test_dir_path.rglob("*")]

    all_image_path.extend(train_image_path)
    all_image_path.extend(val_image_path)
    all_image_path.extend(test_image_path)
    #
    # for train_image_id in train_image_id_list:
    #     for image_path in all_image_path:
    #         if train_image_id in image_path:
    #             shutil.copy2(image_path, train_dir)
    #             break
    #
    # for test_image_id in test_image_id_list:
    #     for image_path in all_image_path:
    #         if test_image_id in image_path:
    #             shutil.copy2(image_path, test_dir)
    #             break
    return all_image_path


def create_unseen_dataset(image_dir, des_dir, src_dir):
    train_dir_path = Path(image_dir) / "train"
    val_dir_path = Path(image_dir) / "val"
    test_dir_path = Path(image_dir) / "test"
    all_image_path = []
    train_image_path = [str(image) for image in train_dir_path.rglob("*")]
    val_image_path = [str(image) for image in val_dir_path.rglob("*")]
    test_image_path = [str(image) for image in test_dir_path.rglob("*")]

    all_image_path.extend(train_image_path)
    all_image_path.extend(val_image_path)
    all_image_path.extend(test_image_path)
    print(len(all_image_path))

    des_image_name_list = ["_".join(image.name.split('_')[:-2]) + '.png' for image in Path(src_dir).rglob("*")]
    print(des_image_name_list)
    for image in all_image_path:
        if image.split("/")[-1] in des_image_name_list:
            shutil.copy2(image, des_dir)


if __name__ == '__main__':
    img_dir = "/home/love_you/mocban/dataset/raw/train"
    saved_dir = "/home/love_you/mocban/dataset/augmented/train"
    generate_images(img_dir, saved_dir, threshold_range_list = [[0.34, 0.39], [0.39, 0.45], [0.45, 0.48], [0.48, 0.51], [0.51, 0.55], [0.55, 0.58]],
                    scaled_method = PIL.Image.LANCZOS, target_size=(256, 256))
    # image_list_file = "/home/love_you/Documents/Study/deep_learning/pytorch-CycleGAN-and-pix2pix/train_list_file.txt"
    # annos_file_path = "/home/love_you/Documents/Study/deep_learning/pytorch-CycleGAN-and-pix2pix/train_annos.txt"
    # checkpoint_id_file_path = "/home/love_you/Documents/Study/deep_learning/pytorch-CycleGAN-and-pix2pix/train_checkpoint.txt"
    # labelUtil = LabelUtil(image_list_file, annos_file_path, checkpoint_id_file_path)
    #labelUtil.create_iteractive_board(resume=False)

#   image_dir = "/media/love_you/DOCUMENTS/Study/mocban_all/depthmap_train"
#   image_list_file = "/home/love_you/Documents/Study/deep_learning/pytorch-CycleGAN-and-pix2pix/train_list_file.txt"
#    list_all_files_and_save_list(image_dir, image_list_file)
#     annos_file_path = "/home/love_you/Documents/Study/deep_learning/pytorch-CycleGAN-and-pix2pix/val_annos.txt"
#     problem_file_path = "/home/love_you/Documents/Study/deep_learning/pytorch-CycleGAN-and-pix2pix/val_problem.txt"
#     des_dir = "/media/love_you/DOCUMENTS/Study/b_images"
#     # image_path_list = create_valid_images_list(annos_file_path, problem_file_path)
#     # move_files(image_path_list, des_dir)
#     saved_dir = "/media/love_you/DOCUMENTS/Study/a_images"
#     create_2d_images(annos_file_path, problem_file_path, saved_dir)

    


