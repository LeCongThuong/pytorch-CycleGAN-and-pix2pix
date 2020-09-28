from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from IPython.display import display
from functools import partial
from ipywidgets import interactive, widgets
import shutil


class LabelTool:
    def __init__(self, full_strokes_image_dir, depth_map_image_dir, train_unique_ids_file_path, only_style_test_dir):
        self.full_strokes_image_dir = Path(full_strokes_image_dir)
        self.depth_map_image_dir = Path(depth_map_image_dir)
        self.train_unique_ids_file_path = train_unique_ids_file_path
        self.only_style_test_dir = Path(only_style_test_dir)
        self.train_unique_id_list = load_file_to_list(train_unique_ids_file_path)
        self.full_stroke_image_name_list = [image_path.name for image_path in Path(self.full_strokes_image_dir).rglob("*")
                                            if str(image_path.name).split('_')[0] in self.train_unique_id_list]
        self._temp_image_id = 0
        self._is_training = False
        self.output = widgets.Output

    def create_interactive_board(self):
        next_button = widgets.Button(max=1000, description="Next")
        save_button = widgets.Button(max=1000, description="Save")
        self.image_id_widget = widgets.IntSlider(value=self._temp_image_id)
        next_button.on_click(self.click_next_button)
        save_button.on_click(self.click_save_button)
        output = widgets.interactive_output(self.make_plot_images, {'image_id': self.image_id_widget})
        display(next_button, save_button, self.image_id_widget, output)

    def click_next_button(self, change):
        self.image_id_widget.value += 1

    def click_save_button(self, change):
        full_strokes_path = self.full_strokes_image_dir / self.full_stroke_image_name_list[self._temp_image_id]
        depth_mapth_path = self.depth_map_image_dir / self.full_stroke_image_name_list[self._temp_image_id]

        full_strokes_name = self.full_stroke_image_name_list[self._temp_image_id].split(".")[0] + "_real_A.png"
        depth_mapth_name = self.full_stroke_image_name_list[self._temp_image_id].split(".")[0] + "_fake_B.png"
        shutil.copyfile(str(full_strokes_path), str(self.only_style_test_dir / full_strokes_name))
        shutil.copyfile(str(depth_mapth_path), str(self.only_style_test_dir / depth_mapth_name))
        self.image_id_widget.value += 1
        self._temp_image_id += 1

    def make_plot_images(self, image_id):
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 9))
        axes[0].axis('off')
        axes[1].axis('off')

        full_strokes_path = self.full_strokes_image_dir / self.full_stroke_image_name_list[image_id]
        depth_mapth_path = self.depth_map_image_dir / self.full_stroke_image_name_list[image_id]

        pil_full_strokes_image = Image.open(full_strokes_path)
        pil_depth_mapth_image = Image.open(depth_mapth_path)

        np_full_strokes_image = np.asarray(pil_full_strokes_image, dtype=np.float32) / 255.
        np_full_strokes_image = np_full_strokes_image[:, :, 0]
        axes[0].imshow(np_full_strokes_image, cmap='gray')

        np_depth_map_image = np.asarray(pil_depth_mapth_image, dtype=np.float32) / 255.
        np_depth_map_image = np_depth_map_image[:, :, 0]
        axes[1].imshow(np_depth_map_image, cmap='gray')


def load_file_to_list(file_path):
    return [line.rstrip('\n') for line in open(file_path)]


def dump_list_to_file(self, dump_list, file_path):
    with open(file_path, 'w') as f:
        for item in dump_list:
            f.write("%s\n" % item)

