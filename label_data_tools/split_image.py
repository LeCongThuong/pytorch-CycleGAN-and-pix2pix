from PIL import Image
import glob
import os


def split_image(img_path):
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    width_2 = int(width / 2)
    # img_2d = img.crop((0, 0, width_2, height))
    depthmap = img.crop((width_2, 0, width, height))
    return depthmap


def main():
    phases = ['train', 'val', 'test']
    correspond_phases = ['depthmap_train', 'depthmap_val', 'depthmap_test']
    img_src_dir = '/media/love_you/DOCUMENTS/Study/mocban_all'
    for index, phase in enumerate(phases):
        saved_dir = img_src_dir + '/' + correspond_phases[index]
        phase_dir = img_src_dir + '/' + phase
        img_paths = glob.glob(phase_dir + '/*')
        i = 0
        for img_path in img_paths:
            print(f"Phase : {phase}, {i}", end='')
            img_name = img_path.split('/')[-1]
            depthmap = split_image(img_path)
            saved_img_path = saved_dir + '/' + img_name
            depthmap.save(saved_img_path)
            i = i + 1


if __name__ == '__main__':
    main()