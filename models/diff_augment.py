import torch
import torchvision


class ImgAugTransform:
    def __init__(self, opt):
        self.opt = opt
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomAffine(degrees=0,
                                                translate=(opt.translate_aug_strength, opt.translate_aug_strength),
                                                scale=(1.00, opt.zoom_out_aug_strength)
                                                ),
        ])

    def __call__(self, image):
        return self.transforms(image)
