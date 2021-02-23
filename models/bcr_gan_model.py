from .pix2pix_model import Pix2PixModel
from . import networks
import torch
from .diff_augment import ImgAugTransform


class BCRGANModel(Pix2PixModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='wgangp')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--bcr_cofficient', type=float, default=10, help='weight for dicriminator bcr loss ')
        return parser

    def __init__(self, opt):
        Pix2PixModel.__init__(self, opt)
        self.aug = ImgAugTransform(self.opt)
        self.criterionMSE = torch.nn.MSELoss()
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake', 'D_cr_real', 'D_cr_fake']

    def backward_D(self):
        augmented_fake_B = self.aug(self.fake_B)
        fake_AB = torch.cat((self.real_A, self.fake_B),
                            1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        augmented_fake_AB = torch.cat((self.real_A, augmented_fake_B), 1)  # we augmented fake B
        fake_immediate_vec, pred_fake = self.netD(fake_AB.detach(), take_base_model=True)
        aug_fake_immediate_vec, augmented_fake = self.netD(augmented_fake_AB.detach(), take_base_model=True)
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        self.loss_D_cr_fake =self.criterionMSE(fake_immediate_vec, aug_fake_immediate_vec)

        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        augmented_real_B = self.aug(self.real_B)
        augmented_real_AB = torch.cat((self.real_A, augmented_real_B), 1)
        real_immediate_vec, pred_real = self.netD(real_AB, take_base_model=True)
        aug_real_immediate_vec, augmented_real = self.netD(augmented_real_AB, take_base_model=True)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        self.loss_D_cr_real = self.criterionMSE(aug_real_immediate_vec, real_immediate_vec)
        # combine loss and calculate gradients
        self.loss_D_bcr = self.loss_D_cr_fake + self.loss_D_cr_real
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5 + self.opt.bcr_cofficient * self.loss_D_bcr
        self.loss_D.backward()
