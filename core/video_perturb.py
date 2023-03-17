import numpy as np
import cv2
import torch
import skimage
import torchvision.datasets as datasets
import random
import math
import pdb
import submodules.pixmix.pixmix_utils as utils
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)
from torchvision import transforms

from PIL import Image
from io import BytesIO
from scipy.ndimage import zoom as scizoom

from wand.image import Image as WandImage
from wand.api import library as wandlibrary

import ctypes
# Tell Python about the C method
wandlibrary.MagickMotionBlurImage.argtypes = (ctypes.c_void_p,  # wand
                                              ctypes.c_double,  # radius
                                              ctypes.c_double,  # sigma
                                              ctypes.c_double)  # angle


try:
    import accimage
except ImportError:
    accimage = None


# Constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Change this to where your local Pix
MIXING_SET = '/home/c3-0/datasets/pixmix/fractals'


class PixMix(torch.nn.Module):
    def __init__(self, mixing_dir=MIXING_SET, severity=1, mixing_iter=4, all_ops=True):
        """
        This is used to extend PixMix from https://github.com/andyzoujm/pixmix to videos.
        Here we add a random image from the fractals data provided by PixMix to each frame in a clip.

        :param mixing_dir:
        :param preprocess:
        """
        super().__init__()
        self.mixing_dir = mixing_dir
        self.mixing = utils.mixings
        self.augment_severity = severity
        self.mixing_severity = severity
        self.all_ops = all_ops
        self.k = mixing_iter
        to_tensor = transforms.ToTensor()

        normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        self.preprocess = {'normalize': normalize,
                           'tensorize': to_tensor,
                           'chill_PIL': transforms.ToPILImage()}

        self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224)
            ])
        self.mixing_set = datasets.ImageFolder(
            MIXING_SET,
            transform=self.transform
        )

    def forward(self, x):
        rnd_idx = np.random.choice(len(self.mixing_set))
        mixing_pic, _ = self.mixing_set[rnd_idx]
        return self.pixmix(x, mixing_pic)

    def pixmix(self, orig, mixing_pic):
        mixings = self.mixing
        orig = self.transform(orig)
        tensorize, normalize = self.preprocess['tensorize'], self.preprocess['normalize']
        if np.random.random() < 0.5:
            mixed = tensorize(self.augment_input(orig))
        else:
            mixed = tensorize(orig)

        for _ in range(np.random.randint(self.k + 1)):

            if np.random.random() < 0.5:
                aug_image_copy = tensorize(self.augment_input(orig))
            else:
                aug_image_copy = tensorize(mixing_pic)

            mixed_op = np.random.choice(mixings)

            mixed = mixed_op(mixed, aug_image_copy, self.mixing_severity)
            mixed = torch.clip(mixed, 0, 1)
        mixed = normalize(mixed)
        return mixed

    def augment_input(self, image):
        aug_list = utils.augmentations_all if self.all_ops else utils.augmentations
        op = np.random.choice(aug_list)
        return op(image.copy(), self.augment_severity)


# Extend wand.image.Image class to include method signature
class MotionImage(WandImage):
    def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
        wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)


class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors.
    """

    def __init__(self):
        super().__init__()

    def forward(self, frames: torch.Tensor):
        alpha = 4
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list


class VideoPerturbation(torch.nn.Module):
    def __init__(self, perturb_type, sev=1, train=False, debug=False):
        super().__init__()
        mapping = {
            'motion_blur': self.motion_blur,
            'compression': self.jpeg_compression,
            'defocus_blur': self.defocus_blur,
            'gaussian_noise': self.gaussian_noise,
            'shot_noise': self.shot_noise,
            'impulse_noise': self.impulse_noise,
            'zoom_blur': self.zoom_blur,
            'rotate': self.var_rotate,
            'static_rotate': self.static_rotate,
            "speckle_noise": self.speckle_noise,
            "sampling": self.sampling,
            "reverse_sampling": self.reverse_sampling,
            'jumble': self.jumble,
            'box_jumble': self.box_jumble,
            'freeze': self.freeze,
            'translate': self.translate,

            # Mixed randomly with trainin severities randomly either 1,2,3 and testing either 4,5
            'temporal': self.mix_temporal,
            'spatial': self.mix_spatial,
            'mixed': self.mix,
            'pixmix': PixMix(severity=sev)
        }
        self.mapping = mapping

        self.compression_types = ['compression']
        self.temporal_types = ['jumble', 'box_jumble', 'freeze', 'sampling', 'reverse_sampling']
        self.spatio_types = ['speckle_noise', 'shot_noise', 'impulse_noise', 'gaussian_noise']
        self.camera_types = ['translate', 'rotate', 'static_rotate']

        self.debug = debug
        self.non_pil_transforms = ['jumble', 'box_jumble', 'freeze', 'sampling', 'translate', 'reverse_sampling']
        self.static_transforms = ['static_rotate']
        try:
            self.transform = mapping[perturb_type]
        except KeyError:
            print(f"Please pass a valid perturbation from {list(mapping.keys())}")

        self.perturb_type = perturb_type
        self.train = train
        if perturb_type == 'mixed' or perturb_type == 'spatial' or perturb_type == 'temporal':
            self.mixed = True
            if perturb_type == 'spatial':
                print("Running mixed spatial.")
                self.mixed_spatial = True
                self.mixed_temporal = False
            elif perturb_type == 'temporal':
                print("Running mixed temporal.")
                self.mixed_temporal = True
                self.mixed_spatial = False
            else:
                self.mixed_spatial = False
                self.mixed_temporal = False
        else:
            self.mixed = False
        self.sev = sev

    def mix_temporal(self):
        if self.train:
            types = ['jumble',  'freeze', 'sampling']
            severities = [1, 2, 3]
        else:
            types = ['jumble',  'freeze', 'sampling', 'box_jumble', 'reverse_sampling']
            severities = [4, 5]
        self.perturb_type = random.sample(types, k=1)[0]
        self.transform = self.mapping[self.perturb_type]
        self.sev = random.sample(severities, k=1)[0]

    def mix_spatial(self):
        if self.train:
            types = ['speckle_noise', 'gaussian_noise', 'rotate']
            severities = [1, 2, 3]
        else:
            types = ['shot_noise', 'static_rotate', 'translate',  'impulse_noise']
            severities = [4, 5]
        self.perturb_type = random.sample(types, k=1)[0]
        self.transform = self.mapping[self.perturb_type]
        self.sev = random.sample(severities, k=1)[0]

    def mix(self):
        types = random.sample([self.compression_types, self.temporal_types, self.spatio_types, self.camera_types], k=1)[0]
        self.perturb_type = random.sample(types, k=1)[0]
        self.transform = self.mapping[self.perturb_type]
        self.sev = random.sample([1, 2, 3], k=1)[0]
        # print(f"Perturbation {self.perturb_type} and {self.sev}")

    def forward(self, frames):
        """
        This function will return a tensor that has been perturbed based on severity in motion blur.

        :param torch.Tensor frames: Expected shape (C, T, H, W)
        :return: torch.Tensor (T, H, W, C) of shape ([34, 224, 224, 3])
        """
        if frames.shape[-1] == 3:
            frames = frames.permute(3, 0, 1, 2)

        # For frame in frames, apply transform
        if self.mixed == True:
            if self.mixed_spatial:
                self.mix_spatial()
            elif self.mixed_temporal:
                self.mix_temporal()
            else:
                self.mix()

        if self.perturb_type in self.non_pil_transforms:
            frames_list = self.transform(frames).permute(1, 2, 3, 0)
        elif self.perturb_type in self.static_transforms:
            frames_list = self.transform(frames)
        else:
            frames2 = frames.permute(1, 0, 2, 3)
            t1 = transforms.ToPILImage()
            v = []
            for f in range(frames2.size()[0]):
                # Convert to PIL image
                tmp = t1(frames2[f])

                # Run transform
                tmp = self.transform(tmp)
                # Transform to tensor
                tmp = np.array(tmp)
                tmp = torch.from_numpy(tmp)

                v.append(tmp)

            frames_list = torch.stack(v, 0)
            # frames_list = frames_list.permute(1, 0, 2, 3)
        return frames_list

    def translate(self, x):
        if self.debug:
            pdb.set_trace()
        size = x.shape[-1]
        if size == 288:
            size = 256
        else:
            size = 224

        _x = np.random.randint(0, 8 * self.sev)
        y = np.random.randint(0, 8 * self.sev)
        x = x[:, :, _x:_x + size, y:y + size]
        if x.shape[-1] != size or x.shape[-2] != size:
            # Not sure why this is happening but will pad or cut off other space.
            test = torch.zeros(x.shape[0], x.shape[1], size, size)
            test[:, :, :min(x.shape[-2], size), :min(x.shape[-1], size)] = x[:, :, :size, :size]
            return test
        return x

    def reverse_sampling(self, x):
        """
        Originally had interleave, removed
        :param x:
        :return:
        """
        if self.debug:
            pdb.set_trace()
        sampling_rate = [2, 4, 8, 16, 32][self.sev-1]
        x = x[:, ::sampling_rate]
        # x = torch.repeat_interleave(x, sampling_rate, dim=1)
        x = x.flip(dims=(1, ))
        return x

    def sampling(self, x):
        """
        Originally had interleave, removed
        :param x:
        :return:
        """
        if self.debug:
            pdb.set_trace()
        sampling_rate = [2, 4, 8, 16, 32][self.sev - 1]
        x = x[:, ::sampling_rate]
        # x = torch.repeat_interleave(x, sampling_rate, dim=1)
        return x

    def speckle_noise(self, x):
        if self.debug:
            pdb.set_trace()
        c = [.15, .2, 0.25, 0.3, 0.35][self.sev - 1]

        x = np.array(x) / 255.
        return Image.fromarray(np.uint8(np.clip(x + x * np.random.normal(size=x.shape, scale=c), 0, 1) * 255))

    def static_rotate(self, frames):
        if self.debug:
            pdb.set_trace()
        # deg = random.randint(-self.sev * 6 - 1, self.sev * 6 + 1)
        deg = [7.5, 15, 30, 45, 90, 180][self.sev - 1]
        frames2 = frames.permute(1, 0, 2, 3)
        t1 = transforms.ToPILImage()
        v = []
        for f in range(frames2.size()[0]):
            # Convert to PIL image and rotate
            tmp = t1(frames2[f]).rotate(deg)

            # Transform to tensor
            tmp = np.array(tmp)
            tmp = torch.from_numpy(tmp)
            v.append(tmp)
        frames_list = torch.stack(v, 0)
        return frames_list

    def box_jumble(self, x):
        """
        Divided segments are jumbled.
        In this case, since we have 4 clips of 32 frames.
        We have a total of 128 frames.

        Segments will range from length 36 to 4.
        :param x:
        :return:
        """
        if self.debug:
            pdb.set_trace()

        # Would equate to [4, 9, 16, 25, 36]
        seg = (self.sev+1)**2

        final = list()
        total = x.shape[1]
        for j in range(math.ceil(total / seg)):
            tmp = list(range(j * seg, min((j + 1) * seg, total)))
            final.append(tmp)
        random.shuffle(final)
        x = x[:, np.concatenate(final)]
        return x

    def jumble(self, x):
        """
        Randomly jumble frames for each clip.
        :param x:
        :return:
        """
        if self.debug:
            pdb.set_trace()
        # Results in [32, 16, 8, 4, 2]
        seg = 64 // (2 ** self.sev)
        total = x.shape[1]
        final = list()
        for j in range(math.ceil(total/seg)):
             tmp = list(range(j*seg, min((j+1)*seg, total)))
             random.shuffle(tmp)
             final += tmp
        x = x[:, final]  # C, T, H, W
        return x

    def freeze(self, x):
        """
        Randomly freezes a frame.
        :param x:
        :return:
        """
        if self.debug:
            pdb.set_trace()
        total = x.shape[1]
        k = int([.4*total, .2*total, .1*total, max(.05*total, 2), 1][self.sev - 1])

        final = list()
        indices = list(range(0, total))
        subselect = random.sample(indices, k=k)
        subselect.sort()

        for idx, frame_ind in enumerate(subselect):
            if idx + 1 < len(subselect):
                # Add as many until the next one
                b = (subselect[idx + 1]) - frame_ind
            else:
                # If at the end of the list, go until total
                b = total - frame_ind
            final.extend([x[:, frame_ind]] * b)

        final = torch.stack(final).permute(1, 0, 2, 3)
        return final

    def var_rotate(self, x):
        """
        Var rotate
        :param x:
        :return:
        """
        if self.debug:
            pdb.set_trace()
        deg = random.randint(-self.sev*6-1, self.sev*6+1)
        return x.rotate(deg)

    def zoom_blur(self, x):
        if self.debug:
            pdb.set_trace()
        c = [np.arange(1, 1.11, 0.01),
             np.arange(1, 1.16, 0.01),
             np.arange(1, 1.21, 0.02),
             np.arange(1, 1.26, 0.02),
             np.arange(1, 1.31, 0.03)][self.sev - 1]

        x = (np.array(x) / 255.0).astype(np.float32)
        out = np.zeros_like(x)
        for zoom_factor in c:
            out += self.clipped_zoom(x, zoom_factor)

        x = (x + out) / (len(c) + 1)
        return Image.fromarray(np.uint8(np.clip(x, 0, 1) * 255.0))

    @staticmethod
    def clipped_zoom(img, zoom_factor):
        h = img.shape[0]
        # ceil crop height(= crop width)
        ch = int(np.ceil(h / zoom_factor))

        top = (h - ch) // 2
        img = scizoom(img[top:top + ch, top:top + ch], (zoom_factor, zoom_factor, 1), order=1)
        # trim off any extra pixels
        trim_top = (img.shape[0] - h) // 2

        return img[trim_top:trim_top + h, trim_top:trim_top + h]

    def impulse_noise(self, x):
        if self.debug:
            pdb.set_trace()
        c = [.03, .06, .09, 0.17, 0.27][self.sev - 1]

        x = skimage.util.random_noise(np.array(x) / 255., mode='s&p', amount=c)
        return Image.fromarray(np.uint8(np.clip(x, 0, 1) * 255))

    def shot_noise(self, x):
        if self.debug:
            pdb.set_trace()
        c = [250, 100, 50, 30, 15][self.sev - 1]

        x = np.array(x) / 255.
        return Image.fromarray(np.uint8(np.clip(np.random.poisson(x * c) / c, 0, 1) * 255))

    def gaussian_noise(self, x):
        if self.debug:
            pdb.set_trace()
        c = [.08, .12, 0.18, 0.26, 0.38][self.sev - 1]

        x = np.array(x) / 255.
        return Image.fromarray(np.uint8(np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255))

    @staticmethod
    def disk(radius, alias_blur=0.1, dtype=np.float32):
        if radius <= 8:
            L = np.arange(-8, 8 + 1)
            ksize = (3, 3)
        else:
            L = np.arange(-radius, radius + 1)
            ksize = (5, 5)
        X, Y = np.meshgrid(L, L)
        aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
        aliased_disk /= np.sum(aliased_disk)

        # supersample disk to antialias
        return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)

    def defocus_blur(self, x):
        if self.debug:
            pdb.set_trace()
        c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][self.sev - 1]

        x = np.array(x) / 255.0
        kernel = self.disk(radius=c[0], alias_blur=c[1])
        # *255
        channels = []
        for d in range(3):
            channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
        channels = np.array(channels).transpose((1, 2, 0))  # 3x224x224 -> 224x224x3
        return Image.fromarray(np.uint8(np.clip(channels, 0, 1) * 255))

    def jpeg_compression(self, x):
        if self.debug:
            pdb.set_trace()
        c = [25, 18, 15, 10, 7][self.sev - 1]

        output = BytesIO()
        x.save(output, 'JPEG', quality=c)
        x = Image.open(output)

        return x

    def motion_blur(self, x):
        if self.debug:
            pdb.set_trace()
        c = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)][self.sev - 1]

        output = BytesIO()
        x.save(output, format='PNG')
        x = MotionImage(blob=output.getvalue())

        x.motion_blur(radius=c[0], sigma=c[1], angle=np.random.uniform(-45, 45))

        x = cv2.imdecode(np.fromstring(x.make_blob(), np.uint8),
                         cv2.IMREAD_UNCHANGED)

        if x.shape != (224, 224) and x.shape != (256, 256):
            return Image.fromarray(np.uint8(np.clip(x[..., [2, 1, 0]], 0, 255)))  # BGR to RGB
        else:  # greyscale to RGB
            return Image.fromarray(np.uint8(np.clip(np.array([x, x, x]).transpose((1, 2, 0)), 0, 255)))


