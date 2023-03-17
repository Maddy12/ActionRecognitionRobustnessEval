from torch.utils.data import Dataset
import numpy as np
import os
import torch
import pandas as pd
import random
import pdb
from core.video_perturb import VideoPerturbation
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    UniformCropVideo
)
from torchvision.transforms.functional import InterpolationMode
from torchvision import transforms
import decord
import sys
decord.bridge.set_bridge("torch")

NUM_FRAMES = {
    'mvit': 16,
    'resnet50': 8,
    'r2d1': 8,
    'slowfast_r50': 32,
    'x3d':16,
    'i3d_r50': 8
}

INPUT_RES = {
    'mvit': 224,
    'resnet50': 224,
    'r2d1': 224,
    'slowfast_r50': 256,
    'x3d': 224,
    'i3d_r50': 224
}

class ActionRecognitionUniformFrames(Dataset):
    def __init__(self, config, split, few_shot_sample=None):
        super().__init__()

        # Video configurations
        self.video_root = config.root_dir
        self.num_frames = config.num_frames  # 4
        self.sample = NUM_FRAMES[config.model_type] # config.sample_type  # 'uniform'
        self.model_type = config.model_type
        self.fix_start = config.fix_start   #None
        if split == 'train':
            self.perturbation = config.train_perturbation
            self.severity = config.train_severity
        else:
            self.perturbation = config.test_perturbation
            self.severity = config.test_severity

        # input_res = int(config.input_res)
        # center_crop = int(config.input_res * 1.25)
        input_res = int(INPUT_RES[config.model_type])
        center_crop = int(input_res * 1.25)

        randcrop_scale = (0.5, 1.0)
        color_jitter = (0, 0, 0)
        norm_mean = (0.485, 0.456, 0.406)
        norm_std = (0.229, 0.224, 0.225)

        self.temporal_perturbations = ['box_jumble', 'freeze', 'jumble', 'reverse_sampling', 'sampling', 'temporal',
                                       'mixed']
        normalize = transforms.Normalize(mean=norm_mean, std=norm_std)
        # if self.model_type in MODEL_TRANSFORMS.keys():
        #     self.transform = MODEL_TRANSFORMS[self.model_type](self.perturbation, self.severity)

        if split == 'train' and self.perturbation is None:
            self.transform = transforms.Compose([
                SimpleTransform(),
                transforms.RandomResizedCrop(input_res, scale=randcrop_scale),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=color_jitter[0], saturation=color_jitter[1], hue=color_jitter[2]),
                normalize,
            ])

        elif self.perturbation == 'pixmix':
            self.transform = transforms.Compose([
                RandomPixMixPerturbation(self.perturbation, input_res, randcrop_scale, color_jitter, normalize,
                                         self.severity, p=.75, train=False)
            ])
        elif self.perturbation is not None:
            self.transform = transforms.Compose([
                RandomPerturbation(self.num_frames, self.perturbation, self.severity, p=.75, train=True),
                SimpleTransform(),
                transforms.RandomResizedCrop(input_res, scale=randcrop_scale),
                transforms.RandomHorizontalFlip(),
                normalize,
                ])

        else:
            if self.perturbation == 'temporal' or self.perturbation == 'spatial':
                print(f"Running either temporal or spatial mix.")
                self.transform = transforms.Compose([
                    RandomPerturbation(self.num_frames, self.perturbation, self.severity, p=.75, train=False),
                    SimpleTransform(),
                    transforms.Resize(center_crop),
                    transforms.CenterCrop(center_crop),
                    transforms.Resize(input_res),
                    normalize,
                ])
            elif self.perturbation == 'pixmix':
                self.transform = RandomPixMixPerturbation(self.perturbation, input_res, randcrop_scale, color_jitter, normalize,
                                         self.severity, p=.75, train=False)
            else:
                self.transform = transforms.Compose([
                    SimpleTransform(),
                    transforms.Resize(center_crop),
                    transforms.CenterCrop(center_crop),
                    transforms.Resize(input_res),
                    normalize,
                ])

        if self.model_type == 'slowfast_r50':
            self.multipath = PackPathway(4)

        self.split = split
        if split == 'train':
            meta_pth = config.train_path
            self.dataset = config.train_dataset
        elif split == 'val':
            meta_pth = config.test_path
            self.dataset = config.test_dataset
        else:
            meta_pth = config.test_path
            self.dataset = config.test_dataset

        if self.dataset == 'ucf101':
            self.extension = '.avi'
            # if split == 'train' or 'train' in meta_pth:
            try:
                df = pd.read_csv(meta_pth, header=None, names=['filename', 'label'], delimiter='\s')
                if few_shot_sample:
                    df = df.groupby('label').head(few_shot_sample)
            except Exception as e:
                print(e)
                pdb.set_trace()

            # else:
            #     df = pd.read_csv(meta_pth, header=None, names=['filename'])

        elif self.dataset == 'kinetics400':
            self.extension = '.mp4'
            self.dataset = 'kinetics'
            # Train, Val and Test have label,youtube_id,start,end,filename,split,is_cc/split
            df = pd.read_csv(meta_pth)
            self.df = df

        elif self.dataset == 'hmdb51':
            self.dataset = 'hmdb51'
            self.extension = '.avi'
            if 'shots' in meta_pth:
                df = pd.read_csv(meta_pth, header=None, names=['filename', 'label'])

            else:
                df = pd.read_csv(meta_pth).rename(columns={'filepath': 'filename'}) #, header=True, names=['filename', 'label'])
        elif self.dataset == 'ssv2':
            self.dataset = 'ssv2'
            self.extension = '.webm'
            try:
                df = pd.read_csv(meta_pth)
            except Exception as e:
                print(e)
                pdb.set_trace()
            if few_shot_sample:
                df = df.groupby('label').head(few_shot_sample)
            df['n_label'] = df['label']
            df['label'] = df['classname']
            self.df = df
        else:
            print("Unsupported dataset passed. Exiting...")
            sys.exit()

        if self.dataset == 'kinetics' or self.dataset == 'ssv2':
            self.classnames = np.array(df['label'].unique())
        else:
            self.classnames = np.array(list(set([row.split('/')[0] for row in df['filename'].values])))

        self.classnames.sort()
        self.labels = np.array(range(0, len(self.classnames)))
        self.label_mapping = {k: v for k, v in zip(self.classnames, self.labels)}
        print(f"Loading data from {meta_pth} for split {split} "
              f"for dataset {self.dataset} with {len(self.classnames)} classes.")

        # Generate data list of (filename, class name, class number)
        if self.dataset == 'ucf101':
            data = [(row, row.split('/')[0], np.where(self.classnames == row.split('/')[0])[0][0]) for row in
                    df['filename'].values]
        elif self.dataset == 'hmdb51':
            data = [(row, row.split('/')[0], np.where(self.classnames == row.split('/')[0])[0][0]) for row in
                    df['filename'].values]
        elif self.dataset == 'kinetics' or self.dataset == 'ssv2':
            data = [(row['filename'], row['label'], np.where(self.classnames == row['label'])[0][0]) for _, row in df.iterrows()]

        self.data = data

    def __len__(self):
        return len(self.data)

    def generate_few_shot(self):
        """
        Generate a few-shot dataset (typically for the training set).

        This function is useful when one wants to evaluate a model
        in a few-shot learning setting where each class only contains
        a small number of images.
        :return:
        """
        raise NotImplementedError

    def sample_frames(self, vlen):
        acc_samples = min(self.num_frames, vlen)
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if self.sample == 'rand':
            frame_idxs = [random.choice(range(x[0], x[1])) for x in ranges]
        elif self.fix_start is not None:
            frame_idxs = [x[0] + self.fix_start for x in ranges]
        elif self.sample == 'uniform':
            frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]
        else:
            raise NotImplementedError

        return frame_idxs

    def read_frames_decord(self, video_path):
        video_reader = decord.VideoReader(video_path, num_threads=1)
        vlen = len(video_reader)
        frame_idxs = self.sample_frames(vlen)

        # Need to read in full video then transform
        if self.perturbation is not None and self.perturbation in self.temporal_perturbations:
            full_frame_idxs = np.array(range(vlen))
            frames = video_reader.get_batch(list(full_frame_idxs))  # (T, H, W, C)
            frames = self.transform(frames)

            # Subsample from full video
            if frames.shape[0] != vlen:
                frame_idxs = self.sample_frames(frames.shape[0])
            frames = frames[frame_idxs]

        else:
            frames = video_reader.get_batch(frame_idxs)
            try:
                frames = self.transform(frames)
            except:
                pdb.set_trace()


        if frames.shape[0] != self.num_frames:
            T, C, H, W = frames.shape
            frames = torch.cat([frames, torch.zeros(self.num_frames - T, C, H, W)])

        if self.model_type == 'slowfast_r50':
            frames = self.multipath(frames.permute(1,0,2,3))
        return frames, frame_idxs

    def __getitem__(self, idx):
        filename, classname, label = self.data[idx]
        if isinstance(filename, int):
            filename = str(filename)
        if self.extension not in filename:
            filename = filename+self.extension

        video_pth = os.path.join(self.video_root, filename)
        frames, frame_idxs = self.read_frames_decord(video_pth)
        # return filename, classname, label, frames

        if isinstance(frames, torch.Tensor) and frames.shape[0] != 3:
            frames = frames.permute(1, 0, 2, 3)

        return frames, label


class SimpleTransform(object):
    def __call__(self, frames):
        frames = frames.float() / 255
        frames = frames.permute(0, 3, 1, 2)
        return frames


class RandomPerturbation(object):
    def __init__(self, num_frames, perturbation, severity, p=.5, train=False):
        temporal_perturbations = ['box_jumble', 'freeze', 'jumble', 'reverse_sampling', 'sampling', 'temporal',
                                  'mixed']
        if perturbation in temporal_perturbations:
            self.clip_video = False
        else:
            self.clip_video = UniformTemporalSubsample(num_frames)

        self.transform = VideoPerturbation(perturbation, severity, debug=False, train=train)
        self.p = p

    def __call__(self, frames):
        if torch.rand(1) < self.p:
            # If not temporal, will clip so we do not run over full video
            if self.clip_video:
                frames = self.clip_video(frames)
            frames = self.transform(frames)
        return frames


class RandomPixMixPerturbation(object):
    def __init__(self, perturbation, input_res, randcrop_scale, color_jitter, normalize, severity, p=.5, train=False):
        self.transform = VideoPerturbation(perturbation, severity, debug=False, train=train)
        self.p = p
        self.normal_transform = transforms.Compose([
                SimpleTransform(),
                transforms.RandomResizedCrop(input_res, scale=randcrop_scale),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=color_jitter[0], saturation=color_jitter[1], hue=color_jitter[2]),
                normalize,
            ])

    def __call__(self, frames):
        if torch.rand(1) < self.p:
            frames = self.transform(frames)
        else:
            frames = self.normal_transform(frames)
        return frames


class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors.
    This code is adapted from torchvisions.
    Check out more requirements: https://pytorchvideo.org/docs/tutorial_torchhub_inference
    """

    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // self.alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list
