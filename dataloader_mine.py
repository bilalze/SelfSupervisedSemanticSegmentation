from numpy.core.numeric import zeros_like
from torch.utils.data import Dataset
import torch
import os
# import io
from skimage import io
import numpy as np
from PIL import Image
from os import listdir
from os.path import isfile, join
from torchvision import transforms as T
import matplotlib.pyplot as plt
from torch.utils import data

def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
        color_jitter = T.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
        rnd_color_jitter = T.RandomApply([color_jitter], p=0.8)
        rnd_gray = T.RandomGrayscale(p=0.2)
        color_distort = T.Compose([
        rnd_color_jitter,
        rnd_gray])
        return color_distort
class XviewsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, img_dir, label_dir, transform1=None,transform2=None,transform3=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.label_dir = label_dir
        self.img_dir = img_dir
        self.images=[f for f in listdir(img_dir) if isfile(join(img_dir, f))]
        self.labels=[f for f in listdir(label_dir) if isfile(join(label_dir, f))]
        self.transform1 = transform1
        self.transform2 = transform2
        self.transform3 = transform3

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # print(self.img_dir)
        # print(self.label_dir)
        # print(self.images[idx])
        img_name = os.path.join(self.img_dir,
                                self.images[idx])
        label_name=os.path.join(self.label_dir,
                                self.labels[idx])
        # print(img_name)
        # print(label_name)
        image1=Image.open(img_name)
        image2=Image.open(img_name)
        label=Image.open(label_name)
        if self.transform1:
            image1 = self.transform1(image1)
        if self.transform2:
            image2=self.transform2(image2)
        if self.transform3:
            label=self.transform3(label)
        # landmarks = self.landmarks_frame.iloc[idx, 1:]
        # landmarks = np.array([landmarks])
        # landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image1': image1, 'image2': image2,'label':label}

        # if self.transform:
        #     sample = self.transform(sample)

        return sample

transform = T.Compose([
                T.Resize([513,513]),
                T.CenterCrop([513,513]),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
transform2=T.Compose([
        get_color_distortion(),
        transform
        
    ])

transform3=T.Compose([
        T.Resize([65,65],T.InterpolationMode.NEAREST),
        T.ToTensor()
    ])
