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
import random
from torch.utils import data

def creator(img_dir, label_dir, transform1=None,transform2=None):
    images=[f for f in listdir(img_dir) if isfile(join(img_dir, f))]
    random.shuffle(images)

    train_data = images[:2500]
    val_data=images[2500:]
    tr=XviewsDataset2(img_dir,label_dir,train_data,transform1)
    vl=XviewsDataset2(img_dir,label_dir,val_data,transform2)
    return tr,vl
    # labels=[f for f in listdir(label_dir) if isfile(join(label_dir, f))]


class XviewsDataset2(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, img_dir, label_dir,images, transform1=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.label_dir = label_dir
        self.img_dir = img_dir
        self.images=images
        self.labels=images
        self.transform1 = transform1

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
        # label_name=os.path.join(self.label_dir,
        #                         self.labels[idx])
        label_name=self.label_dir+'/'+self.labels[idx][:-4]+'_target.png'
        # print(img_name)
        # print(label_name)
        image1=Image.open(img_name)
        label=Image.open(label_name)
        if self.transform1:
            image1,label = self.transform1(image1,label)
        # landmarks = self.landmarks_frame.iloc[idx, 1:]
        # landmarks = np.array([landmarks])
        # landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image1': image1,'label':label}

        # if self.transform:
        #     sample = self.transform(sample)

        return image1,label


