import os
import glob
import tqdm
from cv2 import imread, resize
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

import torch
from parameters import *
from torchvision import transforms as transforms


class CustomImageDataset:
    def __init__(
        self,
        annotations_file="line_data/labels",
        img_dir="line_data/images",
        transform=transforms.ToTensor(),
    ):
        self.label_dict = {}

        self.img_path = img_dir
        self.transform = transform
        self.num_example = num_example
        self.label_path = annotations_file
        self.label_dir = os.listdir(self.label_path)
        self.img_dir = glob.glob(self.img_path + "/*/*/*.png")
        del self.label_dir[1230]
        for file in tqdm.tqdm(self.label_dir):
            tree = ET.parse(source=os.path.join(self.label_path, file), parser=None)
            root = tree.getroot()
            for child in root.iter("line"):
                data = child.attrib
                self.label_dict[data["id"]] = data["text"]

    def image_lbl(self, index):
        img = imread(self.img_dir[index], 0)
        img = 255 - img
        img_height, img_width = img.shape[0], img.shape[1]
        n_repeats = int(np.ceil(IMAGE_WIDTH / img_width))
        padded_image = np.concatenate([img] * n_repeats, axis=1)
        padded_image = padded_image[:IMAGE_HEIGHT, :IMAGE_WIDTH]
        resized_img = resize(padded_image, (IMAGE_WIDTH, IMAGE_HEIGHT))
        label = self.label_dict[os.path.basename(self.img_dir[index])[:-4]]
        return resized_img, label
        """ print(padded_image.shape)
        plt.imshow(padded_image)
        plt.axis('off')
        plt.show()
        print()"""

    def __len__(self):
        return len(self.label_dict)

    def __getitem__(self, idx):
        Images = list()
        Labels = list()
        data_set = {}
        random_idxs = np.random.choice(
            len(self.img_dir), self.num_example, replace=True
        )
        for index in random_idxs:
            Img, lbl = self.image_lbl(index)
            Images.append(torch.tensor(Img, device=device).float())
            # Images.append(torch.nn.functional.pad(tensor, (0, self.IMAGE_WIDTH - tensor.size(2), 0, self.IMAGE_HEIGHT - tensor.size(1))))
            Labels.append(lbl)
        concate_image = torch.cat(Images, 1)
        """if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)"""
        return Images, Labels
