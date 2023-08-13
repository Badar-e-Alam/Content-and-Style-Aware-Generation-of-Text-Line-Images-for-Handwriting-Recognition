from cv2 import imread, resize, imshow, destroyAllWindows, waitKey,INTER_CUBIC
import numpy as np
import torch
import glob, os
import json
import xml.etree.cElementTree as ET
from load_data import IMG_HEIGHT, IMG_WIDTH
from parameters import device


class CustomImageDataset:
    def __init__(
        self,
        base_path="short_data/Words_data",
        img_dir=glob.glob("short_data/Words_images/*"),
    ):

        self.base_path = base_path
        self.img_dir = img_dir

    def Load_Image_Label(self, image_path):
        # Open the image file
        label = tuple()
        json_path = os.path.join(
            self.base_path, image_path.split("\\")[-1][:-4] + ".json"
        )
        with open(json_path, "r") as json_file:
            label = json.load(json_file)
        img = imread(image_path, 0)
        rate = float(IMG_HEIGHT) / img.shape[0]
        img = resize(
                img,
                (int(img.shape[1] * rate) + 1, IMG_HEIGHT),
                interpolation=INTER_CUBIC,
            )
        img = img / 255.0  # 0-255 -> 0-1
        img = 1.0 - img
        img_width = img.shape[-1]
        if img_width > IMG_WIDTH:
            outImg = img[:, :IMG_WIDTH]
            img_width = IMG_WIDTH
        else:
            outImg = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype="float32")
            outImg[:, :img_width] = img
        outImg = outImg.astype("float32")

        mean = 0.5
        std = 0.5
        outImgFinal = (outImg - mean) / std
        # img = 1- img/255
        # img_height, img_width = img.shape[0], img.shape[1]
        # n_repeats = int(np.ceil(IMG_WIDTH / img_width))
        # padded_image = np.concatenate([img] * n_repeats, axis=1)
        # padded_image = padded_image[:IMG_HEIGHT, :IMG_WIDTH]
        # resized_img = resize(padded_image, (IMG_WIDTH, IMG_HEIGHT))
        return (outImgFinal, label)
        # plt.imshow(img)
        # plt.show()

    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, idx):
        # import pdb;pdb.set_trace()
        Image, Labels = self.Load_Image_Label(self.img_dir[idx])
        return torch.tensor(Image, device=device).float(), Labels
        # return Image,Labels
