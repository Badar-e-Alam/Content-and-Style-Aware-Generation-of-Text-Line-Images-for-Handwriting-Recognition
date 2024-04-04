import glob, os, sys
import cv2
from PIL import UnidentifiedImageError
from PIL import Image
import xml.etree.cElementTree as ET
from tqdm import tqdm
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from parameters import *


class data_set:
    def __init__(self) -> None:
        self.img_dir = word_folder
        self.label_dir = json_dict

    def show_image(image):
        imgplot = plt.imshow(image)
        plt.show()

    def __len__(self):
        return len(self.img_dir)

    def read_json_files(self, label):
        # using to read json file 1 @ a time
        file = open(label)
        dbl = json.load(file)
        dbl = list(dbl.values())
        file.close()
        return dbl

    def data_loading(self):
        train_data = {}
        count = 0
        max_width = 192
        for image, label in tqdm(zip(self.img_dir[:10], self.label_dir[:10])):
            print("coming here")
            folder_data = list()
            image_list = glob.glob(os.path.join(image, "*.png"))
            print("img_list", image_list)
            data = self.read_json_files(label)
            import pdb

            pdb.set_trace()
            for index, s_image in enumerate(image_list):
                tmp_dict = dict()
                print("s_image,", s_image)
                img = Image.open(s_image)
                hight, width = img.size
                n_repeats = int(np.ceil(max_width / width))
                repeated_image = np.tile(np.array(img), (1, n_repeats, 1))
                concatenated_img = np.concatenate(repeated_image, axis=0)
                # Convert the numpy array back to an image
                # output_img = Image.fromarray(concatenated_img)
                # plt.imshow(concatenated_img)
                print(concatenated_img.shape)
                # print(f"{type(img)=}")
                # img=img.resize((h,79))

                tmp_dict["img"] = concatenated_img

                tmp_dict["label"] = data[index]
                folder_data.append(tmp_dict)

            train_data[str(count)] = folder_data
            count += 1
        return train_data


if __name__ == "__main__":
    s = data_set()
    data = s.data_loading()
    print(data)
