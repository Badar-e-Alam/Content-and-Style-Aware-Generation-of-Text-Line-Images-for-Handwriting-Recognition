import os, glob
import torch

scale_factor = 2**1
IMAGE_HEIGHT = 15  # 342
IMAGE_WIDTH = 150  # 2270
# device = torch.device("cuda" if torch.cu+2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"
batch_size = 2
num_example = 2
embedding_size = 64
Max_str = 81
text_max_len = Max_str + 4
Data_pth = "files\IAM-32.pickle"
base_folder = os.path.join("words_data/*/", "*")
word_folder = glob.glob(base_folder)
json_dict = glob.glob("..json/*")
resolution = 16
vocab = {
    " ",
    "!",
    '"',
    "#",
    "&",
    "'",
    "(",
    ")",
    "*",
    "+",
    ",",
    "-",
    ".",
    "/",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    ":",
    ";",
    "?",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
}

cfg = {
    "E": [
        64,
        64,
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
    ],
}
encoder = {data: i for i, data in enumerate(vocab)}
decoder = {i: data for i, data in enumerate(vocab)}
"""
encoder= {"A":0,"B":1}
decoder={"0":A,"1":B}
"""
tokens = {"GO_TOKEN": 0, "END_TOKEN": 1, "PAD_TOKEN": 2}
NUM_WRITERS = 500
