import torch

scale_factor = 1
number_feature = 2000
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"
batch_size=2
IMG_HEIGHT = 64 #64
IMG_WIDTH = 200 #216
MAX_CHARS = 25+2
NUM_CHANNEL = 1
EXTRA_CHANNEL = NUM_CHANNEL+1
NUM_WRITERS = 500 # iam
NORMAL = True
num_heads=2
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


encoder = {data: i for i, data in enumerate(vocab)}
decoder = {i: data for i, data in enumerate(vocab)}
