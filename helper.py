from parameters import *


# def pad_str(label):
#     complete_str = list()
#     lbl = list()
#     for index in range(len(label)):
#         for line in range(len(label[index])):
#             repeat = text_max_len - len(label[index][line])
#             half_str=label[index][line]
#             full_str = half_str + " " * repeat
#             label[index][line].replace(half_str,full_str)
#     return label
#     #         complete_str.append(full_str)
#     #     lbl.append(tuple(complete_str))
#     # return lbl
def pad_str(data):
    # data:str [('hello',"what",),("on the road","where we are")]
    # data :- lenght of data is dependent on the number_example and the batch_size data[num_examples][batchsize]
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i] = tuple(s.ljust(text_max_len, " ") for s in data[i])
    return data


def encoding(label: list, encoder):
    lst = torch.zeros((batch_size, num_example))
    for row in range(len(label)):
        for col in range(len(label[row])):
            string = list()
            # for char in words:
            #     string.append(encoder[char])
            label[row] = tuple(
                torch.tensor([encoder[char] for char in label[row][col]])
            )
    return label


def decoding(label, decoder):
    # Label[example][batch_size]
    words = []
    for str1 in label:
        chars = []
        for word in str1:
            chars.append([decoder[char] for char in word])
        words.append(torch.tensor(chars))
    return words  # [examples][batch_size]
