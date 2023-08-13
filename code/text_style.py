# import torch
# from torch import nn
# from parameters import *
# from load_data import IMG_HEIGHT,IMG_WIDTH,NUM_CHANNEL

# class TextEncoder_FC(nn.Module):
#     def __init__(self) -> None:
#         super(TextEncoder_FC, self).__init__()
#         """
#          self.embed = Apply the embedding layer on the text tensor(2,85) -> (batch_size,max_text_len) -> out= (batch_size,max_len,embedding_size)
#          xx = (batch_size, max_len_embedding_size)
#          xxx = reshape the embedding output  from (batch_size,max_len_text,embedding_size) -> (batch_size,max_len*embedding_size)
#          out = Contained the output of the text style_network out_dim -> (batch_size,4096)

#          xx_new =  apply the Linear layer on the embedding output

#         """
#         self.embed = nn.Embedding(len(vocab), embedding_size)  # 81,64
#         self.fc = nn.Sequential(
#             nn.Flatten(),  # flatten the input tensor to a 1D tensor
#             nn.Linear(text_max_len * embedding_size*NUM_CHANNEL, 1024),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(inplace=False),
#             nn.Linear(1024, 2048),
#             nn.BatchNorm1d(2048),
#             nn.ReLU(inplace=False),
#             nn.Linear(2048, number_feature),
#         )
#         self.linear = nn.Linear(
#             embedding_size * text_max_len*NUM_CHANNEL, embedding_size * text_max_len*NUM_CHANNEL
#         )  # 64,512
#         self.linear1 = nn.Linear(embedding_size, embedding_size * text_max_len)

#     def forward(self, x):
#         """
#         X: tensor of dim batch_size, max_text_len and embed_dim plz take other things will work accordingly
#         just take care of it.

#         """
#         embedding = self.embed(x)  # b,t,embed

#         batch_size = embedding.shape[0]
#         xxx = embedding.reshape(batch_size, -1)  # b,t*embed
#         import pdb;pdb.set_trace()
#         out = self.fc(xxx)
#         """embed content force"""
#         xx_new = self.linear(embedding.view(embedding.size(0), -1)).view(
#             embedding.size(0), embedding.size(1), embedding.size(3)
#         )  # b, text_max_len, 64

#         ts = xx_new.shape[1]  # b,512,8,27
#         height_reps = IMG_HEIGHT  # 8 [-2]
#         width_reps = max(1, IMG_WIDTH // ts)  # [-2] 27
#         tensor_list = list()
#         for i in range(ts):
#             text = [xx_new[:, i : i + 1]]  # b, text_max_len, 512
#             tmp = torch.cat(text * width_reps, dim=1)
#             tensor_list.append(tmp)

#         padding_reps = IMG_WIDTH % ts
#         if padding_reps:
#             embedded_padding_char = self.embed(
#                 torch.full((1, 1), 2, dtype=torch.long, device=device)
#             )
#             # embedded_padding_char = self.linear1(embedded_padding_char)
#             padding = embedded_padding_char.repeat(batch_size, padding_reps, 1)
#             tensor_list.append(padding)

#         res = torch.cat(
#             tensor_list, dim=1
#         )  # b, text_max_len * width_reps + padding_reps, 512
#         res = res.permute(0, 2, 1).unsqueeze(
#             2
#         )  # b, 512, 1, text_max_len * width_reps + padding_reps
#         final_res = torch.cat([res] * height_reps, dim=2)
#         return out, final_res  # 2,85,5440 , batch,c,h,w
