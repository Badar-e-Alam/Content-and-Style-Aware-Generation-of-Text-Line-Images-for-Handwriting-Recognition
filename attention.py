import torch
from torch import nn


class Head(nn.Module):
    def __init__(self, infeature, out_feature):
        super().__init__()

        # Q, K, V weight matrices for each head
        self.wq = nn.Linear(infeature, out_feature, bias=False)
        self.wk = nn.Linear(infeature, out_feature, bias=False)
        self.wv = nn.Linear(infeature, out_feature, bias=False)
        self.scale = 1.0 / (infeature**0.5)

        # Output projection matrix
        self.proj = nn.Linear(infeature, out_feature, bias=False)

    def forward(self, x):
        # x shape: [batch_size, num_channels* image_height* image_width]
        batch, CHW = x.shape
        # Reshape input to [batch_size, num_channels*image_height, image_width]
        # x = x.reshape(x.size(0), -1, x.size(1))

        # Compute Q, K, V matrices for each head
        q = self.wq(x)  # q shape: [batch_size, num_channels*image_height, d_model]
        k = self.wk(x)  # k shape: [batch_size, num_channels*image_height, d_model]
        v = self.wv(x)  # v shape: [batch_size, num_channels*image_height, d_model]
        weights = torch.matmul(q, k.transpose(-2, -1))
        weights = weights * self.scale
        weights = nn.functional.softmax(weights, dim=-1)

        # Apply attention weights to values
        output = torch.matmul(weights, v)
        return output


class MultiHeadAttention(nn.Module):
    "Multiple heads of the self_attention in parallel"

    def __init__(self, infeature, out_feature, num_heads, dropout):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                Head(infeature=infeature, out_feature=out_feature)
                for _ in range(num_heads)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads])

        out = self.dropout(out)
        print(out.shape)
        return out


class Cross_attention(nn.Module):
    def __init__(self, infeature, out_feature):
        super().__init__()

        # Q, K, V weight matrices for each head
        self.wq = nn.Linear(infeature, out_feature, bias=False)
        self.wk = nn.Linear(infeature, out_feature, bias=False)
        self.wv = nn.Linear(infeature, out_feature, bias=False)
        self.scale = 1.0 / (infeature**0.5)

        # Output projection matrix
        self.proj = nn.Linear(infeature, out_feature, bias=False)

    def forward(self, decoder, encoder):
        # x shape: [batch_size, num_channels* image_height* image_width]
        # batch, CHW = x.shape

        # Reshape input to [batch_size, num_channels*image_height, image_width]
        # x = x.reshape(x.size(0), -1, x.size(1))

        # Compute Q, K, V matrices for each head
        q = self.wq(
            decoder
        )  # q shape: [batch_size, num_channels*image_height, d_model]
        k = self.wk(
            encoder
        )  # k shape: [batch_size, num_channels*image_height, d_model]
        v = self.wv(
            decoder
        )  # v shape: [batch_size, num_channels*image_height, d_model]
        weights = torch.matmul(q, k.transpose(-2, -1))
        weights = weights * self.scale
        weights = nn.functional.softmax(weights, dim=-1)

        # Apply attention weights to values
        output = torch.matmul(weights, v)
        return output


class MultiHead_CrossAttention(nn.Module):
    "Multiple heads of the self_attention in parallel"

    def __init__(self, infeature, out_feature, num_heads, dropout):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                Cross_attention(infeature=infeature, out_feature=out_feature)
                for _ in range(num_heads)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder, decoder):
        out = torch.cat([h.forward(encoder, decoder) for h in self.heads])

        out = self.dropout(out)
        print(out.shape)
        return out
