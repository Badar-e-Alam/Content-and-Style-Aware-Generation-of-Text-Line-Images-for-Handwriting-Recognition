import torch
from torch import nn
from einops import rearrange

class Head(nn.Module):
    def __init__(self, infeature, out_feature):
        super().__init__()

        # Q, K, V weight matrices for each head
        self.wq = nn.Linear(infeature, out_feature, bias=False)
        self.wk = nn.Linear(infeature, out_feature, bias=False)
        self.wv = nn.Linear(infeature, out_feature, bias=False)
        self.scale = 1.0 / (infeature ** 0.5)

        # Output projection matrix
        self.proj = nn.Linear(infeature, out_feature, bias=False)

    def forward(self, x):
        # x shape: [batch_size, num_channels* image_height* image_width]
        #batch, CHW = x.shape
        # Reshape input to [batch_size, num_channels*image_height, image_width]
        # x = x.reshape(x.size(0), -1, x.size(1))

        # Compute Q, K, V matrices for each head
        q = self.wq(x)  # q shape: [batch_size, num_channels*image_height, image_width]
        k = self.wk(x)  # k shape: [batch_size, num_channels*image_height, image_width]
        v = self.wv(x)  # v shape: [batch_size, num_channels*image_height, image_width]
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
        return out


class Cross_attention(nn.Module):
    def __init__(self, infeature, out_feature):
        super().__init__()

        # Q, K, V weight matrices for each head
        self.wq = nn.Linear(infeature, out_feature, bias=False)
        self.wk = nn.Linear(infeature, out_feature, bias=False)
        self.wv = nn.Linear(infeature, out_feature, bias=False)
        self.scale = 1.0 / (infeature ** 0.5)

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
        return out



class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads 
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    

class CrossAttention(nn.Module):
        def __init__(self, query_dim, key_dim, value_dim, heads=8, dim_head=64, dropout=0.):
            super().__init__()
            inner_dim = dim_head * heads
            project_out = not (heads == 1 and dim_head == query_dim)

            self.heads = heads
            self.scale = dim_head ** -0.5

            self.norm = nn.LayerNorm(query_dim)

            self.attend = nn.Softmax(dim=-1)
            self.dropout = nn.Dropout(dropout)

            self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
            self.to_k = nn.Linear(key_dim, inner_dim, bias=False)
            self.to_v = nn.Linear(value_dim, inner_dim, bias=False)

            self.to_out = nn.Sequential(
                nn.Linear(inner_dim, query_dim),
                nn.Dropout(dropout)
            ) if project_out else nn.Identity()

        def forward(self, query, key, value):
            query = self.norm(query)

            q = rearrange(self.to_q(query), 'b n (h d) -> b h n d', h=self.heads)
            k = rearrange(self.to_k(key), 'b n (h d) -> b h n d', h=self.heads)
            v = rearrange(self.to_v(value), 'b n (h d) -> b h n d', h=self.heads)

            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

            attn = self.attend(dots)
            attn = self.dropout(attn)

            out = torch.matmul(attn, v)
            out = rearrange(out, 'b h n d -> b n (h d)')
            return self.to_out(out)
