import math

import torch
import torch.nn.functional as F
from torch import nn


def nll_loss(y_, y, mask):
    # 添加一个小的常数以防止对数零
    y_ = y_ + 1e-9
    # 计算负对数似然损失
    loss = F.nll_loss(torch.log(y_), y, reduction='none')
    # 计算加权损失
    weighted_loss = loss * mask
    return weighted_loss.sum() / mask.sum()


class MLP(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 norm_layer=None,
                 activation_layer=nn.ReLU,
                 bias=True,
                 dropout=0.5):
        super(MLP, self).__init__()

        layers = []
        in_dim = in_channels

        for hidden_dim in hidden_channels[:-1]:
            layers.append(nn.Linear(in_dim, hidden_dim, bias=bias))
            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim))
            layers.append(activation_layer())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, hidden_channels[-1], bias=bias))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class PositionalEncoding(nn.Module):
    """Implement the Positional Encoding function."""

    def __init__(self, d_model, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # 增加 batch 维度
        self.register_buffer('pe', pe)  # 注册为 buffer，不会被更新

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class MultiHeadedAttention(nn.Module):
    def __init__(self, head, hidden_sizes, dropout_rate, input_sizes=None):
        super(MultiHeadedAttention, self).__init__()
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes] * 4
        if input_sizes is None:
            input_sizes = hidden_sizes

        for hidden_size in hidden_sizes:
            assert hidden_size % head == 0

        self.head = head
        self.head_size = hidden_sizes[0] // head
        self.hidden_size = hidden_sizes[-1]
        self.d_k = math.sqrt(hidden_sizes[0] // head)

        # Linear layers for query, key, and value
        self.linear_s = nn.ModuleList([
            nn.Linear(input_size, hidden_size) for input_size, hidden_size in zip(input_sizes, hidden_sizes)
        ])
        self.dropout = nn.Dropout(p=dropout_rate)

    def attention(self, query, key, value, mask=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.d_k
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)  # Masking

        p_attn = torch.softmax(scores, dim=-1)
        return torch.matmul(p_attn, value), p_attn

    def forward(self, query, key, value, mask=None):
        q_1,q_2 = query.size(0),query.size(1)
        query, key, value = [
            l(x).view(x.size(0), x.size(1), self.head, self.head_size).transpose(1, 2)
            for l, x in zip(self.linear_s, (query, key, value))
        ]

        x, _ = self.attention(query, key, value, mask)  # (B, Head, L, D_H)
        x = x.transpose(1, 2).contiguous().view(-1, self.head * self.head_size)  # Flatten
        return self.linear_s[-1](x).view(q_1, q_2, self.hidden_size)


class FeedForward(nn.Module):
    def __init__(self, head, input_size, dropout_rate):
        super(FeedForward, self).__init__()
        self.mh = MultiHeadedAttention(head, input_size, dropout_rate)  # 假设已经定义了 MultiHeadedAttention 类
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.activate = nn.LeakyReLU()
        self.ln1 = nn.LayerNorm(input_size)
        self.ln2 = nn.LayerNorm(input_size)
        self.fc1 = nn.Linear(input_size, input_size)
        self.fc2 = nn.Linear(input_size, input_size)

    def forward(self, s, mask):
        s = s + self.dropout1(self.mh(s, s, s, mask))
        s = self.ln1(s)
        s_ = self.activate(self.fc1(s))
        s_ = self.dropout2(self.fc2(s_))
        s = self.ln2(s + s_)
        return s


class Transformer(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate, head=1, b=1, position=False, transformer_mask=True):
        super(Transformer, self).__init__()
        self.position = position

        if position:
            self.pe = PositionalEncoding(input_size, dropout_rate)
        self.fc = nn.Linear(input_size, hidden_size)
        self.SAs = nn.ModuleList([MultiHeadedAttention(head, hidden_size, dropout_rate) for _ in range(b)])
        self.FFNs = nn.ModuleList([FeedForward(head,hidden_size, dropout_rate) for _ in range(b)])
        self.b = b
        self.transformer_mask = transformer_mask

    def forward(self, inputs, mask=None):
        if self.position:
            inputs = self.pe(inputs)
        inputs = self.fc(inputs)
        max_len = inputs.shape[1]

        if self.transformer_mask:
            mask = torch.tril(torch.ones(1, max_len, max_len, dtype=torch.bool))
        elif mask is not None:
            mask = mask.unsqueeze(1)  # (B, 1, L)

        if mask is not None:
            mask = mask.unsqueeze(1)  # (B, 1, 1, L) or (B, 1, L, L)

        for i in range(self.b):
            inputs = self.SAs[i](inputs, inputs, inputs, mask)
            inputs = self.FFNs[i](inputs,mask)

        return inputs
