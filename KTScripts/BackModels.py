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
        q_1, q_2 = query.size(0), query.size(1)
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
        self.FFNs = nn.ModuleList([FeedForward(head, hidden_size, dropout_rate) for _ in range(b)])
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
            inputs = self.FFNs[i](inputs, mask)

        return inputs


class CoKT(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate, head=2):
        super(CoKT, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        self.ma_inter = MultiHeadedAttention(head, hidden_size, dropout_rate,
                                             input_sizes=(hidden_size + input_size - 1,
                                                          hidden_size + input_size - 1,
                                                          hidden_size + input_size,
                                                          hidden_size))
        self.ma_intra = MultiHeadedAttention(head, hidden_size, dropout_rate,
                                             input_sizes=(input_size - 1,
                                                          input_size - 1,
                                                          hidden_size + 1,
                                                          hidden_size))
        self.wr = nn.Parameter(torch.randn(1, 1, 2))
        self.ln = nn.Linear(2 * hidden_size + input_size - 1, hidden_size)

    def forward(self, intra_x, inter_his, inter_r, intra_mask, inter_len):
        # (B, L, I), (B, L*R, L, I), (B, L, R, I), (B, L), (B, L, R)
        intra_mask = intra_mask.unsqueeze(-1)  # (B, L, 1)

        intra_h, _ = self.rnn(intra_x)  # (B, L, H)
        intra_h_mask = intra_h.masked_select(intra_mask).view(-1, self.hidden_size)  # (seq_sum, H)
        intra_x_mask = intra_x.masked_select(intra_mask).view(-1, self.input_size)  # (seq_sum, I)

        # inter attention
        intra_mask_ = intra_mask.unsqueeze(-1)  # (B, L, 1, 1)
        inter_his, _ = self.rnn(
            inter_his.view(inter_his.size(0) * inter_his.size(1), *inter_his.size()[2:]))  # (B*L*R, L, H)
        inter_his = inter_his[torch.arange(inter_his.size(0)), inter_len.view(-1) - 1]  # (B*L*R, H)
        inter_his = inter_his.view(*inter_len.size(), self.hidden_size)  # (B, L, R, H)
        inter_his = inter_his.masked_select(intra_mask_).view(-1, *inter_his.size()[2:])  # (seq_sum, R, H)
        inter_r = inter_r.masked_select(intra_mask_).view(-1, *inter_r.size()[2:])  # (seq_sum, R, I)

        M_rv = torch.cat((inter_his, inter_r), dim=-1).view(*inter_r.size()[:2],
                                                            self.hidden_size + self.input_size)  # (seq_sum, R, H+I)
        M_pv = M_rv[:, :, :-1].view(*M_rv.size()[:2], self.input_size + self.hidden_size - 1)  # (seq_sum, R, H+I-1)
        m_pv = torch.cat((intra_h_mask, intra_x_mask[:, :-1]), dim=1).view(M_pv.size(0), 1,
                                                                           self.hidden_size + self.input_size - 1)  # (seq_sum, 1, H+I-1)
        v_v = self.ma_inter(m_pv, M_pv, M_rv).squeeze(1)  # (seq_sum, H)

        # intra attention
        intra_x_p = intra_x[:, :, :-1]  # (B, L, I-1)
        intra_h_p = torch.cat((intra_h, intra_x[:, :, -1:]), dim=-1)  # (B, L, H+1)
        intra_mask_attn = torch.tril(torch.ones((1, 1, intra_x_p.size(1), intra_x_p.size(1)), dtype=torch.bool))
        v_h = self.ma_intra(intra_x_p, intra_x_p, intra_h_p, mask=intra_mask_attn)  # (B, L, H)
        v_h = v_h.masked_select(intra_mask).view(-1, v_h.size(-1))  # (seq_sum, H)

        v = torch.sum(F.softmax(self.wr, dim=-1) * torch.stack((v_v, v_h), dim=-1), dim=-1)  # (seq_sum, H)
        return self.ln(torch.cat((v, intra_h_mask, intra_x_mask[:, :-1]), dim=1))  # (seq_sum, H)

    def deal_inter(self, inter_his, inter_r, inter_len):
        inter_his, _ = self.rnn(
            inter_his.view(inter_his.size(0) * inter_his.size(1), *inter_his.size()[2:]))  # (B*L*R, L, H)
        inter_his = inter_his[torch.arange(inter_his.size(0)), inter_len.view(-1) - 1]  # (B*L*R, H)
        inter_his = inter_his.view(*inter_len.size(), self.hidden_size)  # (B, L, R, H)
        M_rv = torch.cat((inter_his, inter_r), dim=-1).view(*inter_r.size()[:3],
                                                            self.hidden_size + self.input_size)  # (B, L, R, H+I)
        M_pv = M_rv[:, :, :-1].view(*M_rv.size()[:3], self.input_size + self.hidden_size - 1)  # (B, L, R, H+I)
        return M_rv, M_pv

    def step(self, m_rv, M_pv, intra_x, o, intra_h_p=None):
        # M_*: (B, R, H)
        # intra_h_p:(B, L-1, H+1), with the y
        # intra_x:(B, L, I-1), without the y
        # o: y from last step
        intra_h_next, _ = self.rnn(torch.cat((intra_x[:, -1:], o), dim=-1),
                                   None if intra_h_p is None else intra_h_p[:, -1, :-1].unsqueeze(0))  # (B, 1, H)
        m_pv = torch.cat((intra_h_next, intra_x[:, -1:]), dim=-1)  # (B, 1, H+I-1)
        v_v = self.ma_inter(m_pv, M_pv, m_rv)  # (B, 1, H)

        intra_x_p = intra_x
        intra_h_next = torch.cat((intra_h_next, o), dim=-1)
        intra_h_p = intra_h_next if intra_h_p is None else torch.cat((intra_h_p, intra_h_next), dim=1)  # (B, L, H+1)

        # Sequence mask
        v_h = self.ma_intra(intra_x_p[:, -1:], intra_x_p, intra_h_p)  # (B, 1, H), only query last target item
        v = torch.sum(F.softmax(self.wr, dim=-1) * torch.stack((v_v, v_h), dim=-1), dim=-1)  # (B, 1, H)
        return self.ln(torch.cat((v, intra_h_p[:, -1:, :-1], intra_x[:, -1:]), dim=-1)), intra_h_p  # (B, 1, 2*H+I-1)
