import torch
import torch.nn as nn
import torch.nn.functional as F

from KTScripts.BackModels import MLP, CoKT


class PredictModel(nn.Module):
    def __init__(self, feat_nums, embed_size, hidden_size, pre_hidden_sizes, dropout, output_size=1, with_label=True,
                 model_name='DKT'):
        super(PredictModel, self).__init__()
        self.item_embedding = nn.Embedding(feat_nums, embed_size)
        self.mlp = MLP(hidden_size, pre_hidden_sizes + [output_size],
                       dropout=dropout)  # Assuming MLP is defined elsewhere
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.with_label = with_label
        self.move_label = True
        input_size_label = embed_size + 1 if with_label else embed_size
        self.model_name = model_name

        if model_name == 'DKT':
            self.rnn = nn.LSTM(input_size_label, hidden_size, batch_first=True)
            self.return_tuple = True
        elif model_name == 'GRU4Rec':
            self.rnn = nn.GRU(input_size_label, hidden_size, batch_first=True)
            self.move_label = False

    def forward(self, x, y, mask=None):
        # x: (B, L), y: (B, L)
        x = self.item_embedding(x)  # Embedding
        if self.with_label:
            if self.move_label:
                y_ = torch.cat((torch.zeros_like(y[:, 0:1]), y[:, :-1]), dim=1)
            else:
                y_ = y
            x = torch.cat((x, y_.unsqueeze(-1)), dim=-1)  # Concatenate along the last dimension

        o, _ = self.rnn(x)  # Get RNN output
        if mask is not None:
            o = o.masked_select(mask.unsqueeze(-1))  # Apply mask to output
            y = y.masked_select(mask)

        o = o.view(-1, self.hidden_size)  # Reshape for MLP
        # y = y.view(-1)
        o = self.mlp(o)  # MLP processing

        if self.model_name == 'GRU4Rec':
            o = F.softmax(o, dim=-1)
        else:
            o = torch.sigmoid(o).squeeze(-1)

        return o, y

    def learn_lstm(self, x, states1=None, states2=None, get_score=True):
        if states1 is None:
            states = None
        else:
            states = (states1, states2)
        return self.learn(x, states, get_score)
    # x是输入的学习路径，通过lstm获得分数
    def learn(self, x, states=None, get_score=True):
        x = self.item_embedding(x)  # (B, L, E)
        o = torch.zeros_like(x[:, 0:1, 0:1])  # (B, 1, 1)
        os = []

        for i in range(x.shape[1]):
            x_i = x[:, i:i + 1]
            if self.with_label and get_score:
                x_i = torch.cat((x_i, o), dim=-1)
            with torch.no_grad():
                o, states = self.rnn(x_i, states)
            if get_score:
                o = self.mlp(o.squeeze(1))
                o = torch.sigmoid(o).unsqueeze(1)
            os.append(o)

        os = torch.cat(os, dim=1)  # (B, L) or (B, L, H)
        if self.output_size == 1:
            os = os.squeeze(-1)
        return os, states

    def GRU4RecSelect(self, origin_paths, n, skill_num, initial_logs):
        ranked_paths = [None] * n
        batch_size = origin_paths.shape[0]

        # 创建索引和选中路径的标志
        a1 = torch.arange(batch_size).unsqueeze(-1)  # [batch_size, 1]
        selected_paths = torch.ones((batch_size, skill_num), dtype=torch.bool)
        selected_paths[a1, origin_paths] = False

        path = initial_logs  # 初始路径
        states = None  # 初始状态
        a1 = a1.squeeze(-1)  # [batch_size]

        for i in range(n):
            o, states = self.learn(path, states)  # 调用学习方法
            o = o[:, -1]  # 取最后一个时间步的输出
            o[selected_paths] = -1  # 将已选择的路径的得分设为 -1
            path = torch.argmax(o, axis=-1)  # 选择得分最高的路径
            ranked_paths[i] = path  # 存储当前选择的路径
            selected_paths[a1, path] = True  # 更新已选择的路径标记
            path = path.unsqueeze(1)  # 扩展维度以进行下一次迭代

        ranked_paths = torch.stack(ranked_paths, dim=-1)  # 堆叠路径
        return ranked_paths


class PredictRetrieval(nn.Module):
    def __init__(self, feat_nums, input_size, hidden_size, pre_hidden_sizes, dropout, with_label=True, model_name='CoKT'):
        super(PredictRetrieval, self).__init__()
        self.with_label = with_label
        self.item_embedding = nn.Embedding(feat_nums, input_size)  # 假设 feat_nums 是词汇量
        self.rnn = CoKT(input_size + 1, hidden_size, dropout, head=2)  # 假设 CoKT 是定义好的 RNN 模型
        self.mlp = nn.Linear(hidden_size, 1)  # 假设 hidden_size 是 RNN 输出的大小

    def forward(self, intra_x, inter_his, inter_r, y, mask, inter_len):
        intra_x = self.item_embedding(intra_x)
        if self.with_label:
            y_ = torch.cat((torch.zeros_like(y[:, 0:1]), y[:, :-1].unsqueeze(-1)), dim=1).float()
            intra_x = torch.cat((intra_x, y_), dim=-1)

        inter_his = torch.cat((self.item_embedding(inter_his[:, :, :, 0]),
                               inter_his[:, :, :, 1:].float()), dim=-1)
        inter_r = torch.cat((self.item_embedding(inter_r[:, :, :, 0]),
                             inter_r[:, :, :, 1:].float()), dim=-1)

        o = self.rnn(intra_x, inter_his, inter_r, mask, inter_len)
        o = torch.sigmoid(self.mlp(o)).squeeze(-1)
        y = y.masked_select(mask).view(-1)
        return o, y

    def learn(self, intra_x, inter_his, inter_r, inter_len, states=None):
        his_len, seq_len = 0, intra_x.shape[1]
        intra_x = self.item_embedding(intra_x)  # (B, L, I)
        intra_h = None

        if states is not None:
            his_len = states[0].shape[1]
            intra_x = torch.cat((intra_x, states[0]), dim=1)  # (B, L_H + L, I)
            intra_h = states[1]

        o = torch.zeros_like(intra_x[:, 0:1, 0:1])
        inter_his = torch.cat((self.item_embedding(inter_his[:, :, :, 0]),
                               inter_his[:, :, :, 1:].float()), dim=1)
        inter_r = torch.cat((self.item_embedding(inter_r[:, :, :, 0]),
                             inter_r[:, :, :, 1:].float()), dim=-1)

        M_rv, M_pv = self.rnn.deal_inter(inter_his, inter_r, inter_len)  # (B, L, R, H)
        os = []

        for i in range(seq_len):
            o, intra_h = self.rnn.step(M_rv[:, i], M_pv[:, i], intra_x[:, :i + his_len + 1], o, intra_h)
            o = torch.sigmoid(self.mlp(o))
            os.append(o)

        o = torch.cat(os, dim=1)  # (B, L, 1)
        return o, (intra_x, intra_h)


class ModelWithLoss(nn.Module):
    def __init__(self, model, criterion):
        super(ModelWithLoss, self).__init__()
        self.model = model
        self.criterion = criterion

    def forward(self, *data):
        output_data = self.model(*data)
        loss = self.criterion(*output_data)
        return loss, output_data

    def output(self, *data):
        output_data = self.model(*data)
        loss = self.criterion(*output_data)
        return loss, output_data
