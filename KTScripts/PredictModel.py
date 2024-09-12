import torch
import torch.nn as nn
import torch.nn.functional as F

from KTScripts.BackModels import MLP


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
            training = self.rnn.training
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
