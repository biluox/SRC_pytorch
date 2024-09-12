import torch
from torch import nn

from KTScripts.BackModels import Transformer, MLP
import torch.nn.functional as F


class SRC(nn.Module):
    def __init__(self, skill_num, input_size, weight_size, hidden_size, dropout, allow_repeat=False,
                 with_kt=False):
        super(SRC, self).__init__()
        self.embedding = nn.Embedding(skill_num, input_size)
        self.l1 = nn.Linear(input_size + 1, input_size)
        self.l2 = nn.Linear(input_size, hidden_size)
        self.state_encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.path_encoder = Transformer(hidden_size, hidden_size, 0.0, head=1, b=1, transformer_mask=False)
        self.W1 = nn.Linear(hidden_size, weight_size, bias=False)  # blending encoder
        self.W2 = nn.Linear(hidden_size, weight_size, bias=False)  # blending decoder
        self.vt = nn.Linear(weight_size, 1, bias=False)  # scaling sum of enc and dec by v.T
        self.decoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)

        # 知识追踪辅助模块
        if with_kt:
            self.ktRnn = nn.LSTM(hidden_size, hidden_size, batch_first=True)
            self.ktMlp = MLP(hidden_size, [hidden_size // 2, hidden_size // 4, 1], dropout=dropout)

        self.allow_repeat = allow_repeat
        self.withKt = with_kt
        self.skill_num = skill_num

    def begin_episode(self, targets, initial_logs, initial_log_scores):
        # targets: (B, K), where K is the number of targets in this batch
        targets = self.l2(self.embedding(targets).mean(dim=1, keepdim=True))  # (B, 1, H)

        if initial_logs is not None:
            states = self.step(initial_logs, initial_log_scores, None)
        else:
            zeros = torch.zeros_like(targets).transpose(0, 1)  # (1, B, H)
            states = (zeros, zeros)

        return targets, states

    def step(self, x, score, states):
        x = self.embedding(x)  # 将输入 x 转换为嵌入
        x = self.l1(torch.cat((x, score.unsqueeze(-1)), dim=-1))  # 连接 score 并通过全连接层

        # 处理 LSTM 的状态
        _, states = self.state_encoder(x, states)
        return states

    def forward(self, targets, initial_logs, initial_log_scores, origin_path, n):
        targets, states = self.begin_episode(targets, initial_logs, initial_log_scores)
        inputs = self.l2(self.embedding(origin_path))
        encoder_states = self.path_encoder(inputs) + inputs

        blend1 = self.W1(encoder_states + encoder_states.mean(dim=1, keepdim=True) + targets)  # (B, L, W)
        decoder_input = torch.zeros_like(inputs[:, 0:1])  # (B, 1, I)
        probs, paths = [], []
        selecting_s = []
        a1 = torch.arange(inputs.shape[0], device=inputs.device)
        selected = torch.zeros_like(inputs[:, :, 0], dtype=torch.bool)
        minimum_fill = torch.full_like(selected, -1e9, dtype=torch.float32)
        hidden_states = []

        for i in range(n):
            hidden, states = self.decoder(decoder_input, states)
            if self.withKt and i > 0:
                hidden_states.append(hidden)

            # 计算每个解码器时间步的融合表示
            blend2 = self.W2(hidden)  # (B, 1, W)
            blend_sum = blend1 + blend2  # (B, L, W)
            out = self.vt(blend_sum).squeeze(-1)  # (B, L)

            if not self.allow_repeat:
                out = torch.where(selected, minimum_fill, out.detach())
                out = F.softmax(out, dim=-1)
                if self.training:
                    selecting = torch.multinomial(out, 1).squeeze(-1)
                else:
                    selecting = torch.argmax(out, dim=1)
                selected[a1, selecting] = True
            else:
                out = F.softmax(out, dim=-1)
                selecting = torch.multinomial(out, 1).squeeze(-1)

            selecting_s.append(selecting)
            path = origin_path[a1, selecting]
            decoder_input = encoder_states[a1, selecting].unsqueeze(1)  # (B, 1, H)
            out = out[a1, selecting]
            paths.append(path)
            probs.append(out)

        probs = torch.stack(probs, dim=1)
        paths = torch.stack(paths, dim=1)  # (B, n)
        selecting_s = torch.stack(selecting_s, dim=1)

        if self.withKt and self.training:
            hidden_states.append(self.decoder(decoder_input, states)[0])
            hidden_states = torch.cat(hidden_states, dim=1)
            kt_output = torch.sigmoid(self.ktMlp(hidden_states))
            result = [paths, probs, selecting_s, kt_output]
            return result

        return paths, probs, selecting_s

    def backup(self, targets, initial_logs, initial_log_scores, origin_path, selecting_s):
        targets, states = self.begin_episode(targets, initial_logs, initial_log_scores)
        inputs = self.l2(self.embedding(origin_path))
        encoder_states = inputs
        encoder_states = self.path_encoder(encoder_states)
        encoder_states += inputs

        blend1 = self.W1(encoder_states + encoder_states.mean(dim=1, keepdim=True) + targets)  # (B, L, W)
        selecting_states = encoder_states[torch.arange(encoder_states.shape[0]).unsqueeze(1), selecting_s]
        selecting_states = torch.cat((torch.zeros_like(selecting_states[:, 0:1]), selecting_states[:, :-1]), dim=1)
        hidden_states, _ = self.decoder(selecting_states, states)
        blend2 = self.W2(hidden_states)  # (B, n, W)
        blend_sum = blend1.unsqueeze(1) + blend2.unsqueeze(2)  # (B, n, L, W)
        out = self.vt(blend_sum).squeeze(-1)  # (B, n, L)

        # Masking probabilities according to output order
        mask = selecting_s.unsqueeze(1).repeat(1, selecting_s.shape[-1], 1)  # (B, n, n)
        mask = torch.tril(mask + 1, diagonal=-1).view(-1, mask.shape[-1])
        out = out.view(-1, out.shape[-1])
        out = torch.cat((torch.zeros_like(out[:, 0:1]), out), dim=-1)
        out[torch.arange(out.shape[0]).unsqueeze(1), mask] = -1e9
        out = out[:, 1:].view(origin_path.shape[0], -1, origin_path.shape[1])

        out = F.softmax(out, dim=-1)
        probs = out.gather(2, selecting_s.unsqueeze(-1)).squeeze(-1)
        return probs
