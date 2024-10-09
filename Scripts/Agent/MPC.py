import torch
import torch.nn as nn
import torch.nn.functional as F

from KTScripts.BackModels import MLP


class MPC(nn.Module):
    def __init__(self, skill_num, input_size, hidden_size, pre_hidden_sizes, dropout, hor):
        super(MPC, self).__init__()
        self.l1 = nn.Sequential(
            nn.Linear(input_size + 1, input_size),
            nn.LeakyReLU(),
            nn.Dropout(p=1 - dropout)
        )
        self.embed = nn.Embedding(skill_num, input_size)
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.decoder = MLP(hidden_size, pre_hidden_sizes, dropout=dropout)
        self.hor = hor

    def sample(self, b, n):
        candidate_order = torch.rand(b, n)
        _, candidate_order = torch.sort(candidate_order)  # (B*Hor, n)
        return candidate_order

    def test(self, targets, states):
        x, states = self.encoder(targets, states)
        x = x[:, -1]
        x = torch.sigmoid(self.decoder(x).squeeze())
        return x

    def begin_episode(self, targets, initial_logs, initial_log_scores):
        targets = torch.mean(self.embed(targets), dim=1, keepdim=True)  # (B, 1, I)
        targets_repeat = targets.repeat(self.hor, 1, 1)  # (B*Hor, 1, I)

        if initial_logs is not None:
            states = self.step(initial_logs, initial_log_scores, None)
        else:
            zeros = torch.zeros_like(targets).transpose(0, 1)  # (1, B, Hor)
            states = (zeros, zeros)

        return targets, targets_repeat, states

    def step(self, x, score, states):
        x = self.embed(x)
        if score is not None:
            x = self.l1(torch.cat((x, score.unsqueeze(-1)), dim=-1))
        _, states = self.encoder(x, states)
        return states

    def forward(self, targets, initial_logs, initial_log_scores, origin_path, n):
        targets, targets_repeat, states = self.begin_episode(targets, initial_logs, initial_log_scores)
        unselected = torch.ones_like(origin_path, dtype=torch.bool)
        a1 = torch.arange(targets.shape[0]).unsqueeze(1).repeat(1, n).repeat(self.hor, 1)  # (B*Hor, n)
        result_path = []
        target_args = None
        max_len, batch = origin_path.shape[1], targets_repeat.shape[0]

        for i in range(n):
            candidate_args = self.sample(batch, max_len - i)[:, :(n - i)]  # (B*Hor, n-i)
            if i > 0:
                candidate_args = candidate_args.view(-1, self.hor, n - i)
                candidate_args[:, -1] = target_args
                candidate_args = candidate_args.view(-1, n - i)

            candidate_paths = origin_path[unselected].view(-1, max_len - i)[a1, candidate_args]  # (B*Hor, n-i)
            a1 = a1[:, :-1]

            states_repeat = [state.repeat(1, self.hor,1) for state in states]
            _, states_repeat = self.encoder(self.embed(candidate_paths), states_repeat)  # (B*Hor, L, H)
            candidate_scores = self.test(targets_repeat, states_repeat).view(-1, self.hor)
            selected_hor = torch.argmax(candidate_scores, dim=1)  # (B,)

            target_args = candidate_args.view(-1, self.hor, n - i)[a1, selected_hor]
            target_path = candidate_paths.view(-1, self.hor, n - i)[a1, selected_hor]
            result_path.append(target_path[:, 0])

            modified = unselected.clone().view(unselected.shape[0], -1)
            modified[a1, target_args[:, 0]] = False
            unselected = modified.view(unselected.shape)

            target_args = torch.where(target_args > target_args[:, :1], target_args - 1, target_args)[:, 1:]

            states = self.step(target_path[:, :1], None, states)

        result_path = torch.stack(result_path, dim=1)
        return result_path

    def backup(self, targets, initial_logs, initial_log_scores, result_path):
        _, _, states = self.begin_episode(targets, initial_logs, initial_log_scores)
        history_states, _ = self.encoder(self.embed(result_path), states)
        history_scores = torch.sigmoid(self.decoder(history_states)).squeeze(-1)  # (B, L)
        return history_scores