import torch

from Scripts.Envs.utils import load_d_agent, episode_reward


class KESEnv():
    def __init__(self, dataset, model_name='DKT', dataset_name='assist09',device=torch.device('cpu')):
        self.skill_num = dataset.feats_num
        self.model = load_d_agent(model_name, dataset_name, self.skill_num).to(device)
        self.targets = None
        self.states = (None, None)
        self.initial_score = None
        self.device = device

    def exam(self, targets, states):
        scores = []
        for i in range(targets.shape[1]):
            score, states = self.model.learn(targets[:, i:i + 1].to(self.device), states)  # (B, 1)
            scores.append(score)
        return torch.mean(torch.cat(scores, dim=1), dim=1)  # (B,)

    def begin_episode(self, targets, initial_logs):
        self.targets = targets
        initial_score, initial_log_scores, states = self.begin_episode_(targets, initial_logs)
        self.initial_score = initial_score
        self.states = states
        return initial_log_scores

    def begin_episode_(self, targets, initial_logs=None):
        states = (None, None)
        score = None
        if initial_logs is not None:
            score, states = self.model.learn_lstm(initial_logs.to(self.device))
        initial_score = self.exam(targets, states)
        return initial_score, score, states

    def n_step(self, learning_path, binary=False):
        scores, states = self.model.learn(learning_path, self.states)  # 调用模型的 learn 方法
        self.states = states  # 更新状态
        if binary:
            scores = (scores > 0.5).float()  # 将得分转换为二进制（0 或 1）
        return scores

    def end_episode(self, length):
        final_score, reward,reward_punish = self.end_episode_(self.initial_score, self.targets, *self.states,length)
        # if 'score' in kwargs:
        #     return final_score, reward
        return reward,reward_punish

    # 学到最后一轮，根据以前学习过的知识的状态，计算目标知识点的得分，并计算Et
    def end_episode_(self, initial_score, targets, states1, states2,length):
        final_score = self.exam(targets, (states1, states2))
        reward = episode_reward(initial_score, final_score, 1).unsqueeze(-1)
        punish = (2 * (length - 1) / 29).unsqueeze(-1)
        reward_punish = reward - punish
        return final_score, reward,reward_punish

