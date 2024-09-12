
'''
Et的计算
initial_score: 任务开始时的得分。
final_score: 任务结束时的得分。
full_score: 任务的满分，通常是该任务能达到的最高得分。
'''
import os
from argparse import ArgumentParser

import torch

from KTScripts.options import get_exp_configure, get_options
from KTScripts.utils import load_model


def episode_reward(initial_score, final_score, full_score) -> (int, float):
    delta = final_score - initial_score
    normalize_factor = full_score - initial_score + 1e-9
    return delta / normalize_factor

# 加载DKT
def load_d_agent(model_name, args, skill_num, with_label=True):
    model_parameters = get_exp_configure(model_name)
    model_parameters.update({'feat_nums': skill_num, 'model': model_name, 'without_label': not with_label})
    if model_name == 'GRU4Rec':
        model_parameters.update({'output_size': skill_num})
    model = load_model(model_parameters)
    parser = ArgumentParser("LearningPath-Planing")
    args = get_options(parser)
    model_path = os.path.join(args.save_dir, args.exp_name)
    # if not with_label:
    #     model_path += '_without'
    # load_param_into_net(model, load_checkpoint(f'{model_path}.ckpt'))
    model.load_state_dict(torch.load(f'{model_path}'))  # 假设使用 .pth 格式
    model.eval()
    return model