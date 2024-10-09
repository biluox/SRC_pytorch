import torch

from Scripts.Agent.SRC import SRC
from Scripts.Agent.MPC import MPC
from Scripts.Agent.utils import generate_path


def load_agent(args):
    if args.agent == 'SRC':
        return SRC(
            skill_num=args.skill_num,
            input_size=args.embed_size,
            weight_size=args.hidden_size,
            hidden_size=args.hidden_size,
            dropout=args.dropout,
            with_kt=args.withKT
        )

    if args.agent == 'MPC':
        return MPC(
            skill_num=args.skill_num,
            input_size=args.embed_size,
            hidden_size=args.hidden_size,
            pre_hidden_sizes=args.predict_hidden_sizes,
            dropout=args.dropout,
            hor=args.hor
        )
    raise NotImplementedError


def get_data(batch_size, skill_num, target_num, initial_len, path_type, n):
    targets = torch.randint(0, skill_num, (batch_size, target_num), dtype=torch.int32)
    initial_logs = torch.randint(0, skill_num, (batch_size, initial_len), dtype=torch.int32)
    paths = torch.tensor(generate_path(batch_size, skill_num, path_type, n))
    return targets, initial_logs, paths