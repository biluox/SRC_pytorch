import random
from argparse import Namespace

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from KTScripts.PredictModel import PredictModel, PredictRetrieval


def set_random_seed(seed):
    if seed < 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 如果使用 GPU，还需设置 CUDA 随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置种子
    print(f"random seed set to be {seed}")


def load_model(args: (Namespace, dict)):
    if isinstance(args, dict):
        args = Namespace(**args)
    if args.model in ('DKT', 'Transformer', 'GRU4Rec'):
        return PredictModel(
            feat_nums=args.feat_nums,
            embed_size=args.embed_size,
            hidden_size=args.hidden_size,
            pre_hidden_sizes=args.pre_hidden_sizes,
            dropout=args.dropout,
            output_size=args.output_size,
            with_label=not args.without_label,
            model_name=args.model)

    if args.model == 'CoKT':
        return PredictRetrieval(
            feat_nums=args.feat_nums,
            input_size=args.embed_size,
            hidden_size=args.hidden_size,
            pre_hidden_sizes=args.pre_hidden_sizes,
            dropout=args.dropout,
            with_label=not args.without_label,
            model_name=args.model)
    raise NotImplementedError


def evaluate_utils(y_, y):
    if not isinstance(y_, np.ndarray):
        y_, y = y_.cpu().detach().numpy(), y.cpu().detach().numpy()
    acc = np.mean(np.equal(np.argmax(y_, -1) if len(y_.shape) > 1 else y_ > 0.5, y))
    auc = acc
    if not (np.equal(y, y[0])).all():
        if len(y_.shape) == 1:
            auc = roc_auc_score(y_true=y, y_score=y_)
    return acc, auc
