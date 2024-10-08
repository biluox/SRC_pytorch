from argparse import ArgumentParser, Namespace

def get_exp_configure(agent):
    config_dict = {
        'embed_size': 48,
        'hidden_size': 64,
        'output_size': 1,
        'dropout': 0.5,
        'decay_step': 1000,
        'min_lr': 1e-5,
        'l2_reg': 1e-4,
        'predict_hidden_sizes': [256, 64, 16]
    }
    if agent == 'MPC':
        config_dict.update({'hor': 20})
    if agent == 'DQN':
        config_dict['hidden_size'] = 128
    return config_dict


def get_options(parser: ArgumentParser, reset_args=None):
    if reset_args is None:
        reset_args = {}
    agent = ['MPC', 'DQN', 'SRC']
    model = ['DKT', 'CoKT']
    dataset = ['junyi', 'assist09', 'assist15']
    parser.add_argument('-a', '--agent', type=str, choices=agent, default='SRC')
    parser.add_argument('-train', '--isTrain', type=bool, default=True)
    parser.add_argument('-m', '--model', type=str, choices=model, default='DKT', help='Model used in MPC or KES')
    parser.add_argument('-d', '--dataset', type=str, choices=dataset, default='assist09')
    parser.add_argument('-w', '--worker', type=int, default=6)
    parser.add_argument('-b', '--batch_size', type=int, default=256)
    parser.add_argument('-p', '--path', type=int, default=2, choices=[0, 1, 2, 3])
    parser.add_argument('--data_dir', type=str, default='./datasets')
    parser.add_argument('--save_dir', type=str, default='./SavedModels')
    parser.add_argument('--visual_dir', type=str, default='./VisualResults')
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--load_model', action='store_true', default=False)
    parser.add_argument('--withKT', action='store_true', default=True, help='Whether to use KT as a secondary task')
    parser.add_argument('--binary', action='store_true', default=False, help='Whether the reward is binary')

    parser.add_argument('-c', '--cuda', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--postfix', type=str, default='')
    parser.add_argument('--rand_seed', type=int, default=-1)
    parser.set_defaults(**reset_args)
    args = parser.parse_args()
    # Get experiment configuration
    exp_configure = get_exp_configure(args.agent)
    args = Namespace(**vars(args), **exp_configure)

    args.exp_name = '_'.join([args.agent, args.model, args.dataset])
    if args.postfix != '':
        args.exp_name += '_' + args.postfix

    return args