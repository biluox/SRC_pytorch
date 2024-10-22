from argparse import ArgumentParser, Namespace

def get_exp_configure(model):
    configure_list = {
        'embed_size': 64,
        'hidden_size': 64,
        'output_size': 1,
        'batch_size': 1024,
        'dropout': 0.5,
        'decay_step': 100,
        'l2_reg': 1e-4,
        'pre_hidden_sizes': [256, 64, 8],
        'retrieval': False,
        'forRec': False
    }
    if model == 'CoKT':
        configure_list.update({'batch_size': 8, 'retrieval': True, 'embed_size': 48,
                               'hidden_size': 64, 'pre_hidden_sizes': [128, 64, 16]})
    if model == 'GRU4Rec':
        configure_list.update({'forRec': True, 'without_label': True})
    return configure_list


def get_options(parser: ArgumentParser, reset_args=None):
    if reset_args is None:
        reset_args = {}
    model = ['DKT', 'Transformer', 'CoKT', 'GRU4Rec']
    dataset = ['junyi', 'assist09', 'assist15']
    parser.add_argument('-m', '--model', type=str, choices=model, default='DKT', help="Model to use")
    parser.add_argument('-d', '--dataset', type=str, choices=dataset, default='assist09', help="Dataset to use")
    parser.add_argument('--data_dir', type=str, default='./datasets')
    parser.add_argument('--save_dir', type=str, default='./SavedModels')
    parser.add_argument('--load_model',default=False, action='store_true')
    parser.add_argument('--without_label', action='store_true')
    parser.add_argument('-c', '--cuda', type=int, default=0)
    parser.add_argument('-e', "--num_epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('--valid_step', type=int, default=1000)
    parser.add_argument("--postfix", type=str, default='', help="a string appended to the file name of the saved model")
    parser.add_argument("--rand_seed", type=int, default=-1, help="random seed for torch and numpy")
    parser.set_defaults(**reset_args)
    args = parser.parse_args().__dict__
    # Get experiment configuration
    exp_configure = get_exp_configure(args['model'])
    args.update(exp_configure)
    args = Namespace(**args)
    args.exp_name = '_'.join([args.model, args.dataset])
    if args.without_label:
        args.exp_name += '_without'
    if args.postfix != '':
        args.exp_name += f'_{args.postfix}'

    return args
