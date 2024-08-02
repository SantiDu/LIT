import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def make_evaluation_parser():
    """
    Returns argparse parser to control evaluation from command line
    """

    parser = argparse.ArgumentParser()
    # setup
    parser.add_argument("--name", type=str, default='sem-prototyping', help="high level name")
    parser.add_argument("--exp_result_folder", type=str, default=None, help="where to store dump to")
    parser.add_argument("--random_seed", type=int, default=2, help="random seed")
    parser.add_argument("--model_seed", type=int, default=0, help="random seed for model/training procedure")
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--use_wandb", action='store_true')
    parser.add_argument("--log_wandb_images", action='store_true')
    parser.add_argument("--callback", type=int, default=1, help="Whether or not to use a callback function during training")
    parser.add_argument("--callback_every", type=int, default=1000, help="callback after every `n` steps")
    parser.add_argument('--method', type=str, default='contrastive_ica', choices=["contrastive_ica"], help="method")

    # data
    parser.add_argument("--num_vars", type=int, default=3, help="number of variables")
    parser.add_argument("--num_latent_vars", type=int, default=0, help="number of latent variables")
    parser.add_argument("--interv_targets", type=int, nargs='*', default=[], help="intervention targets. if empty then randomly select interv targets.")
    parser.add_argument("--num_vars_intervened", type=int, default=2, help="number of intervened variables")
    parser.add_argument("--latent_interv_perc", type=float, default=0., help="percantage of latent variables being intervened on")
    parser.add_argument("--perc_intervened", type=float, default=-1., help="perc of vars to change noise of, overrides num_env. currently, one env per changed noise is created (i.e. not multiple targets)")
    parser.add_argument("--num_env", type=int, default=16, help="number of environments")
    parser.add_argument("--num_obs", type=int, default=2000, help="number of samples per env")
    parser.add_argument("--num_layer_mixing", type=int, default=1, help="number of layers for mixing")
    parser.add_argument("--lrange_min", type=float, default=1., help="bounds where to uniformly sample noise scale from")
    parser.add_argument("--lrange_max", type=float, default=3., help="bounds where to uniformly sample noise scale from")
    parser.add_argument("--lrange_min_interv", type=float, default=3., help="bounds where to uniformly sample noise scale from")
    parser.add_argument("--lrange_max_interv", type=float, default=5., help="bounds where to uniformly sample noise scale from")
    parser.add_argument("--sourcetype", type=str, default='gaussian', choices=['gaussian', 'laplace'], help="distribution of sources/noise")
    parser.add_argument("--test_split_ratio", type=float, default=0.1, help="ratio of heldout test set size")
    parser.add_argument("--indep_test", type=str, default='hsic', choices=['hsic', 'corr'], help="method for independence test")
     

    # MLP
    parser.add_argument("--linear_ica", type=bool, default=True, help="whether to use linear ica")
    parser.add_argument("--num_layers", type=int, default=2, help="number of layers of learner")
    parser.add_argument("--activation", type=str, default='maxout', choices=['maxout', 'leakyrelu'], help="nonlinear activation")
    parser.add_argument("--lambda_reg", type=float, default=1e-4, help="regularization factor for weights")
    parser.add_argument("--opt", type=str, default='sgd', choices=['sgd', 'adam'], help="optimizer")
    parser.add_argument("--batch_size", type=int, default=512, help="size of batch")
    parser.add_argument("--initial_lr", type=float, default=1e-2, help="initial learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum. only used for sgd")
    parser.add_argument("--max_steps", type=int, default=int(4e4), help="max number of gradient steps / batch evals")
    parser.add_argument("--decay_steps", type=int, default=int(5e4), help="number of steps for exponential lr decay")
    parser.add_argument("--decay_factor", type=float, default=0.1, help="decay factor for exp lr decay")
    parser.add_argument("--max_steps_init", type=int, default=int(5e3), help="last layer init: max number of gradient steps / batch evals")
    parser.add_argument("--decay_steps_init", type=int, default=int(5e3), help="last layer init: number of steps for exponential lr decay")
    parser.add_argument("--moving_average_decay", type=float, default=0.999, help="decay factor for exp decay of moving avg of weights")
    
    return parser