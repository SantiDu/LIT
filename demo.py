import json
import os
import sys
from contrastive import run_contrastive_learning
from localization import IntervLocalizer
import subprocess
import time

from parser import make_evaluation_parser

import jax.numpy as jnp
import numpy as np


def metric(true_tgt, rec_tgt, error=True):
    # true_tgt: nd array; rec_tgt: list or nd array
    if true_tgt.shape[0] == 0:
        return [1 / max(len(rec_tgt), 1)] * 3, [1 / max(len(rec_tgt), 1) ** 2] * 3
    
    common = np.intersect1d(true_tgt, rec_tgt).shape[0]
    recall = common / max(true_tgt.shape[0], 1)
    precision = common / max(len(rec_tgt), 1)
    f1_score = 2 * precision * recall / max(precision + recall, 1)
    return np.array([precision, recall, f1_score])


def linear(num_vars=3, num_env=16, G=None):
    parser = make_evaluation_parser()
    kwargs = parser.parse_args()

    kwargs.num_vars = num_vars
    kwargs.num_vars_intervened = (num_vars + 1) // 2
    kwargs.random_seed = 10
    kwargs.callback = False
    kwargs.num_layer_mixing = 1
    kwargs.num_layers = 2
    kwargs.max_steps = int(1e4)
    kwargs.indep_test = 'corr'
    kwargs.num_env = num_env
    kwargs.lrange_min_interv = 1
    kwargs.lrange_max_interv = 9
    kwargs.num_obs = 5000

    alpha = 0.2

    ind_collection, observ, noise, label, target, _, _ = run_contrastive_learning(kwargs)

    ind = ind_collection[1]
    locate_model = IntervLocalizer(w=ind, observation=observ, rec_noise=noise[0], label=label)
    locate_model.normalize(prune=True, alpha=alpha) # 0.15

    interv_set, CI_lit, _ = locate_model.interv_nolatent(ci_test='partial_corr', permute_column=False)

    cal = metric(target, interv_set)

    print('Linear, no latent:')
    print(f'  True target: {target}')
    print(f'  Recovered target: {interv_set}')


def linear_latent(num_vars=3, num_env=16):
    parser = make_evaluation_parser()
    kwargs = parser.parse_args()

    kwargs.num_vars = num_vars
    kwargs.num_vars_intervened = int(num_vars * 0.5)
    kwargs.num_latent_vars = int(num_vars * 0.5)
    kwargs.latent_interv_perc = 0.8
    kwargs.random_seed = 10
    kwargs.callback = False
    kwargs.num_layer_mixing = 1
    kwargs.num_layers = 2
    kwargs.max_steps = int(1e4)
    kwargs.indep_test = 'corr'
    kwargs.num_env = num_env
    kwargs.lrange_min_interv = 1
    kwargs.lrange_max_interv = 9
    kwargs.num_obs = 5000

    alpha = 0.05

    ind_collection, observ, noise, label, target, _, _ = run_contrastive_learning(kwargs)

    ind = ind_collection[1]
    locate_model = IntervLocalizer(w=ind, observation=observ, rec_noise=noise[0], label=label)
    locate_model.normalize(prune=True, alpha=alpha) # 0.15

    interv_set, CI_lit, _ = locate_model.interv_latent(ci_test='partial_corr', permute_column=False, threshold=0.05)

    cal = metric(target, interv_set)

    print('Linear, latent:')
    print(f'  True target: {target}')
    print(f'  Recovered target: {interv_set}')


def nonlinear(num_vars=3, num_env=16):
    parser = make_evaluation_parser()
    kwargs = parser.parse_args()

    kwargs.callback = False
    kwargs.num_vars = num_vars
    kwargs.num_layer_mixing = 2
    kwargs.random_seed = 10
    kwargs.num_layers = 4
    kwargs.num_env = num_env
    kwargs.num_vars_intervened = (num_vars + 1) // 2
    kwargs.max_steps = int(4e4)
    kwargs.indep_test = 'corr'
    kwargs.sourcetype = 'laplace'
    kwargs.lrange_min_interv = 1
    kwargs.lrange_max_interv = 9
    kwargs.num_obs = 3000

    alpha = 0.2

    ind_collection, observ, noise, label, target, _, _ = run_contrastive_learning(kwargs)

    ind = ind_collection[1]
    locate_model = IntervLocalizer(w=ind, observation=observ, rec_noise=noise[0], label=label)
    locate_model.normalize(prune=True, alpha=alpha) # 0.15

    interv_set, CI_lit, _ = locate_model.interv_nolatent(ci_test='partial_corr', permute_column=False)

    cal = metric(target, interv_set)

    print('Noninear, latent:')
    print(f'  True target: {target}')
    print(f'  Recovered target: {interv_set}')

if __name__ == '__main__':

    linear(num_vars=5, num_env=16)
    linear_latent(num_vars=9, num_env=16)
    nonlinear(num_vars=5, num_env=16)
        
