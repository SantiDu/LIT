from enum import unique
from itertools import accumulate
import os
import copy
import time
from pprint import pprint

# Set GPU device: current installation is JAX CPU version only, hence the following line has no effect
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # set gpu device
os.environ['JAX_DEBUG_NANS'] = 'True'  # debugs NaNs

import jax.numpy as jnp
import numpy as np
import jax
from sklearn.model_selection import train_test_split
import sklearn
import wandb
import haiku as hk

# ICA
from sklearn.decomposition import FastICA
from eval.metrics import SolveHungarian, get_correlation_metrics, get_env_change_metrics, find_source_change_metric

# conditional independence test (KCI Test)
from causallearn.utils.cit import CIT
from conditional_independence import hsic_test as hsic
from scipy.stats import spearmanr as spearman
from scipy.stats import kendalltau as kendall
from dcor import distance_correlation as dist_corr

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
import causaldag

# #### Data generation
from data.sem import generate_artificial_data as generate_sem
from data.sem_latent import generate_artificial_data as generate_sem_latent
from data.preprocessing import pca_whitening

# model
from models.tcl import TCL
# import breakpoint as bp


def indicator_set(label, sensor, feat_val, num_env, method='hsic'):
    corr = []
    num_vars = sensor.shape[1]
    num_vars_intervened = feat_val.shape[1]

    for var in range(num_vars):
        corr_ = []
        # x = sensor[:, var]
        for s in range(num_vars_intervened):
            pvalue = 0
            for e in range(num_env):
                x = sensor[label == e, var]
                y = feat_val[label == e, s]

                ############# kci test #######
                # kci_obj = CIT(np.vstack((x, y)).T, "kci") 
                # pvalue1 = kci_obj(0, 1)
                # print('  ', pvalue1)
                ############# hsic test #######
                if method == 'hsic':
                    pvalue = max(pvalue, hsic(np.vstack((x, y)).T, 0, 1)['p_value'])
                elif method == 'corr':
                    pvalue = max(pvalue, abs(np.corrcoef(y, x)[0][1]))
                elif method == 'dist_corr':
                    pvalue = max(pvalue, abs(dist_corr(y, x)))
                elif method == 'kendall':
                    pvalue = max(pvalue, kendall(x, y).pvalue)
                elif method == 'spearman':
                    pvalue = max(pvalue, spearman(x, y).pvalue)

            if method in ['corr', 'dist_corr']:
                corr_.append(pvalue)
            else:
                corr_.append(1 - pvalue)

        corr.append(corr_)
    return np.array(corr) 


def find_K(G, selected_var_all, ind_true=None):
    num_obs = G.shape[0]
    num_source = G.shape[1]
    num_lat = num_source - num_obs
    if ind_true is None:
        Ainv = np.eye(num_obs) - G[:, num_lat:]
        W = np.hstack((G[:, :num_lat], np.eye(num_obs)))
        ind_true = np.linalg.solve(Ainv, W)[:, selected_var_all]

    # Augmented graph
    T_L = [_ for _ in range(num_lat) if _ in selected_var_all]
    L = [_ for _ in range(num_lat) if _ not in T_L]
    T_O = [ _ - num_lat for _ in selected_var_all if _ not in T_L]
    num_TO = len(T_O)

    Aug_A = np.zeros((num_source + num_TO, num_source + num_TO))
    Aug_A[num_lat:(num_lat+num_obs), :num_source] = G.copy()
    for i, x in enumerate(T_O):
        Aug_A[num_lat + x, num_source + i] = 1
    
    # Auxilliary graph
    D = causaldag.DAG.from_amat(Aug_A.T)
    M = D.marginal_mag(latent_nodes=set(L))
    # N_o
    K = []
    for var in range(num_TO):
        xi = list(D.children_of(var + num_source))[0]
        K.append(xi - num_lat)
        xi_ind = np.where(ind_true[xi - num_lat])[0]
        for xj in M.children_of(var + num_source):
            if xj != xi and np.array_equal(np.where(ind_true[xj - num_lat])[0], xi_ind):
                K.append(xj - num_lat)
    # N_l
    for var in T_L:
        ch = np.array(list(D.children_of(var))) - num_lat  # children set
        # print(ch)
        count = np.count_nonzero(ind_true[ch,:], axis=1)
        small_ch = ch[count == count.min()]
        # print(small_ch)
        xi_ind = np.where(ind_true[small_ch[0]])[0]
        ind_check = ind_true[small_ch][:, xi_ind]
        # print(ind_check)
        if not ind_check.all(axis=None):
            continue
        K += list(small_ch)
        # print(K)
        xj_list = M.children_of(var) -  D.children_of(var)
        for xj in xj_list:
            if np.array_equal(np.where(ind_true[xj - num_lat])[0], xi_ind):
                K.append(xj - num_lat)

    return K

def train_tcl(tcl: TCL,
              X: jnp.ndarray,
              y: jnp.array,
              X_test: jnp.ndarray,
              y_test: jnp.ndarray,
              batch_size: int,
              max_steps: int,
              max_steps_pretrain: int = 0,
              config: dict = {},
              callback=None,
              callback_every: int = 100,
              random_seed: int = 42):
    """
        tcl: 
        X: 2D array [n_samples, n_features]
        y: 1D array [n_samples, ]
    """

    # wandb for logging
    if config.use_wandb:
        wandb.init(project=config.name, config=config)

    train_num_examples = X.shape[0]
    rng_seq = hk.PRNGSequence(random_seed)
    # Initialize model
    params, non_trainable_params, opt_state = tcl.init(
        next(rng_seq),
        jnp.zeros((1, X.shape[1])),
        pretrain=max_steps_pretrain > 0)

    # Train/eval loop.
    if tcl.verbose:
        print("Training..")
    X_ = X.copy()
    y_ = y.copy()

    num_steps_in_epoch = int(train_num_examples / batch_size)
    step_in_epoch = 0

    # one training step with loss, update and eval ---------------------
    def train_step(step,
                   params,
                   non_trainable_params,
                   opt_state,
                   ema_params=None,
                   ema_state=None,
                   pretrain=False):
        nonlocal step_in_epoch, X_, y_, rng_seq
        # Make shuffled batches each epoch -----------------------------
        if step % num_steps_in_epoch == 0:
            step_in_epoch = 0
            ind = jnp.arange(train_num_examples, dtype=int)
            ind = jax.random.permutation(next(rng_seq), ind)
            X_ = X[ind]
            y_ = y[ind]

        # Train/eval
        start_time = time.time()
        x_batch = X_[batch_size * step_in_epoch:batch_size *
                     (step_in_epoch + 1)]
        y_batch = y_[batch_size * step_in_epoch:batch_size *
                     (step_in_epoch + 1)]
        step_in_epoch = step_in_epoch + 1
        loss, params, opt_state = tcl.update(params, non_trainable_params,
                                             opt_state, x_batch, y_batch)

        full_params = hk.data_structures.merge(params, non_trainable_params)
        train_accuracy = tcl.accuracy(full_params, x_batch, y_batch)

        metrics = dict(train_loss=loss, train_acc=train_accuracy)
        if not pretrain:
            # update exponential moving average
            ema_params, ema_state = tcl.update_ema(params, ema_state)

        end_time = time.time()
        if step % callback_every == 0:
            tcl._params = full_params
            # test set
            metrics['test_loss'] = tcl.loss(params, non_trainable_params,
                                            X_test, y_test)
            metrics['test_acc'] = tcl.accuracy(full_params, X_test, y_test)
            if not pretrain:
                tcl._ema_params = ema_params
            if callback is not None:
                if pretrain and tcl.verbose:
                    print("Pretrain MLR:", end="\t")
                add_metrics = callback(step=step,
                                       tcl=tcl,
                                       params=full_params,
                                       opt_state=opt_state,
                                       end_time=end_time,
                                       start_time=start_time,
                                       pretrain=pretrain)
                metrics = {**metrics, **add_metrics}

        if config.use_wandb:
            wandb.log(metrics)
        if pretrain:
            return params, opt_state
        else:
            return params, opt_state, ema_params, ema_state

    # ------------------------------------------------------------------
    # pretraining the MLR as done in original TCL
    for step in range(max_steps_pretrain):
        params, opt_state = train_step(step,
                                       params,
                                       non_trainable_params,
                                       opt_state,
                                       pretrain=True)

    # full training ----------------------------------------------------
    # merge params for full training
    params = hk.data_structures.merge(params, non_trainable_params)
    opt_state = tcl.reinit_opt(params)

    # exponential moving average of parameters
    ema_state = tcl.init_ema(params)
    ema_params, ema_state = tcl.update_ema(params, ema_state)

    for step in range(max_steps):
        params, opt_state, ema_params, ema_state = train_step(step,
                                                              params, {},
                                                              opt_state,
                                                              ema_params,
                                                              ema_state,
                                                              pretrain=False)

    return tcl, params, ema_params


def run_contrastive_learning(kwargs, G=None):
    """
    Run one experiment
    """
    config = copy.deepcopy(kwargs)
    if config.perc_intervened != -1.:
        config.perc_intervened = min(max(config.perc_intervened, 0.), 1.)
        config.num_env = int(config.perc_intervened * config.num_vars) + 1
        print("Overriding num_env through perc_intervened: {}".format(
            config.num_env))
    # Generate sensor signal --------------------------------
    if config.num_latent_vars == 0:
        sensor, source, label, G, selected_var = generate_sem(
            num_vars=config.num_vars,
            interv_targets=config.interv_targets,
            num_vars_intervened=config.num_vars_intervened,
            num_env=config.num_env,
            num_obs=config.num_obs,
            activation='tanh',
            G=G,
            num_layer=config.num_layer_mixing,
            sourcetype=config.sourcetype,
            lrange_min=config.lrange_min,
            lrange_max=config.lrange_max,
            lrange_min_interv=config.lrange_min_interv,
            lrange_max_interv=config.lrange_max_interv,
            random_seed=config.random_seed)
    else:
        sensor, source, label, G, selected_var, selected_var_all = generate_sem_latent(
            num_vars=config.num_vars,
            num_latent_vars=config.num_latent_vars,
            interv_targets=config.interv_targets,
            num_vars_intervened=config.num_vars_intervened,
            num_env=config.num_env,
            num_obs=config.num_obs,
            activation='tanh',
            G=G,
            latent_interv_perc=config.latent_interv_perc,
            num_layer=config.num_layer_mixing,
            sourcetype=config.sourcetype,
            lrange_min=config.lrange_min,
            lrange_max=config.lrange_max,
            lrange_min_interv=config.lrange_min_interv,
            lrange_max_interv=config.lrange_max_interv,
            random_seed=config.random_seed)
    
    sensor_ = sensor  # observation
    ##### Training
    # MLP ---------------------------------------------------------
    list_hidden_nodes = [2 * config.num_vars] * (config.num_layers - 1) + [
        config.num_vars
    ]
    # print("hidden layers", list_hidden_nodes)
    n_epochs = int(config.max_steps / (sensor.shape[0]) * config.batch_size)
    # print("With {} steps will roughly do {} epochs".format(
        # config.max_steps, n_epochs))

    X_train, X_test, y_train, y_test = train_test_split(
        sensor, label, test_size=config.test_split_ratio)

    # sources to be compared with ICA
    if config.sourcetype == 'laplace':
        xseval = jnp.abs(source)
    elif config.sourcetype == 'gaussian':
        xseval = source**2

    ##############  'True' mixing matrix #####################
    if config.num_latent_vars == 0:
        ind_true = np.linalg.inv(np.eye(config.num_vars) - G)[:, selected_var]
    else:
        num_obs_vars = config.num_vars - config.num_latent_vars
        Ainv = np.eye(num_obs_vars) - G[:, config.num_latent_vars:]
        W = np.hstack((G[:, :config.num_latent_vars], np.eye(num_obs_vars)))
        ind_true = np.linalg.solve(Ainv, W)[:, selected_var_all]
    ###################################

    t_before = time.time()
    if config.num_layer_mixing == 1 and config.linear_ica:
        rec_ica = FastICA(n_components=config.num_vars_intervened, whiten='unit-variance', random_state=42)
        feat_val_ica = rec_ica.fit_transform(sensor_)
        rec_ica.fit(sensor_)
        ind_set_ica = rec_ica.mixing_
        ind_set_ica = ind_set_ica / np.maximum(np.linalg.norm(ind_set_ica, axis=0), 1e-6)
        ind_set_ica = ind_set_ica / np.sign(ind_set_ica[np.argmax(abs(ind_set_ica), axis=0), range(config.num_vars_intervened)])
        ind_collection = [ind_true, ind_set_ica]

        if config.num_latent_vars == 0:
            source_changed = source[:, selected_var]
        else:
            source_changed = source[:, selected_var_all]

        _, mat, ii = SolveHungarian(feat_val_ica, source_changed)

        feat_val = [feat_val_ica, source_changed, source_changed[:,ii[1]]]

    else:
        # Preprocessing -----------------------------------------------
        X_train, pca_parm = pca_whitening(X_train, num_comp=X_train.shape[1])
        X_test, _ = pca_whitening(X_test,
                                params=pca_parm,
                                num_comp=X_test.shape[1])
                                #   num_comp=config.num_vars)
        sensor, _ = pca_whitening(sensor,
                                params=pca_parm,
                                num_comp=sensor.shape[1])
                            #   num_comp=config.num_vars)

        tcl = TCL(hidden_units=list_hidden_nodes,
                n_classes=config.num_env,
                opt=config.opt,
                momentum=config.momentum,
                lr=config.initial_lr,
                moving_avg_decay=config.moving_average_decay,
                decay_steps=config.decay_steps,
                lambda_reg=config.lambda_reg,
                decay_factor=config.decay_factor,
                verbose=config.verbose)

        try:
            tcl, params, _ = train_tcl(
                tcl,
                X=X_train,
                y=y_train,
                X_test=X_test,
                y_test=y_test,
                batch_size=config.batch_size,
                max_steps=config.max_steps,
                max_steps_pretrain=config.max_steps_init,
                config=config,
                random_seed=config.model_seed,
                callback=None,
                )
        finally:
            params = tcl._params


        ##### Evaluate ---------------------------------------------------------------
        # accuracy
        hidden = tcl.get_hidden(params, sensor)
        # apply ICA to get estimated sources
        ica = FastICA(n_components=config.num_vars_intervened, whiten='unit-variance', random_state=42)
        feat_val = ica.fit_transform(hidden)

        #############   Indicator set     ##############################
        ind_set_corr = indicator_set(label, sensor_, feat_val, config.num_env, method='corr')
        ind_collection = [ind_true, ind_set_corr]

        if config.num_latent_vars == 0:
            source_changed = source[:, selected_var]
            # q_source_changed = xseval[:, selected_var]
        else:
            source_changed = source[:, selected_var_all]
            # q_source_changed = xseval[:, selected_var_all]

        # _, mat, ii = SolveHungarian(feat_val, source_changed)
        feat_val = [feat_val, source_changed]

    t_after = time.time() - t_before

    return ind_collection, sensor_, feat_val, label, selected_var, t_after, find_K(G, selected_var_all, ind_true) if config.num_latent_vars > 0 else None


if __name__ == '__main__':
    pass
