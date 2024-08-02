import numpy as np
import jax.numpy as jnp
from jax import random
import warnings
# import sys, os
# sys.path.append(os.getcwd())
from data.util import l2normalize
# from experiments.contrastive import indication_set_new


def sample_ER(num_vars, num_edges, random_seed=0):
    """Samples DAG"""

    # p = num_edges / ((num_vars * (num_vars - 1)) / 2)
    p = 1.5 / (num_vars - 1)
    # need 0 < p < 1, otherwise get NANs in log computation
    if p >= 1:
        warnings.warn("Parameter p of ER distribution cannot be 1. " +
                      "Setting it to p=(n-1)/n")
        p = (num_vars - 1) / num_vars

    np.random.seed(random_seed)
    mat = np.random.binomial(p=p, n=1,
                             size=(num_vars, num_vars)).astype(np.int32)

    # make DAG by zeroing above diagonal; k=-1 indicates that diagonal is zero too
    dag = np.tril(mat, k=-1)

    # randomly permute
    # P = np.random.permutation(np.eye(num_vars, dtype=np.int32))
    # dag_perm = P.T @ dag @ P
    order = np.random.permutation(num_vars)
    dag = dag[order][:, order]
    return dag, order


def generate_artificial_data(num_vars,
                             interv_targets,
                             num_vars_intervened,
                             num_env,
                             num_obs,
                             G=None,
                             num_layer=1,
                             activation='tanh',
                             num_edges=2,
                             num_segmentdata_test=None,
                             sourcetype='gaussian',
                             cond_thresh_ratio=0.25,
                             lrange_min=0,
                             lrange_max=3,
                             lrange_min_interv=0,
                             lrange_max_interv=3,
                             random_seed=0):
    noise, label, L, selected_var = gen_source_noise(
        num_vars,
        interv_targets,
        num_vars_intervened,
        num_env,
        num_obs,
        Lrange=np.array([lrange_min, lrange_max])
        if not (lrange_min is None or lrange_max is None) else None,
        # diff noise scale for interventions
        Lrange_interv=np.array([lrange_min_interv, lrange_max_interv]) if
        not (lrange_min_interv is None or lrange_max_interv is None) else None,
        sourcetype=sourcetype,
        random_seed=random_seed)
    sensor, G = apply_sem_to_noise(noise,
                                   G=G,
                                   num_layer=num_layer,
                                   activation=activation,
                                   cond_thresh_ratio=cond_thresh_ratio,
                                   num_edges=num_edges,
                                   random_seed=random_seed)

    return sensor, noise, label, G, selected_var


def gen_source_noise(num_vars,
                     interv_targets, 
                     num_vars_intervened,
                     num_env,
                     num_obs,
                     L=None,
                     Ltype='uniform',
                     Lrange=None,
                     Lrange_interv=None,
                     sourcetype='gaussian',
                     random_seed=0):
    """Generate source noise for SEM.
    Args:
        num_vars: number of variables
        num_env: number of environments (different noises)
        num_obs: number of data-points in each environment
        L: (option) modulation parameter. If not given, newly generate based on Ltype
        Ltype: (option) generation method of modulation parameter
        Lrange: (option) range of modulation parameter
        sourcetype: (option) Distribution type of source signal
        random_seed: (option) random seed
    Returns:
        source: source signals. 2D ndarray [num_comp, num_data]
        label: labels. 1D ndarray [num_data]
        L: modulation parameter of each component/segment. 2D ndarray [num_comp, num_segment]
    """
    if Lrange is None:
        Lrange = np.array([0, 1])
    if Lrange_interv is None:
        Lrange = np.array([0, 1])
    # print("Generating noise...")

    # Initialize random generator
    np.random.seed(random_seed)

    # Change random_seed based on num_vars and envs
    for _ in range(num_vars * num_env):
        np.random.rand()

    # Generate modulation parameter
    if L is None:
        if Ltype == "uniform":
            L_obs = np.random.uniform(
                size=[num_vars],
                low=Lrange[0],
                high=Lrange[1],
            )
            L_interv = np.random.uniform(
                size=[num_env, num_vars],
                low=Lrange_interv[0],
                high=(Lrange_interv[0] + Lrange_interv[1]) / 2,
            )
            L_interv[num_env//2:] += (Lrange_interv[1] - Lrange_interv[0]) / 2
            # L_interv = np.random.uniform(
            #     size=[num_env, num_vars],
            #     low=Lrange_interv[0],
            #     high=Lrange_interv[1],
            # )

            # print(L_interv)

            ## for now, change exactly one noise variable per env
            ## and the first will k envs will change exactly k diff vars
            #eye = np.eye(num_vars, num_vars)
            #permuted_eye = np.random.permutation(eye)
            #permuted_eye = permuted_eye[:min([num_env - 1, num_vars])]
            #mask = np.concatenate([np.zeros((1, num_vars)), permuted_eye])
            #if num_env > num_vars + 1:
            #    addit_targets = np.random.choice(num_vars,
            #                                     size=num_env - num_vars - 1)
            #    add_mask = np.eye(num_vars)[addit_targets]
            #    mask = np.concatenate([mask, add_mask])

            ## We select a subset of k_int variables to changes their variances
            ## across the environments
            selected_var = np.array(interv_targets) if len(interv_targets) > 0 else np.random.choice(
                np.arange(num_vars), size=num_vars_intervened, replace=False)
            mask = np.zeros((num_env, num_vars))
            mask[:,selected_var] = 1

            # print(mask)

            # L: [num_env, num_vars]
            L = mask * L_interv + (1 - mask) * L_obs[None]

            targets = np.indices((num_env, num_vars))[1][mask == 1]
            # print("Changing noise of variables {}".format(targets))
            # print(L)
        else:
            raise ValueError

    # Generate source signal ----------------------------------
    num_data = num_env * num_obs

    if sourcetype == 'laplace':
        # laplace with scale to variance 1
        source_sc = np.random.laplace(0,
                                      1 / np.sqrt(2),
                                      size=(num_env, num_obs, num_vars))
    elif sourcetype == 'gaussian':
        # gaussian with scale to variance 1
        source_sc = np.random.normal(size=(num_env, num_obs, num_vars))
    elif sourcetype == 'uniform':
        source_sc = np.random.uniform(low=-1,
                                        high=1,
                                        size=(num_env, num_obs, num_vars))
    else:
        raise ValueError

    # [num_env, num_obs, num_vars]
    source = source_sc * L[:, None]
    source = source.reshape(num_data, num_vars)
    label = np.repeat(np.arange(num_env, dtype=int), num_obs)
    return source, label, L, selected_var


def apply_sem_to_noise(noise,
                       G=None,
                       num_layer=1,
                       activation='tanh',
                       num_edges=2,
                       iter4condthresh=10000,
                       cond_thresh_ratio=0.25,
                       layer_name_base='ip',
                       save_layer_data=False,
                       hidden_nodes=5,
                       Arange=None,
                       negative_slope=0.2,
                       random_seed=0,
                       mix='mlp' # 'post_nonlinear'
                       ):
    """Generate SEM with additive noise and apply it to source noise.
    Args:
        noise: source signals. 2D ndarray [num_data, num_vars]
        num_layer: number of layers
        num_segment: (option) number of segments (only used to modulate random_seed)
        iter4condthresh: (option) number of random iteration to decide the threshold of condition number of mixing matrices
        cond_thresh_ratio: (option) percentile of condition number to decide its threshold
        layer_name_base: (option) layer name
        save_layer_data: (option) if true, save activities of all layers
        Arange: (option) range of value of mixing matrices
        nonlinear_type: (option) type of nonlinearity
        negative_slope: (option) parameter of leaky-ReLU
        random_seed: (option) random seed
    Returns:
        mixedsig: sensor signals. 2D ndarray [num_comp, num_data]
        mixlayer: parameters of mixing layers
    """
    if Arange is None:
        # Arange = [-1, 1]
        Arange = [0.5, 1]

    np.random.seed(random_seed)
    # Change random_seed based on num_segment
    for i in range(num_layer):
        np.random.rand()

    num_vars = noise.shape[1]

    if G is None:
        # get graph G in lower triangular form, i.e. the causal ordering
        G, order = sample_ER(num_vars=num_vars, num_edges=num_edges * num_vars / 2, random_seed=random_seed)

    # Determine condThresh ------------------------------------
    condList = np.zeros([iter4condthresh])
    for i in range(iter4condthresh):
        A = np.random.uniform(low=Arange[0],
                              high=Arange[1],
                              size=[num_vars, num_vars])
        A = l2normalize(A, axis=0)
        condList[i] = np.linalg.cond(A)

    condList.sort()  # Ascending order
    condThresh = condList[int(iter4condthresh * cond_thresh_ratio)]
    # print("    cond thresh: {0:f}".format(condThresh))

    # Generate mixing matrix ------------------------------
    condA = condThresh + 1
    if num_layer == 1:
        # linear SEM
        while condA > condThresh:
            A = np.random.uniform(low=Arange[0],
                                  high=Arange[1],
                                  size=[num_vars, num_vars])
            A = l2normalize(A)  # Normalize (column)
            condA = np.linalg.cond(A)
            # print("    L{0:d}: cond={1:f}".format(0, condA))

            # W = G * A + np.diag(np.ones(num_vars))
            W = np.linalg.inv(np.eye(num_vars) - G * A)
            sensor = np.dot(noise, W.T)
            return sensor, G
    else:
        # sensor = noise.copy()
        # prev_layer = num_vars
        # for layer in range(num_layer):
        #     size = [num_vars, prev_layer, hidden_nodes
        #             ] if layer != num_layer - 1 else [num_vars, prev_layer, 1]
        #     # Generate mixing matrix ------------------------------
        #     condA = np.array([condThresh + 1])
        #     while np.all(condA > condThresh):
        #         A = np.random.uniform(low=Arange[0], high=Arange[1], size=size)
        #         A = l2normalize(A)  # Normalize (column)
        #         condA = np.linalg.cond(A)
        #         A = np.random.normal(size=size)

        #     if layer == 0:
        #         # first layer: mask out parents, no nonlinear activation
        #         W = G[..., None] * A
        #         # [num_obs, num_vars, hidden_layer]
        #         sensor = np.einsum('nv,vhj->nvj', sensor, W)
        #         # bias
        #         sensor = sensor + np.random.uniform(
        #             low=Arange[0], high=Arange[1], size=sensor.shape)
        #     else:
        #         if activation == 'leakyrelu':
        #             sensor[sensor < 0] = negative_slope * sensor[sensor < 0]
        #         elif activation == 'leakytanh':
        #             sensor = np.tanh(sensor) + negative_slope * sensor
        #         elif activation == 'tanh':
        #             sensor = np.tanh(sensor)
        #         elif activation == 'relu':
        #             sensor[sensor < 0] = 0
        #         elif activation == 'sigmoid':
        #             sensor = 1. / (1. + np.exp(-sensor))
        #         else:
        #             raise ValueError(
        #                 '{} nonlinear activation not implemented for mixing'.
        #                 format(activation))
        #         # [num_obs, num_vars, hidden_layer]
        #         sensor = np.einsum('nvh,vhj->nvj', sensor, A)
        #         # bias
        #         sensor = sensor + np.random.uniform(
        #             low=Arange[0], high=Arange[1], size=sensor.shape)
        #     prev_layer = hidden_nodes
        # # sensor: [num_obs, num_vars, 1] -> drop last dim
        # sensor = sensor.squeeze(-1)
        # # mask out source nodes, they're just noise
        # return sensor * (G.sum(1) >= 1) + noise, G

        #####################################################
        if mix == 'post_nonlinear':
            for layer in range(num_layer):
                if layer == 0:
                    # linear SEM
                    while condA > condThresh:
                        A = np.random.uniform(low=Arange[0],
                                      high=Arange[1],
                                      size=[num_vars, num_vars])
                        A = l2normalize(A)  # Normalize (column)
                        condA = np.linalg.cond(A)
                        # print("    L{0:d}: cond={1:f}".format(0, condA))

                        # W = G * A + np.diag(np.ones(num_vars))
                        W = np.linalg.inv(np.eye(num_vars) - G * A)
                        # print(A)
                        # print(W)
                        sensor = np.dot(noise, W.T)
                else:
                    coef = np.random.uniform(low=Arange[0],
                                            high=Arange[1],
                                            size=[2, num_vars])
                    # pointwise activation: sigma(wx+b)
                    # print('   ', coef)
                    sensor = sensor * coef[0] + coef[1]
                    if activation == 'leakyrelu':
                        sensor[sensor < 0] = negative_slope * sensor[sensor < 0]
                    elif activation == 'leakytanh':
                        sensor = np.tanh(sensor) + negative_slope * sensor
                    elif activation == 'tanh':
                        sensor = np.tanh(sensor)
                    elif activation == 'relu':
                        sensor[sensor < 0] = 0
                    elif activation == 'sigmoid':
                        sensor = 1. / (1. + np.exp(-sensor))
                    else:
                        raise ValueError(
                            '{} nonlinear activation not implemented for mixing'.
                            format(activation))
        else:
            def apply_nn(data):
                num_var = data.shape[1]
                coef = np.random.uniform(low=Arange[0],
                                        high=Arange[1],
                                        size=[num_var, num_var])
                intersect = np.random.uniform(low=Arange[0],
                                        high=Arange[1],
                                        size=[1, num_var])
                data = data @ coef + intersect
                data[data < 0] = negative_slope * data[data < 0]
                coef = np.random.uniform(low=Arange[0],
                                        high=Arange[1],
                                        size=[num_var, 1])
                intersect = np.random.uniform(low=Arange[0],
                                        high=Arange[1],
                                        size=[1, 1])
                data = data @ coef + intersect
                data[data < 0] = negative_slope * data[data < 0]
                return data[:,0]
            sensor = np.zeros_like(noise)
            rev_order = np.arange(len(order))[np.argsort(order)]
            for variable in rev_order:
                parent = np.where(G[variable] != 0)[0]
                data_cause = noise[:, variable][:, np.newaxis]
                if len(parent) != 0:
                    data_cause = np.hstack((sensor[:, parent], data_cause))
                sensor[:, variable] = apply_nn(data_cause)

        return sensor, G



if __name__ == '__main__':
    sensor, noise, label, G, targets = generate_artificial_data(num_vars=3,
                                                                interv_targets=[],
                                                                num_vars_intervened=3,
                                                                num_env=16,
                                                                num_obs=1000,
                                                                num_layer=2,
                                                                random_seed=20,
                                                                )
    # from conditional_independence import hsic_test as hsic
    # mat = np.zeros((3,3))
    # for i in range(3):
    #     for j in range(3):
    #         x = sensor[:, i]
    #         y = noise[:, j]
    #         mat[i, j] = (1 - hsic(np.vstack((x, y)).T, 0, 1)['p_value'])
    print(G)
    # print(indication_set_new(label, sensor, noise, 16))
    
