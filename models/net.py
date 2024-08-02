import jax
import jax.numpy as jnp
import haiku as hk


def mlp(hidden_units,
        initializer='None',
        act='lipswish',
        name=None) -> hk.Sequential:
    """
    Returns an haiku MLP with relu nonlinearlties and a number
    of hidden units specified by an array
    :param hidden_units: Array containing number of hidden units of each layer
    :param initializer: if 'None' weights and biases of last layer are
    initialized as zero; else, they are initialized with initializer, from hk.initializers
    :param act: Activation function, can be relu, elu or lipswish
    :param name: Name of hidden layer
    :return: MLP as hk.Sequential
    """
    w_init = None
    if act == 'leakyrelu':
        act_fn = jax.nn.leaky_relu
        # He normal
        w_init = hk.initializers.VarianceScaling(2.0, 'fan_in', 'uniform')
    elif act == 'relu':
        act_fn = jax.nn.relu
        # He normal
        w_init = hk.initializers.VarianceScaling(2.0, 'fan_in', 'uniform')
    elif act == 'lipswish' or act == 'maxout':
        act_fn = None
    elif act == 'elu':
        act_fn = jax.nn.elu
    elif act == 'sigmoid':
        act_fn = jax.nn.sigmoid
        # Xavier/Glorot normal
        w_init = hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform')
    elif act == 'tanh':
        act_fn = jax.nn.tanh
        # Xavier/Glorot normal
        w_init = hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform')
    else:
        raise NotImplementedError('The activation function ' + act +
                                  ' is not implemented.')
    layers = []
    if name is None:
        prefix = ''
    else:
        prefix = name + '_'
    for i in range(len(hidden_units) - 1):
        if act == 'lipswish':
            act_fn = LipSwish(name=prefix + 'lipswish_' + str(i))
        elif act == 'maxout':
            act_fn = MaxOut(name=prefix + 'maxout_' + str(i))
        layer_name = prefix + 'linear_' + str(i)
        layers += [
            hk.Linear(hidden_units[i], name=layer_name, w_init=w_init), act_fn
        ]
    layer_name = prefix + 'linear_' + str(len(hidden_units) - 1)
    if initializer is None or initializer == 'None':
        layers += [hk.Linear(hidden_units[-1], name=layer_name)]
    else:
        layers += [
            hk.Linear(hidden_units[-1],
                      w_init=initializer,
                      b_init=initializer,
                      name=layer_name)
        ]
    return hk.Sequential(layers,
                         name='mlp' + (f'_{name}' if name is not None else ''))


class LipSwish(hk.Module):

    def __init__(self, name=None):
        super().__init__(name=name)

    def __call__(self, x):
        beta = hk.get_parameter('beta',
                                shape=[1],
                                dtype=x.dtype,
                                init=jnp.zeros)
        return jax.nn.swish(jax.nn.softplus(beta + 0.5) * x) / 1.1


class MaxOut(hk.Module):
    """
    MaxOut Activation function
    For an input of shape [..., d]
    with d % k == 0, split the tensor to shape
    [..., d // k, k] and compute the maximum across the last
    dimension 

    Per default, k == 2
    """

    def __init__(self, name=None, k=2):
        super().__init__(name=name)
        self.k = k

    def __call__(self, x):
        ch = x.shape[-1]
        assert ch % self.k == 0
        x = x.reshape((*x.shape[:-1], ch // self.k, self.k))
        return x.max(-1)