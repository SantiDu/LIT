import jax
import haiku as hk
import jax.numpy as jnp


# Let's see how many parameters in our network and their shapes
def get_num_params(params: hk.Params, verbose=False):
    num_params = 0
    param_shapes = []
    for p in jax.tree_leaves(params):
        print('param shape:', p.shape)
        param_shapes.append(p.shape)
        num_params = num_params + jnp.prod(jnp.array(p.shape))
    print("Total number of parameters: {}".format(num_params))
    return num_params, param_shapes
