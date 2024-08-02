"""
Preprocessing data (whitening, PCA)
Parts of code taken from TCL (https://github.com/hmorioka/TCL)
"""

import jax.numpy as jnp
from jax import jit
import functools


# ============================================================
# ============================================================
# @functools.partial(jit, static_argnums=(1, 2))
def pca_whitening(x,
                  num_comp=None,
                  params=None,
                  zerotolerance=1e-7,
                  verbose=False):
    """Apply PCA whitening to data.
    Args:
        x: data. 2D ndarray [num_data, num_comp]
        num_comp: number of components (optional)
        params: (option) dictionary of PCA parameters {'mean':?, 'W':?, 'A':?}. If given, apply this to the data
        zerotolerance: (option)
    Returns:
        x: whitened data
        parms: parameters of PCA
            mean: subtracted mean
            W: whitening matrix
            A: mixing matrix
    """
    if verbose:
        print("PCA...")
    x = x.T

    # Dimension
    if num_comp is None:
        num_comp = x.shape[0]
    if verbose:
        print("    num_comp={0:d}".format(num_comp))

    # From learned parameters --------------------------------
    if params is not None:
        # Use previously-trained model
        if verbose:
            print("    use learned value")
        data_pca = x - params['mean']
        x = jnp.dot(params['W'], data_pca)

    # Learn from data ----------------------------------------
    else:
        # Zero mean
        xmean = jnp.mean(x, 1).reshape([-1, 1])
        x = x - xmean

        # Eigenvalue decomposition
        xcov = jnp.cov(x)
        d, V = jnp.linalg.eigh(xcov)  # Ascending order
        # Convert to descending order
        d = d[::-1]
        V = V[:, ::-1]

        zeroeigval = jnp.sum((d[:num_comp] / d[0]) < zerotolerance)
        if zeroeigval > 0:  # Do not allow zero eigenval
            raise ValueError(
                'zero eigenvalue encountered in PCA decomposition')

        # Calculate contribution ratio
        contratio = jnp.sum(d[:num_comp]) / jnp.sum(d)
        if verbose:
            print("    contribution ratio={0:f}".format(contratio))

        # Construct whitening and dewhitening matrices
        dsqrt = jnp.sqrt(d[:num_comp])
        dsqrtinv = 1 / dsqrt
        V = V[:, :num_comp]
        # Whitening
        W = jnp.dot(jnp.diag(dsqrtinv), V.transpose())  # whitening matrix
        A = jnp.dot(V, jnp.diag(dsqrt))  # de-whitening matrix
        x = jnp.dot(W, x)

        params = {'mean': xmean, 'W': W, 'A': A}

        # Check
        datacov = jnp.cov(x)

    return x.T, params
