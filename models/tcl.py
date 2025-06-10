from audioop import add
from collections import namedtuple
import functools
import time
import haiku as hk
import jax.numpy as jnp
import jax
from jax import jit
import optax
import wandb
from .net import mlp
from .util import Batch, OptState
from typing import Tuple
from pprint import pprint

# use namedtuple to be able to have dot notation `.encode'
TCLApply = namedtuple('TCLApply', ['encode', 'predict', 'feed_forward'])


class TCL():
    """ 
    Reimplementation of the Time-Contrastive-Learning setup.
    The model is essentially just a feedforward neural network
    with special activation functions (`maxout')
    """

    def __init__(self,
                 hidden_units,
                 n_classes,
                 opt='sgd',
                 lr=1e-3,
                 momentum=0.9,
                 moving_avg_decay=0.999,
                 lambda_reg=1e-4,
                 decay_steps=int(5e5),
                 decay_factor=0.1,
                 activation='maxout',
                 random_seed=42,
                 verbose=False):
        super().__init__()
        self._n_classes = n_classes
        self._lambda_reg = lambda_reg
        self._momentum = momentum
        self._seed = random_seed
        if activation == 'maxout':
            # double the hidden sizes because the maxout activation halves them
            self._hidden_units = [2 * h for h in hidden_units]
            # just the last one does not have maxout, so do not double
            self._hidden_units[-1] = hidden_units[-1]
        else:
            self._hidden_units = hidden_units
        self._activation = activation
        self.verbose = verbose
        self._moving_avg_decay = moving_avg_decay

        def tcl():
            # init function
            # - use TCLApply for dot notation of apply function
            return self.feed_forward, TCLApply(self.encode, self.predict,
                                               self.feed_forward)

        self._net = hk.without_apply_rng(hk.multi_transform(tcl))
        if opt == 'adam':
            self._opt = optax.adam(lr)
        elif opt == 'adamw':
            self._opt = optax.adamw(lr, weight_decay=self._lambda_reg)
        elif opt == 'sgd':
            self._scheduler = optax.exponential_decay(
                init_value=lr,
                transition_steps=decay_steps,
                decay_rate=decay_factor,
                staircase=False)
            self._opt = optax.sgd(learning_rate=self._scheduler,
                                  momentum=momentum)
        else:
            raise ValueError("{} optimizer not implemented".format(opt))

        self._ema_fn = hk.transform_with_state(
            lambda x: hk.EMAParamsTree(moving_avg_decay)(x))

    ### Init functions
    def init(self,
             key: jax.random.PRNGKey,
             x: jnp.ndarray,
             pretrain: bool = False) -> Tuple[hk.Params, OptState]:
        params = self._net.init(key, x)
        if self.verbose:
            print("parameter shapes:")
            pprint(jax.tree_map(jnp.shape, params))
        if pretrain:
            params, non_trainable_params = hk.data_structures.partition(
                lambda m, n, p: m == "last_linear", params)
            opt_state = self._opt.init(params)
        else:
            opt_state = self._opt.init(params)
            non_trainable_params = {}
        return params, non_trainable_params, opt_state

    def reinit_opt(self, params: hk.Params) -> OptState:
        opt_state = self._opt.init(params)
        return opt_state

    def init_ema(self, params: hk.Params) -> OptState:
        _, ema_state = self._ema_fn.init(None, params)
        return ema_state

    ### model functions
    def feed_forward(self, x: jnp.ndarray) -> jnp.ndarray:
        hidden = self.encode(x)
        logits = self.predict(hidden)
        return logits

    def encode(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Get the last hidden layer *before*
        another linear transformation and softmax.
        This corresponds to the `hidden' representation of the
        input x.
        """
        # same model as original TCL paper: maxout hidden units
        _mlp = mlp(hidden_units=self._hidden_units, act=self._activation)
        h = _mlp(x)
        # original TCL code and paper does absolute 'activation' before the last hidden
        return jnp.abs(h)
        # return h ** 2

    def predict(self, hidden: jnp.ndarray):
        _linear = hk.Linear(self._n_classes, name='last_linear')
        logits = _linear(hidden)
        return logits

    #### training and eval
    # Training loss
    @functools.partial(jit, static_argnums=0)
    def loss(self, trainable_params: hk.Params,
             non_trainable_params: hk.Params, x: jnp.ndarray,
             y: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the loss of the network:
        cross-entropy plus L2 for regularization.
        """

        params = hk.data_structures.merge(trainable_params,
                                          non_trainable_params)

        logits = self._net.apply.feed_forward(params, x)
        # Generate one_hot labels from index classes
        labels = jax.nn.one_hot(y, self._n_classes)

        # Compute mean softmax cross entropy over the batch
        softmax_xent = -jnp.sum(labels * jax.nn.log_softmax(logits))
        softmax_xent /= labels.shape[0]

        # Compute the weight decay loss by penalising the norm of parameters
        # ignoring the bias term
        params_without_bias, _ = hk.data_structures.partition(
            lambda m, n, p: n != "b", params)
        l2_loss = sum(
            jnp.sum(jnp.square(p))
            for p in jax.tree.leaves(params_without_bias))

        return softmax_xent + self._lambda_reg * 0.5 * l2_loss

    @functools.partial(jit, static_argnums=0)
    def update(self, trainable_params: hk.Params,
               non_trainable_params: hk.Params, opt_state: OptState,
               x: jnp.ndarray, y: jnp.ndarray) -> Tuple[hk.Params, OptState]:
        """Learning rule (stochastic gradient descent)."""
        # Use jax transformation `grad` to compute gradients;
        # it expects the parameters of the model and the input batch
        loss, grads = jax.value_and_grad(self.loss)(trainable_params,
                                                    non_trainable_params, x, y)

        # Compute parameters updates based on gradients and optimiser state
        updates, opt_state = self._opt.update(grads,
                                              opt_state,
                                              params=trainable_params)

        # Apply updates to parameters
        new_params = optax.apply_updates(trainable_params, updates)
        return loss, new_params, opt_state

    @functools.partial(jit, static_argnums=0)
    def update_ema(self, params: hk.Params,
                   ema_state: OptState) -> Tuple[hk.Params, OptState]:
        """Exponential weighted average of parameters."""
        return self._ema_fn.apply(None, ema_state, None, params)

    @functools.partial(jit, static_argnums=0)
    def accuracy(self, params: hk.Params, x: jnp.ndarray,
                 y: jnp.ndarray) -> jnp.ndarray:
        # Get network predictions
        predictions = self._net.apply.feed_forward(params, x)
        # Return accuracy = how many predictions match the ground truth
        return jnp.mean(jnp.argmax(predictions, axis=-1) == y)

    def get_hidden(self, params: hk.Params, x: jnp.ndarray) -> jnp.ndarray:
        return self._net.apply.encode(params, x)
