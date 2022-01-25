import optax
from optax import GradientTransformation
from optax._src.base import NO_PARAMS_MSG
import jax
import chex
import jax.numpy as jnp
import functools
from optax._src import numerics, wrappers
from flax.core.frozen_dict import FrozenDict
from flax.training import train_state
from mreserve.checkpoint import f32_to_bf16, bf16_to_f32
from optax._src.factorized import _factored_dims
import numpy as np
from typing import NamedTuple, Any


class ScaleByAdamState(NamedTuple):
    """State for the Adam algorithm."""
    count: chex.Array
    mu: optax.Updates
    nu: optax.Updates


def _bias_correction(moment, decay, count):
    """Perform bias correction. This becomes a no-op as count goes to infinity."""
    bias_correction = 1 - decay ** count
    return jax.tree_map(lambda t: t / bias_correction.astype(t.dtype), moment)


#########################
# Bfloat16
# using this as the sign bit
# cube root to exchange range for mantissa precision
# i think this encoding probably isn't as good as I would want were i to do this again.
# the issue is for really small numbers we almost never choose the negative option
missing_precision = 1 + (1 / 2 ** 9)

def _unsigned_bfloat16_decode(v):
    v_abs = jnp.abs(v).astype(jnp.float32)
    v_abs = jax.lax.select(v >= 0, v_abs, v_abs * missing_precision)
    return jnp.cbrt(v_abs)


def _unsigned_bfloat16_encode(v):
    v_pow = jnp.power(v, 3)
    v_bf = v_pow.astype(jnp.bfloat16)
    v_bf32 = v_bf.astype(jnp.float32)

    err0 = jnp.abs(v_bf32 - v_pow)
    err1 = jnp.abs(v_bf32 * missing_precision - v_pow)
    return jax.lax.select(err0 < err1, v_bf, -v_bf)


def scale_by_bfloat16_adam(
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        eps_root: float = 0.0,
        use_bfloat16=True,
        do_bias_correction=True,
) -> GradientTransformation:
    """
    Scales by bfloat16 adam
    :param b1:
    :param b2:
    :param eps:
    :param eps_root:
    :param use_bfloat16:
    :param do_bias_correction:
    :return:
    """
    if not use_bfloat16:
        assert do_bias_correction
        return optax.scale_by_adam(b1, b2, eps, eps_root)

    _init = functools.partial(jnp.zeros_like, dtype=jnp.bfloat16)

    def init_fn(params):
        running_m = jax.tree_map(_init, params)
        running_v = jax.tree_map(_init, params)
        return ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=running_m, nu=running_v)

    def _momentum_update(grad, current_m):
        # Cast up here
        current_m = current_m.astype(jnp.float32)
        next_m = (1 - b1) * grad + b1 * current_m
        return next_m

    def _secondorder_update(grad, current_v):
        current_v_dec = _unsigned_bfloat16_decode(current_v)
        next_v = (1 - b2) * jnp.square(grad) + b2 * current_v_dec
        return next_v

    def update_fn(updates, state, params=None):
        del params

        next_m = jax.tree_multimap(_momentum_update, updates, state.mu)
        next_m_enc = jax.tree_map(lambda x: x.astype(jnp.bfloat16), next_m)

        next_v = jax.tree_multimap(_secondorder_update, updates, state.nu)
        next_v_enc = jax.tree_map(_unsigned_bfloat16_encode, next_v)

        count_inc = numerics.safe_int32_increment(state.count)
        if do_bias_correction:
            next_m = _bias_correction(next_m, b1, count_inc)
            next_v = _bias_correction(next_v, b2, count_inc)

        # Apply updates
        updates = jax.tree_multimap(
            lambda m, v: m / (jnp.sqrt(v + eps_root) + eps), next_m, next_v)

        return updates, ScaleByAdamState(count=count_inc, mu=next_m_enc, nu=next_v_enc)

    return GradientTransformation(init_fn, update_fn)


def lr_scale_linearwarmup_cosinedecay(num_warmup_steps, num_train_steps, final_lr_scale=0.1):
    """
    :param num_warmup_steps: Linear warmup for this many steps
    :param num_train_steps: Cosine decay for num_train_steps - num_warmup_steps
    :param final_lr_scale: We will end at this * learning_rate
    :return:
    """
    assert num_warmup_steps <= num_train_steps

    def schedule(step):
        warmup_scale = step / num_warmup_steps
        post_warmup_scale = (step - num_warmup_steps) / (num_train_steps - num_warmup_steps + 1.0)
        post_warmup_scale = jnp.minimum(post_warmup_scale, 1.0)

        # linear -> cosine
        post_warmup_scale = (1.0 - (1.0 - jnp.cos(jnp.pi * post_warmup_scale)) / 2.0)
        post_warmup_scale = final_lr_scale + (1.0 - final_lr_scale) * post_warmup_scale

        return jax.lax.select(step < num_warmup_steps, warmup_scale, post_warmup_scale)

    return schedule


def lr_scale_linearwarmup_lineardecay(num_warmup_steps, num_train_steps):
    """
    :param num_warmup_steps: Linear warmup for this many steps
    :param num_train_steps: Linear decay for num_train_steps - num_warmup_steps
    :param final_lr_scale: We will end at this * learning_rate
    :return:
    """
    assert num_warmup_steps <= num_train_steps

    def schedule(step):
        warmup_scale = step / num_warmup_steps
        post_warmup_scale = (step - num_warmup_steps) / (num_train_steps - num_warmup_steps + 1.0)
        post_warmup_scale = 1.0 - jnp.minimum(post_warmup_scale, 1.0)
        return jax.lax.select(step < num_warmup_steps, warmup_scale, post_warmup_scale)

    return schedule


def construct_train_state(opt_config, model, params, return_chainables=False):
    """
    :param optimizer_params: Dict like
        {
      learning_rate: 0.0001
      num_train_steps: 60000 # 5 epochs
      num_warmup_steps: 10000
      weight_decay_rate: 0.1
      beta_2: 0.98
      clip_norm: 0.0
      adafactor: False
      use_bfloat16_adam: True
      }
    :return:
    """
    opt = scale_by_bfloat16_adam(b1=opt_config.get('beta_1', 0.9),
                                 b2=opt_config.get('beta_2', 0.98),
                                 eps=opt_config.get('eps', 1e-8),
                                 use_bfloat16=opt_config.get('use_bfloat16_adam', True),
                                 do_bias_correction=opt_config.get('do_bias_correction', False),
                                )

    chainables = [
        opt,
        optax.add_decayed_weights(weight_decay=opt_config['weight_decay_rate'],
                                  mask=lambda p: jax.tree_map(lambda x: x.ndim > 1, p),
                                  ),
        optax.scale_by_schedule(lr_scale_linearwarmup_cosinedecay(num_warmup_steps=opt_config['num_warmup_steps'],
                                                                  num_train_steps=opt_config['num_train_steps'],
                                                                  final_lr_scale=opt_config.get('final_lr_scale', 0.02),
                                                                  )),
        optax.scale(-opt_config['learning_rate']),
    ]
    if return_chainables:
        return chainables

    tx = optax.chain(*chainables)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
