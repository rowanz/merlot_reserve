"""
Optimization for finetuning, for tpu vm

a few efficient tweaks -- doing optimizer sharding and weight decay to the initial values
"""
import sys
from flax import jax_utils
from jax._src.api import device_put_sharded
import optax
from typing import Callable, Dict, Tuple
sys.path.append('../')
from pretrain.optimization import *


class DecayedWeightsDeltaState(NamedTuple):
    """Overall state of the gradient transformation."""
    orig_params: chex.Array  # Momentum


def subtract_old_weights(weight_decay: float = 0.0, mask=None) -> GradientTransformation:
    """
    :param weight_decay: Add parameter scaled by `weight_decay` but subtracted by the original parameter -- ie so we don't drift
                         too far away
    :return:
    """
    def init_fn(params):
        return DecayedWeightsDeltaState(orig_params=jax.tree_map(lambda x: x.astype(jnp.bfloat16), params))

    def update_fn(updates, state, params=None):
        updates = jax.tree_multimap(lambda g, orig_param: g - weight_decay * orig_param.astype(g.dtype),
                                    updates, state.orig_params)
        return updates, state

    return wrappers.masked(GradientTransformation(init_fn, update_fn), mask=mask)


def _shard_opt(x):
    """
    Put adam variables on different cores of a tpu
    :param x:
    :return:
    """
    if (x.ndim == 0) or (x.shape[0] % 8 != 0):
        return jax_utils.replicate(x)
    else:
        x_shape2 = [8, x.shape[0] // 8] + list(x.shape[1:])

    # Manual replicate
    devices = jax.local_devices()
    x = x.reshape(x_shape2)
    x = device_put_sharded([x[i] for i in range(8)], devices)
    return x


def construct_finetuning_train_state(opt_config, model, params, only_state=False):
    """
    constructs it from initialization

    note -- we cannot load and restart with this

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
    :param model: -- Model to use
    :param params: -- initial model weights
    :return:
    """
    def mask_fn(p):
        return jax.tree_map(lambda x: (x.ndim > 1) and (x.size > 4096), p)

    tx_fns = [
        scale_by_bfloat16_adam(b1=opt_config.get('beta_1', 0.9),
                               b2=opt_config.get('beta_2', 0.98),
                               eps=opt_config.get('eps', 1e-6),
                               use_bfloat16=opt_config.get('use_bfloat16_adam', True),
                               do_bias_correction=opt_config.get('do_bias_correction', True),
                               ),
        subtract_old_weights(weight_decay=opt_config['weight_decay_rate'], mask=mask_fn),
        optax.add_decayed_weights(weight_decay=opt_config['weight_decay_rate'], mask=mask_fn),
        optax.scale_by_schedule(lr_scale_linearwarmup_lineardecay(num_warmup_steps=opt_config['num_warmup_steps'],
                                                                  num_train_steps=opt_config['num_train_steps'])),
        optax.scale(-opt_config['learning_rate']),
    ]
    tx = optax.chain(*tx_fns)

    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    if only_state:
        return state

    # move state to device
    state = state.replace(step=jax_utils.replicate(state.step))
    state = state.replace(opt_state=jax.tree_map(_shard_opt, state.opt_state))
    state = state.replace(params=jax_utils.replicate(state.params))

    return state, tx_fns


def finetune_train_step(state: train_state.TrainState, batch,
                        loss_fn: Callable[[train_state.TrainState, FrozenDict, Dict], Tuple[float, Dict]], tx_fns,
                        scan_minibatch=False):
    """
    Note: we'll compile this with pmap so no need to jit
    :param state:
    :param batch:
    :param loss_fn: something like `loss_fn_given_preds_nonlocalized`
    :param tx_fns: the raw optimizer functions
    :param num_separate_accumlations: Make this number bigger (not 1) to save memory maybe
    :param scan_minibatch: whether to get the gradients by scanning
    :return:
    """
    def _loss_fn(params):
        return loss_fn(state, params, batch)

    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
    params = f32_to_bf16(state.params)

    if scan_minibatch:
        # split up the batch into sub-batches of size exactly 1
        # note this technically adds the gradients, but that's probably OK since adam will scale everything back right?
        def _microbatch(old_grads, microbatch):
            def _loss_fn(params):
                return loss_fn(state, params, {k: v[None] for k, v in microbatch.items()})
            grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
            (loss, loss_info), grads = grad_fn(params)
            grads = jax.tree_multimap(lambda a, b: a + b, old_grads, grads)
            return grads, (loss, loss_info)

        grads, (loss, loss_info) = jax.lax.scan(_microbatch,
                                                init=jax.tree_map(lambda x: jnp.zeros_like(x, dtype=jnp.bfloat16), params),
                                                xs=batch)
        loss_info = jax.tree_map(lambda x: x.mean(), loss_info)
    else:
        (loss, loss_info), grads = grad_fn(params)

    grads = jax.tree_map(lambda x: jnp.nan_to_num(x, copy=False), grads)
    grads = jax.lax.pmean(grads, axis_name='batch')

    # ##################
    # Adam sharding
    def _idx_grad(x):
        if x.shape[0] % 8 != 0:
            return x
        else:
            x = jnp.reshape(x, [8, x.shape[0] // 8] + list(x.shape[1:]))
            idx = jax.lax.axis_index('batch') % 8
            return x[idx]
    updates = jax.tree_map(_idx_grad, grads)
    updates = bf16_to_f32(updates)

    # HACK - do the first two things separately
    assert len(state.opt_state) == 5
    updates, nos0 = tx_fns[0].update(updates, state.opt_state[0], None)
    updates, nos1 = tx_fns[1].update(updates, state.opt_state[1], None)

    # for weight decay and everything else now we move updates back to the right shape
    def _fix_grad(update, param):
        if update.shape == param.shape:
            return update
        else:
            aig = [[(j * 8 + i) for i in range(8)] for j in range(jax.device_count() // 8)]
            update = jax.lax.all_gather(update, axis_name='batch', axis_index_groups=aig)
            return jnp.reshape(update, param.shape)
    updates = jax.tree_multimap(_fix_grad, updates, state.params)

    # do the final few updates -- weight decay, scale by schedule, scale by LR. as weight decay requires
    # existing params
    new_opt_state = [nos0, nos1]
    for i in range(3):
        updates, nos = tx_fns[i + 2].update(updates, state.opt_state[i + 2], state.params)
        new_opt_state.append(nos)

    new_params = optax.apply_updates(state.params, updates)

    # Average metrics over all replicas
    loss_info = jax.lax.pmean(loss_info, axis_name='batch')
    loss_info = bf16_to_f32(loss_info)

    new_state = state.replace(
        step=state.step + 1,
        params=new_params,
        opt_state=tuple(new_opt_state),
    )
    return new_state, loss_info
