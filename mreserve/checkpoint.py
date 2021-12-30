from flax.training import checkpoints
from flax.training import train_state
import jax
from typing import Optional, Any
import clu.parameter_overview
import operator
import jax.numpy as jnp


def _treemap_cast(from_dtype, to_dtype, tree):
    """
    Convert leaves in a tree from `from_dtype` to `to_dtype`
    :param from_dtype:
    :param to_dtype:
    :param tree:
    :return:
    """

    def _do_cast(x):
        if not hasattr(x, 'dtype'):  # for ints and stuff
            return x
        if x.dtype == from_dtype:
            return x.astype(to_dtype)
        return x

    return jax.tree_map(_do_cast, tree)


def _compress_state(state: train_state.TrainState):
    """
    For saving i'll cast float32 down to float16, keep bfloat unchanged
    I'm doing this because float16 has more precision
    :param state:
    :return:
    """
    return _treemap_cast(from_dtype=jnp.float32, to_dtype=jnp.float16, tree=state)


def _decompress_state(state: train_state.TrainState):
    return _treemap_cast(from_dtype=jnp.float16, to_dtype=jnp.float32, tree=state)


def bf16_to_f32(params):
    """
    Cast params to float32
    :param params:
    :return:
    """
    return _treemap_cast(from_dtype=jnp.bfloat16, to_dtype=jnp.float32, tree=params)


def f32_to_bf16(params):
    """
    Cast params to float32
    :param params:
    :return:
    """
    return _treemap_cast(from_dtype=jnp.float32, to_dtype=jnp.bfloat16, tree=params)


def save_checkpoint(state: train_state.TrainState, path: str, keep=None, overwrite=True, with_shard_optimizer=False,
                    no_optimizer=False):
    """
    :param state:
    :param path: Path where we'll save stuff to
    :param keep: If specified this is how many we should keep
    :param overwrite: If we should overwrite
    :return:
    """
    step = int(state.step[0])

    if keep is None:
        keep = 100000000

    if jax.process_index() == 0:
        print(f"Saving checkpoint at save {step}, path {path}", flush=True)

        if with_shard_optimizer:
            print("Dealing with sharded optimizer", flush=True)
            params = jax.device_get(jax.tree_map(lambda x: x[0], state.params))
            opt_state = jax.device_get(state.opt_state)
            state = state.replace(step=step,
                                  params=params,
                                  opt_state=opt_state,
                                  )
        elif no_optimizer:
            print("Not including the optimizer state", flush=True)
            params = jax.device_get(jax.tree_map(lambda x: x[0], state.params))
            state = state.replace(step=step,
                                  params=params,
                                  opt_state=None,
                                  )
        else:
            # Get first replica
            state = jax.device_get(jax.tree_map(lambda x: x[0], state))

        state = _compress_state(state)

        checkpoints.save_checkpoint(path, state, step=step, prefix='ckpt_', keep=keep, overwrite=overwrite)


def load_checkpoint(path: str, state: Optional[train_state.TrainState] = None, step=None, use_bfloat16_weights=False):
    """
    Loads a checkpoint. I'm saving the weights in float16 and the adam variables in a weird bfloat16 format.
    :param state:
    :param path:
    :param step:
    :param to_float32: Whether to convert weights to float32 -- needed for training
    :return:
    """
    # Temporarily compress the state to be equal to what we're loading
    if state is not None:
        state = _compress_state(state)

    state = checkpoints.restore_checkpoint(ckpt_dir=path, target=state, step=step, prefix='ckpt_', parallel=True)
    state = _decompress_state(state)
    if use_bfloat16_weights:
        state = state.replace(params=f32_to_bf16(state.params))
    return state


def log_param_shapes(params: Any) -> int:
    """
    # Maybe could be useful:
    https://github.com/google-research/scenic/blob/ab3083d8cbfe3216119a0f24fce23ca988e20355/scenic/common_lib/debug_utils.py

    Prints out shape of parameters and total number of trainable parameters.
    Args:
    params: PyTree of model parameters.
    print_params_nested_dict: If True, it prints parameters in shape of a nested
      dict.
    Returns:
    int; Total number of trainable parameters.
    """
    print(clu.parameter_overview.get_parameter_overview(params))
    total_params = jax.tree_util.tree_reduce(operator.add, jax.tree_map(lambda x: x.size, params))
    # logging.info('Total params: %d', total_params)
    return total_params


def tree_map_nested_keys(f, params):
    """
    Tree map, but you get the KEY and the VALUE
    :param f: function returning nested keys joined by a '/' and values
    :param params:
    :return: new tree
    """
    leaves, treedef = jax.tree_util.tree_flatten(params)
    params_flat = clu.parameter_overview.flatten_dict(params)
    for i, k in enumerate(sorted(params_flat.keys())):
        assert params_flat[k] is leaves[i]
        leaves[i] = f(k, leaves[i])
    return treedef.unflatten(leaves)
