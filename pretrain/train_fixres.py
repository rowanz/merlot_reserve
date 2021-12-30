"""
Trains the model at higher resolution for one more epoch

It often OOM's so you need to do some special memory management on some expensive parts (the adam momentum / variance)

python3 train_fixres.py configs/large.yaml -ckpt_path=...
python3 train_fixres.py configs/base.yaml -ckpt_path=...
"""

import sys

sys.path.append('../')
import os

import yaml
from datetime import datetime
import pytz
import jax
import jax.numpy as jnp
from pretrain.dataloader import input_fn_builder
from pretrain.pretrain_model import *
from flax import jax_utils
from pretrain.optimization import construct_train_state, ScaleByAdamState
from mreserve.checkpoint import save_checkpoint, load_checkpoint, bf16_to_f32
import argparse
import numpy as np
import functools
import time
import optax
from jax._src.api import device_put_sharded
from flax.core.frozen_dict import freeze

jax.config.update('jax_log_compiles', True)
assert any([x.platform == 'tpu' for x in jax.local_devices()])
print('JAX process: {} / {}. Local devices {}. Using {}'.format(
    jax.process_index(), jax.process_count(), jax.local_devices(), 'TPU'), flush=True)

parser = argparse.ArgumentParser(description='Train model!')
parser.add_argument(
    'config_file',
    help='Where the config.yaml is located',
    type=str,
)

parser.add_argument(
    '-disable_wandb',
    help='dont log this result on weights and biases',
    dest='disable_wandb',
    action='store_true',
)
parser.add_argument(
    '-ckpt_path',
    dest='ckpt_path',
    type=str,
)
args = parser.parse_args()

print(f"Loading from {args.config_file}", flush=True)
with open(args.config_file, 'r') as f:
    config = yaml.load(f, yaml.FullLoader)

    seattle_time = pytz.utc.localize(datetime.utcnow()).astimezone(pytz.timezone('America/Los_Angeles'))
    seattle_time = seattle_time.strftime("%Y-%m-%d-%H:%M.%S")

    config['device']['output_dir'] = os.path.join(config['device']['output_dir'], 'fixres')

config['_path'] = args.config_file
if (jax.process_index() == 0) and (not args.disable_wandb):
    import wandb
    # wandb.init(... add your args here ...)
else:
    wandb = None

# Set the resolution to something
# possible_res = [[12, 20], [14, 18], [16, 16],
#                 [16, 30], [16, 30], [18, 28], [20, 24], [22, 22]]
# possible_res = [[16, 30], [20, 24]]
possible_res = [[18, 32], [24, 24]]

res = possible_res[jax.process_index() % len(possible_res)]
config['model']['output_grid'] = res


config['data']['random_scale_max'] = max(min(res) / max(res) * 16 / 9, 1.0) + 0.1
config['data']['shrink_both_sides'] = False
config['data']['random_scale_min'] = 1.0
config['data']['max_text_seq_len'] = 1024 # hard-coded due to rotary :/
config['data']['do_flip_if_vertical'] = False

config['data']['seq_len'] = config['data']['lang_seq_len'] + 8 * (res[0] * res[1]) // 4

ds_train_iter = input_fn_builder(config)

model = MerlotReservePretrainer.from_config(config)

# we need to load it
state = load_checkpoint(path=args.ckpt_path, step=None, use_bfloat16_weights=False)

# start_step -- 750k for logging
# but want internal step to be 0 for saving, and for LR sched
start_step = int(state.pop('step'))
opt_state = state.pop('opt_state')
opt_state = [opt_state[str(i)] for i in range(4)]
opt_state[0] = ScaleByAdamState(count=jnp.array(0, dtype=jnp.int32),
                                nu=freeze(opt_state[0]['nu']),
                                mu=freeze(opt_state[0]['mu']))
# opt_state[2] = = jnp.array(0, dtype=jnp.int32) # start LR sched from beginning
opt_state[1] = optax.MaskedState(optax.AddDecayedWeightsState())
opt_state[2] = optax.ScaleByScheduleState(count=jnp.array(0, dtype=jnp.int32))
opt_state[3] = optax.ScaleState()
state = train_state.TrainState(opt_state=opt_state,
                               params=freeze(state.pop('params')),
                               step=0,
                               apply_fn=model.apply,
                               tx=None,
                               )


def _shard_opt(x):
    if (x.ndim == 0) or (x.shape[0] % 8 != 0):
        return jax_utils.replicate(x)
    else:
        x_shape2 = [8, x.shape[0] // 8] + list(x.shape[1:])

    # Manual replicate
    devices = jax.local_devices()
    x = x.reshape(x_shape2)
    x = device_put_sharded([x[i] for i in range(8)], devices)
    return x

# def _shard_opt_after_loading(x):
#     assert x.shape[0] % 8 == 0
#     devices = jax.local_devices()
#     x = device_put_sharded([x[i] for i in range(8)], devices)
#     return x

state = state.replace(step=jax_utils.replicate(state.step))
state = state.replace(opt_state=jax.tree_map(_shard_opt, state.opt_state))
state = state.replace(params=jax_utils.replicate(state.params))

config['optimizer']['num_train_steps'] = 75000
config['optimizer']['final_lr_scale'] = 0.0
config['optimizer']['num_warmup_steps'] = 15000
config['optimizer']['learning_rate'] = 0.02 * config['optimizer']['learning_rate']
tx_raw = construct_train_state(opt_config=config['optimizer'], model=model, params=None, return_chainables=True)

def train_step(state: train_state.TrainState, batch):
    """
    Note: we'll compile this with pmap so no need to jit
    :param state:
    :param batch:
    :param use_bfloat16_grads: Whether to use bfloat16 for storing grads. I think it is probably OK considering
                               momentum is bfloat16 anyways. i'm just going to cast down (rounding down here rather
                               than to nearest or anything)
    :return:
    """
    def _loss_fn(params):
        return loss_fn_given_preds(state.apply_fn({'params': params}, batch))

    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)

    params = f32_to_bf16(state.params)

    (loss, loss_info), grads = grad_fn(params)

    grads = jax.tree_map(lambda x: jnp.nan_to_num(x, copy=False), grads)
    grads = jax.lax.pmean(grads, axis_name='batch')

    # # mega model is really unstable so i'm adding clip by global norm to 1.0
    # max_norm = 1.0
    # g_norm = optax._src.linear_algebra.global_norm(grads)
    # g_norm = jnp.maximum(g_norm, max_norm)
    # grads = jax.tree_map(lambda t: (t / g_norm) * max_norm, grads)
    # loss_info['g_norm'] = g_norm

    # ##################
    # Adam sharding
    def _idx_grad(x):
        if x.shape[0] % 8 != 0:
            return x
        else:
            x = jnp.reshape(x, [8, x.shape[0] // 8] + list(x.shape[1:]))
            idx = jax.lax.axis_index('batch') % 8
            return x[idx]
    grads = jax.tree_map(_idx_grad, grads)

    # HACK - do adam separately
    updates, new_opt_state = tx_raw[0].update(grads, state.opt_state[0], None)

    # for weight decay and everything else now we move updates back to the right shape
    def _fix_grad(update, param):
        if update.shape == param.shape:
            return update
        else:
            aig = [[(j * 8 + i) for i in range(8)] for j in range(jax.device_count() // 8)]
            update = jax.lax.all_gather(update, axis_name='batch', axis_index_groups=aig)
            return jnp.reshape(update, param.shape)

    updates = jax.tree_multimap(_fix_grad, updates, state.params)

    # Cast up grads here (after the pmean) which reduces bandwidth maybe
    updates = bf16_to_f32(updates)

    new_opt_state = [new_opt_state]
    for i in range(3):
        print(i,flush=True)
        updates, nos = tx_raw[i + 1].update(updates, state.opt_state[i + 1], state.params)
        new_opt_state.append(nos)

    new_params = optax.apply_updates(state.params, updates)

    # Average metrics over all replicas (maybe this isn't a great idea, idk)
    loss_info = jax.lax.pmean(loss_info, axis_name='batch')
    loss_info = bf16_to_f32(loss_info)

    new_state = state.replace(
        step=state.step + 1,
        params=new_params,
        opt_state=tuple(new_opt_state),
    )
    return new_state, loss_info

p_train_step = jax.pmap(train_step,  axis_name='batch', donate_argnums=(0, 1,))

train_metrics = []
time_elapsed = []
num_train_steps = config['optimizer'].get('num_train_steps_override', config['optimizer']['num_train_steps'])
log_every = config['device'].get('commit_every_nsteps', 50)

for n in range(num_train_steps):
    st = time.time()
    batch = next(ds_train_iter)
    state, loss_info = p_train_step(state, batch)

    # Async transfer. Basically we queue the last thing, then log the thing from `log_every` iterations ago
    if jax.process_index() == 0:
        train_metrics.append(jax.tree_map(lambda x: x[0], loss_info))
        jax.tree_map(lambda x: x.copy_to_host_async(), train_metrics[-1])

        step_for_logging = n - log_every
        if step_for_logging >= 0:
            train_metrics[step_for_logging] = {k: float(v) for k, v in train_metrics[step_for_logging].items()}

            if wandb is not None:
                wandb.log(train_metrics[step_for_logging], step=step_for_logging + start_step, commit=(n + 1) % log_every == 0)

    if (n + 1) % config['device']['iterations_per_loop'] == 0:
        save_checkpoint(state, path=config['device']['output_dir'], no_optimizer=True)
        print(f"Saving @iter {n:03d}.", flush=True)
        temps = {}
        for i, k in enumerate(['imgs_to_audio', 'text_to_audio', 'stuff_to_span']):
            temps[k] = state.params._dict['contrastive_scales'][0, i].astype(jnp.float32)
        temps = jax.device_get(temps)
        for k, v in temps.items():
            print("{} temperature: log={:.3f}  exp={:.3f}".format(k, v, np.exp(v)), flush=True)

    time_elapsed.append(time.time() - st)
    if len(time_elapsed) >= 100:
        tsum = sum(time_elapsed)
        print("Completed 100 batches in {:.3f}sec, avg {:.3f} it/sec".format(tsum, 100.0/tsum), flush=True)
        time_elapsed = []
