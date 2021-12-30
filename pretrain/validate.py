"""
Validates on TPUs

This isn't needed for training, but it could be useful if you want to track validation loss on held out data throughout training.

"""

import sys

sys.path.append('../')
import os
import yaml
from datetime import datetime
import pytz
import jax
import jax.numpy as jnp
from pretrain.dataloader import input_fn_builder, make_dataset
from pretrain.pretrain_model import MerlotReservePretrainer, train_step, loss_fn_given_preds
from flax import jax_utils
from tqdm import tqdm, trange
from flax.training import checkpoints
import argparse
import pandas as pd
from tensorflow.io import gfile


jax.config.update('jax_log_compiles', True)
is_on_gpu = any([x.platform == 'gpu' for x in jax.local_devices()])
if not is_on_gpu:
    assert any([x.platform == 'tpu' for x in jax.local_devices()])
print('JAX process: {} / {}. Local devices {}. Using {}'.format(
    jax.process_index(), jax.process_count(), jax.local_devices(), 'GPU' if is_on_gpu else 'TPU'), flush=True)

parser = argparse.ArgumentParser(description='Train model!')
parser.add_argument(
    'config_file',
    help='Where the config.yaml is located',
    type=str,
)
parser.add_argument(
    '-output_dir',
    help='Override output directory',
    dest='output_dir',
    type=str,
)
parser.add_argument(
    '-disable_wandb',
    help='dont log this result on weights and biases',
    dest='disable_wandb',
    action='store_true',
)
parser.add_argument(
    '-batch_size',
    help='New batch size to use, could be pretty large',
    dest='batch_size',
    default=32,
    type=int,
)
parser.add_argument(
    '-max_num_megabatch',
    help='Limit evaluation to this number of megabatches',
    dest='max_num_megabatch',
    default=4,
    type=int,
)
args = parser.parse_args()

print(f"Loading from {args.config_file}", flush=True)
if args.output_dir == '':
    raise ValueError("must provide output dir")

with open(args.config_file, 'r') as f:
    config = yaml.load(f, yaml.FullLoader)
    config['device']['output_dir'] = args.output_dir
    orig_batch_size = config['device']['batch_size']
    config['device']['batch_size'] = args.batch_size
    num_accumulations = orig_batch_size // config['device']['batch_size']

    assert orig_batch_size % args.batch_size == 0

config['_path'] = args.config_file
config['data']['do_random_scale'] = False
# config['data']['disable_audio_dataloader'] = True
# config['data']['disable_imgs_dataloader'] = True

# config['data']['convert_extra_span_to_text_prob'] = 1e-30
if args.disable_wandb:
    wandb = None
else:
    import wandb
    cfg_name = args.config_file.split('/')[-1]
    # wandb.init( ... )

checkpoint_files = gfile.glob(os.path.join(args.output_dir, 'ckpt_*'))
checkpoint_files = [x for x in checkpoint_files if (int(x.split('_')[-1]) == 7500) or (int(x.split('_')[-1]) % 75000 == 0)]
# checkpoint_files = [x for x in checkpoint_files if (int(x.split('_')[-1]) <= 750000)]
checkpoint_files = sorted(checkpoint_files, key=lambda x: int(x.split('_')[-1]))
# checkpoint_files = [checkpoint_files[-1]]

# Get representative training and validation fns
val_fns = gfile.glob(config['data']['train_fns'].split('train{:05d}')[0] + 'val*')
train_fns = [config['data']['train_fns'].format(i) for i in range(len(val_fns))]
print("Val fns: {}\nTrain fns: {}\n".format(val_fns, train_fns), flush=True)

def make_ds_iter(dset='train'):
    fns = train_fns if (dset == 'train') else val_fns
    train_dset = make_dataset(config, fns, args.batch_size, num_devices=jax.local_device_count(), is_training=False)
    train_dset_l = map(lambda y: jax.tree_map(lambda x: x._numpy(), y), iter(train_dset))
    return jax_utils.prefetch_to_device(train_dset_l, size=1)

config['model']['data'] = config['data']
model = MerlotReservePretrainer(config['model'])

def batched_preds(p, batch):
    preds = model.apply({'params': p}, batch)
    return preds
batched_preds_pmap = jax.pmap(batched_preds, axis_name='batch', donate_argnums=(1,))
loss_megabatch_pmap = jax.pmap(loss_fn_given_preds, axis_name='batch', donate_argnums=(0,))

# Loop over all checkpoint files, and both datasets (train/val)
info = []
for ckpt_fn in checkpoint_files:
    print(f"Loading {ckpt_fn}", flush=True)
    state_dict = checkpoints.restore_checkpoint(ckpt_fn, target=None)
    params = jax_utils.replicate(state_dict.pop('params'))

    for dset in ['val', 'train']:
        agg_loss_info = []
        ds_iter = make_ds_iter(dset=dset)
        outs = []
        for i, batch_i in enumerate(tqdm(ds_iter)):
            out = batched_preds_pmap(params, batch_i)
            outs.append(out)

            # Have enough to accumulate
            if len(outs) == num_accumulations:
                megabatch = jax.tree_multimap(lambda *xs: jnp.concatenate(xs, 1), *outs)
                loss_info = loss_megabatch_pmap(megabatch)[1]
                loss_info = jax.tree_map(lambda x: float(x.mean()), loss_info)
                agg_loss_info.append(loss_info)
                outs = []
            if len(agg_loss_info) == args.max_num_megabatch:
                break
        avg_loss_info = pd.DataFrame(agg_loss_info).mean(0)
        avg_loss_info['nbatch'] = len(agg_loss_info)
        avg_loss_info['dset'] = dset
        avg_loss_info['ckpt'] = ckpt_fn
        avg_loss_info['nstep'] = state_dict['step']
        info.append(avg_loss_info)
    if wandb is not None:
        bad_keys = ['nbatch', 'dset', 'ckpt', 'nstep']
        log_info = {k + '_train': v for k, v in info[-1].items() if k not in bad_keys}
        log_info.update({k + '_val': v for k, v in info[-2].items() if k not in bad_keys})
        step = int(state_dict['step'])
        wandb.log(log_info, step=step, commit=True)

info = pd.DataFrame(info)
print("TRAIN INFO:\n{}\n\nVAL INFO:\n{}".format(info[info['dset'] == 'train'], info[info['dset'] == 'val']), flush=True)