"""
Finetune a VCR model on TPUs, combining Q->A and QA->R on a single forward pass
"""

import sys

sys.path.append('../../')
import yaml
from datetime import datetime
import pytz
import jax
import jax.numpy as jnp
from pretrain.dataloader import input_fn_builder, MASK, encoder
from finetune.common_dataloader import finetune_input_fn_builder, finetune_val_input_fn_builder
from mreserve.modeling import MerlotReserve

from flax.training import train_state
from flax import jax_utils
import flax.linen as nn
from finetune.optimization import construct_finetuning_train_state, finetune_train_step
from mreserve.checkpoint import save_checkpoint, load_checkpoint, bf16_to_f32, f32_to_bf16
import argparse
import pandas as pd
import numpy as np
from flax.core.frozen_dict import freeze
from copy import deepcopy
import clu.parameter_overview
import functools
import time
import os

jax.config.update('jax_log_compiles', True)
is_on_gpu = any([x.platform == 'gpu' for x in jax.local_devices()])
print('JAX process: {} / {}. Local devices {}. Using {}'.format(
    jax.process_index(), jax.process_count(), jax.local_devices(), 'GPU' if is_on_gpu else 'TPU'), flush=True)

parser = argparse.ArgumentParser(description='Train model!')

parser.add_argument(
    'pretrain_config_file',
    help='Where the config.yaml is located',
    type=str,
)
parser.add_argument(
    'ckpt',
    help='checkpoint to use',
    type=str,
)
parser.add_argument(
    '-lr',
    help='lr',
    type=float,
)
parser.add_argument(
    '-ne',
    help='ne',
    type=int,
)
parser.add_argument(
    '-output_grid_h',
    help='output_grid_h',
    type=int,
    default=12,
)
parser.add_argument(
    '-output_grid_w',
    help='output_grid_w',
    type=int,
    default=20,
)
parser.add_argument(
    '-output_name',
    help='output_name',
    type=str,
    default='',
)
parser.add_argument(
    '-wandb_name',
    help='wandb_name',
    type=str,
    default='merlotreserve-vcr-hsweep2',
)
parser.add_argument(
    '-data_name',
    help='data_name',
    type=str,
    default='vcr_oct29_2021_segm',
)
args = parser.parse_args()

# print(f"Loading from {args.config_file}", flush=True)
with open(args.pretrain_config_file, 'r') as f:
    config = yaml.load(f, yaml.FullLoader)


config['data']['train_fns'] = "${your_path}/" + args.data_name + "/train{:03d}of256.tfrecord"
config['data']['num_train_files'] = 256
config['data']['num_answers'] = 4
config['data']['random_scale_max'] = 1.1
config['data']['random_scale_min'] = 1.0

config['device']['batch_size'] = 32
config['device']['prefetch_size'] = 0
config['device']['n_fns_per_cycle'] = 256

NUM_EPOCH = args.ne
TRAIN_SIZE = 212736
steps_per_epoch = TRAIN_SIZE // config['device']['batch_size']
config['optimizer'] = {
    'beta_2': 0.98,
    'eps': 1e-6,
    'learning_rate': args.lr,
    'num_train_steps': NUM_EPOCH * steps_per_epoch,
    'num_warmup_steps': int(0.5 * steps_per_epoch),
    'use_bfloat16_adam': True,
    'weight_decay_rate': 0.1,
    'do_bias_correction': True,
}

config['device']['iterations_per_loop'] = steps_per_epoch
config['data']['lang_seq_len'] = 144
cfg_name = args.pretrain_config_file.split('/')[-1]
seattle_time = pytz.utc.localize(datetime.utcnow()).astimezone(pytz.timezone('America/Los_Angeles'))
seattle_time = seattle_time.strftime("%Y-%m-%d-%H:%M.%S")
config['device']['output_dir'] = f'your_path/{args.data_name}/{cfg_name}/'
if args.output_name != '':
    config['device']['output_dir'] = os.path.join(config['device']['output_dir'], args.output_name)
config['device']['output_dir'] = os.path.join(config['device']['output_dir'], seattle_time)

np.random.seed(123456)
config['model']['output_grid'] = [args.output_grid_h, args.output_grid_w]
ds_train_iter = finetune_input_fn_builder(config, 'vcr')
_, dummy_batch = next(ds_train_iter)


import wandb
config['_ckpt'] = args.ckpt
tags = [cfg_name]
if args.output_name != '':
    tags.append(args.output_name)
# wandb.init( ... )


class MerlotReserveVCR(MerlotReserve):
    def setup(self):
        super().setup()
        self.proj = nn.Dense(features=1, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(stddev=0.02), name='proj',
                             use_bias=False)

    def __call__(self, batch):

        batch_size, two_, num_ans_per, token_length = batch['answers'].shape
        answers2d = batch['answers'].reshape(batch_size * 2 * num_ans_per, token_length)

        imgs_enc = self.vision_encoder(batch['image'])['seq_attnpool'].repeat(2 * num_ans_per, axis=0)

        mm_inputs = self.prepare_multimodal_inputs(
            tokens=answers2d,
            token_segment_idx=jnp.zeros([batch_size * 2 * num_ans_per, token_length], dtype=jnp.int32),
            vision_input=imgs_enc,
        )
        joint_encoding = self.joint_transformer(**mm_inputs)['seq']
        joint_encoding = joint_encoding[:, :token_length].reshape(batch_size * 2 * num_ans_per, token_length, self.hidden_size)

        # Pool from the right tokens
        pool_idx = jnp.argmax((answers2d == MASK).astype(jnp.float32), 1)
        pooled_h = joint_encoding[jnp.arange(batch_size * 2 * num_ans_per), pool_idx]

        logits = self.proj(pooled_h).reshape([batch_size, 2, num_ans_per])
        return logits


model = MerlotReserveVCR.from_config(config)

if args.ckpt == '':
    params = model.init_from_dummy_batch(dummy_batch).unfreeze()
else:
    params = load_checkpoint(args.ckpt)['params']

# Don't need those
for k in ['audio_encoder', 'head', 'span_encoder']:
    params.pop(k, None)
params['proj'] = {'kernel': np.random.randn(params['joint_transformer']['final_ln']['bias'].shape[0], 1).astype(np.float32) * 0.01}
params = freeze(params)

state, tx_fns = construct_finetuning_train_state(opt_config=config['optimizer'], model=model, params=params)

def train_loss_fn(state, params, batch):
    logits = state.apply_fn({'params': params}, batch)
    log_p = jax.nn.log_softmax(logits, axis=-1)
    labels_oh = jax.nn.one_hot(batch['labels'], dtype=log_p.dtype, num_classes=log_p.shape[-1])

    loss = -jnp.mean(jnp.sum(labels_oh * log_p, axis=-1))
    is_right = (jnp.argmax(log_p, -1) == batch['labels']).astype(jnp.float32).mean()
    return loss, {'is_right': is_right, 'loss': loss}


p_train_step = jax.pmap(functools.partial(finetune_train_step, loss_fn=train_loss_fn, tx_fns=tx_fns),
                                          axis_name='batch', donate_argnums=(0,1))


def pred_step(state: train_state.TrainState, batch):
    logits = state.apply_fn({'params': state.params}, batch)
    return {'logprobs': jax.nn.log_softmax(logits), 'preds': jnp.argmax(logits, -1)}
p_pred_step = jax.pmap(pred_step, axis_name='batch', donate_argnums=(1,))


def val_epoch(state: train_state.TrainState):
    """
    perform a validation epoch
    :param state:
    :return:
    """
    val_config = deepcopy(config)
    val_config['data']['val_fns'] = "your_path" + args.data_name + "/val{:03d}of008.tfrecord"
    val_config['data']['num_val_files'] = 8
    val_config['data']['do_random_scale'] = False
    val_iter = finetune_val_input_fn_builder(val_config, 'vcr')

    qa_preds = []
    qar_preds = []
    for ids, batch in val_iter:
        val_pred = p_pred_step(state, batch)
        val_argmax_pred = val_pred['preds'].reshape(-1, 2)
        labels = batch['labels'].reshape(-1, 2)
        for (p_i, id_i, label_i) in zip(val_argmax_pred, ids, labels):
            if id_i == 'pad':
                continue
            qa_preds.append({'pred': p_i[0], 'label': label_i[0], 'id': id_i})
            qar_preds.append({'pred': p_i[1], 'label': label_i[1], 'id': id_i})

    qa_preds = pd.DataFrame(qa_preds)
    qa_preds['is_right'] = qa_preds['pred'] == qa_preds['label']
    qa_acc = qa_preds['is_right'].mean()

    qar_preds = pd.DataFrame(qar_preds)
    qar_preds['is_right'] = qar_preds['pred'] == qar_preds['label']
    qar_acc = qar_preds['is_right'].mean()
    return {'qa_acc': qa_acc, 'qar_acc': qar_acc}

train_metrics = []
log_every = config['device'].get('commit_every_nsteps', 50)
time_elapsed = []

for n in range(config['optimizer']['num_train_steps']):
    st = time.time()
    id_, batch = next(ds_train_iter)
    state, loss_info = p_train_step(state, batch)

    if jax.process_index() == 0:
        train_metrics.append(jax.tree_map(lambda x: x[0], loss_info))
        jax.tree_map(lambda x: x.copy_to_host_async(), train_metrics[-1])

        step_for_logging = n - log_every
        if step_for_logging >= 0:
            train_metrics[step_for_logging] = {k: float(v) for k, v in train_metrics[step_for_logging].items()}
            wandb.log(train_metrics[step_for_logging], step=step_for_logging, commit=(n + 1) % log_every == 0)

        if (n + 1) % config['device']['iterations_per_loop'] == 0:
            print("Done 1 epoch", flush=True)

            save_checkpoint(state, path=config['device']['output_dir'], no_optimizer=True)
            val_info = val_epoch(state)
            print(f"Saving @iter {n:03d}.\nInfo: {pd.Series(val_info)}\n~\n", flush=True)
            wandb.log({k + '_val': v for k, v in val_info.items()}, step=step_for_logging, commit=True)

        time_elapsed.append(time.time() - st)
        if len(time_elapsed) >= 100:
            tsum = sum(time_elapsed)
            print("Completed 100 batches in {:.3f}sec, avg {:.3f} it/sec".format(tsum, 100.0 / tsum), flush=True)
            time_elapsed = []
