"""
Submit an existing VCR model to the leaderboard

"""
import optax

import sys

sys.path.append('../../')
import yaml
from datetime import datetime
import pytz
import jax
import jax.numpy as jnp
from pretrain.dataloader import input_fn_builder, MASK, encoder
from finetune.common_dataloader import finetune_input_fn_builder, finetune_val_input_fn_builder
from mreserve.modeling import MerlotReserve, unit_normalize

from flax.training import train_state
from flax import jax_utils
import flax.linen as nn
from mreserve.checkpoint import save_checkpoint, load_checkpoint, bf16_to_f32, f32_to_bf16
import argparse
import pandas as pd
import numpy as np
from flax.core.frozen_dict import freeze
from tqdm import tqdm
from collections import defaultdict

jax.config.update('jax_log_compiles', True)
is_on_gpu = any([x.platform == 'gpu' for x in jax.local_devices()])
print('JAX process: {} / {}. Local devices {}. Using {}'.format(
    jax.process_index(), jax.process_count(), jax.local_devices(), 'GPU' if is_on_gpu else 'TPU'), flush=True)

parser = argparse.ArgumentParser(description='Train model!')

# '../../pretrain/configs/ytt180m_base_v4_bsize=1024.yaml'
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
    '-darpa',
    help='darpa',
    action='store_true'
)
args = parser.parse_args()

# print(f"Loading from {args.config_file}", flush=True)
with open(args.pretrain_config_file, 'r') as f:
    config = yaml.load(f, yaml.FullLoader)

config['data']['num_answers'] = 4
config['device']['batch_size'] = 16
config['device']['prefetch_size'] = 0
config['device']['n_fns_per_cycle'] = 8
config['data']['val_fns'] = "${your_path}/test{:03d}of008.tfrecord"
config['data']['num_val_files'] = 8
config['data']['do_random_scale'] = False
config['data']['lang_seq_len'] = 140
cfg_name = args.pretrain_config_file.split('/')[-1]
seattle_time = pytz.utc.localize(datetime.utcnow()).astimezone(pytz.timezone('America/Los_Angeles'))
seattle_time = seattle_time.strftime("%Y-%m-%d-%H:%M.%S")
config['model']['output_grid'] = [args.output_grid_h, args.output_grid_w]

np.random.seed(123456)

val_iter = finetune_val_input_fn_builder(config, 'vcr')

# probably shouldn't duplicate this...
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

params = freeze(load_checkpoint(args.ckpt)['params'])
params = f32_to_bf16(params)
state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optax.identity())
state = jax_utils.replicate(state)

def pred_step(state: train_state.TrainState, batch):
    logits = state.apply_fn({'params': state.params}, batch)
    return {'logprobs': jax.nn.log_softmax(logits), 'preds': jnp.argmax(logits, -1)}
p_pred_step = jax.pmap(pred_step, axis_name='batch', donate_argnums=(1,))

out = defaultdict(dict)
for ids, batch in tqdm(val_iter):
    val_pred = p_pred_step(state, batch)
    val_probs = np.exp(val_pred['logprobs'].reshape(-1, 2, 4).astype(np.float32))
    for (id_i, p_i) in zip(ids, val_probs):
        if id_i == 'pad':
            continue
        annot_id = '-'.join(id_i.split('-')[:2])
        conditionee = id_i.split('_')[-1]
        sub_dict = {f'answer_{i}': p_i[0,i] for i in range(4)}
        for i in range(4):
            sub_dict[f'rationale_conditioned_on_{conditionee}_{i}'] = p_i[1,i]
        out[annot_id].update(sub_dict)
out_df = pd.DataFrame.from_records(out).T
out_df = out_df.sort_index(key=lambda x: [int(z[1]) for z in out_df.index.str.split('-')])
out_df.index.name = 'annot_id'
assert not pd.isnull(out_df).any().any()
out_df.to_csv(cfg_name + '_submission.csv')
