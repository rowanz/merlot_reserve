"""
Submit an existing TVQA model on the leaderboard

ipython -i submit_to_leaderboard.py -- ../../pretrain/configs/base.yaml ${ckpt}
ipython -i submit_to_leaderboard.py -- ../../pretrain/configs/large.yaml ${ckpt}
"""

import sys

sys.path.append('../../')
import yaml
from datetime import datetime
import pytz
import jax
import jax.numpy as jnp
from pretrain.dataloader import input_fn_builder, MASK, encoder, AUDIOSPAN
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
import optax
from tqdm import tqdm
import json

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
args = parser.parse_args()

# print(f"Loading from {args.config_file}", flush=True)
with open(args.pretrain_config_file, 'r') as f:
    config = yaml.load(f, yaml.FullLoader)

config['data']['val_fns'] = "${path_to_tvqa}/test{:03d}of008.tfrecord"
config['data']['num_val_files'] = 8
config['data']['num_answers'] = 5
config['data']['do_random_scale'] = False

config['data']['num_segments'] = 7

config['device']['batch_size'] = 8
config['device']['prefetch_size'] = 0
config['device']['n_fns_per_cycle'] = 256

TRAIN_SIZE = 122112

config['data']['lang_seq_len'] = 256
cfg_name = args.pretrain_config_file.split('/')[-1]
seattle_time = pytz.utc.localize(datetime.utcnow()).astimezone(pytz.timezone('America/Los_Angeles'))
seattle_time = seattle_time.strftime("%Y-%m-%d-%H:%M.%S")

np.random.seed(123456)
config['model']['output_grid'] = [18, 32]


config['_ckpt'] = args.ckpt

class MerlotReserveTVQA(MerlotReserve):
    def setup(self):
        super().setup()
        self.proj = nn.Dense(features=1, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(stddev=0.02), name='proj',
                             use_bias=False)

    def __call__(self, batch):

        # Encode images (twice)
        batch_size, images_per_batch, seq_size, img_dim = batch['images'].shape
        imgs_enc = self.vision_encoder(batch['images'].reshape(batch_size * images_per_batch, seq_size, img_dim))['seq_attnpool']
        imgs_enc = imgs_enc.reshape(batch_size, images_per_batch, seq_size // 4, self.hidden_size)

        # Add the "first image"
        imgs_enc = jnp.concatenate([
            jnp.zeros([batch_size, 1, seq_size // 4, self.hidden_size], dtype=imgs_enc.dtype),
            imgs_enc,
        ], 1)

        # duplicate so that we have one per answer
        images_per_batch += 1
        batch_size, num_ans_per, joint_seq_len, two_ = batch['textonly_seqs'].shape
        imgs_enc = imgs_enc.reshape(batch_size, images_per_batch * seq_size // 4, self.hidden_size).repeat(num_ans_per, axis=0)

        #########################
        text_toks = batch['textonly_seqs'][..., 0].reshape(batch_size * num_ans_per, joint_seq_len)
        textonly_inputs = self.prepare_multimodal_inputs(
            tokens=text_toks,
            token_segment_idx=batch['textonly_seqs'][..., 1].reshape(batch_size * num_ans_per, joint_seq_len),
            vision_input=imgs_enc,
        )

        # Encode audio
        # Audio clips are provided as [batch_size, num_segments, num_audio_subsegments, audio_seq_len, num_mels]
        batch_size, num_segments, num_audio_subsegments, audio_seq_len, num_mels = batch['audio_clips'].shape
        audio_enc = self.audio_encoder(batch['audio_clips'].reshape(-1, audio_seq_len, num_mels))['seq_attnpool']

        _, audio_token_len, hidden_size = audio_enc.shape
        num_audio_spans = num_segments * num_audio_subsegments

        audio_enc = audio_enc.reshape(batch_size, num_audio_spans, audio_token_len, hidden_size)
        audio_enc = audio_enc.repeat(num_ans_per, axis=0)

        audio_toks = batch['audio_seqs'][..., 0].reshape(batch_size * num_ans_per, joint_seq_len)
        audio_pointers = (jnp.cumsum((audio_toks == AUDIOSPAN).astype(jnp.int32), -1) - 1) // audio_token_len
        audio_pointers = audio_pointers % num_audio_spans

        audio_inputs = self.prepare_multimodal_inputs(
            tokens=batch['audio_seqs'][..., 0].reshape(batch_size * num_ans_per, joint_seq_len),
            token_segment_idx=batch['audio_seqs'][..., 1].reshape(batch_size * num_ans_per, joint_seq_len),
            vision_input=imgs_enc,
            audio_spans=audio_enc,
            audio_pointers=audio_pointers,
        )
        # hack: remove 'first img' from sequence lengths
        start_imgs = joint_seq_len + seq_size // 4
        for k in ['x', 'rotary_coords', 'attention_mask']:
            textonly_inputs[k] = jnp.concatenate([textonly_inputs[k][:, :joint_seq_len],
                                                  textonly_inputs[k][:, start_imgs:]], 1)

            audio_inputs[k] = jnp.concatenate([audio_inputs[k][:, :joint_seq_len],
                                               audio_inputs[k][:, start_imgs:]], 1)

        textonly_inputs['attention_mask'] = jnp.concatenate([textonly_inputs['attention_mask'][:, :, :joint_seq_len],
                                                             textonly_inputs['attention_mask'][:, :, start_imgs:]], 2)

        audio_inputs['attention_mask'] = jnp.concatenate([audio_inputs['attention_mask'][:, :, :joint_seq_len],
                                                          audio_inputs['attention_mask'][:, :, start_imgs:]], 2)
        #############################################################################################################
        x = jnp.concatenate([audio_inputs['x'], textonly_inputs['x']], 0)
        coords = jnp.concatenate([audio_inputs['rotary_coords'], textonly_inputs['rotary_coords']], 0)
        attnmask = jnp.concatenate([audio_inputs['attention_mask'], textonly_inputs['attention_mask']], 0)

        joint_enc = self.joint_transformer(x, rotary_coords=coords, attention_mask=attnmask)['seq']
        joint_enc = joint_enc[:, :joint_seq_len].reshape(batch_size * 2 * num_ans_per, joint_seq_len, self.hidden_size)

        # Pool from the right tokens
        pool_idx = jnp.argmax((jnp.concatenate([audio_toks, text_toks], 0) == MASK).astype(jnp.float32), 1)
        pooled_h = joint_enc[jnp.arange(batch_size * 2 * num_ans_per), pool_idx]
        joint_enc = jnp.squeeze(self.proj(pooled_h), -1)

        logits_from_audio, logits_from_text = jnp.split(joint_enc, 2, axis=0)
        logits_from_audio = logits_from_audio.reshape(batch_size, num_ans_per)
        logits_from_text = logits_from_text.reshape(batch_size, num_ans_per)

        return logits_from_audio, logits_from_text


model = MerlotReserveTVQA.from_config(config)

params = freeze(load_checkpoint(args.ckpt))['params']
params = f32_to_bf16(params)
state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optax.identity())
state = jax_utils.replicate(state)

def pred_step(state: train_state.TrainState, batch):
    logits_from_audio, logits_from_text = state.apply_fn({'params': state.params}, batch)

    out = {'logprobs_audio': jax.nn.log_softmax(logits_from_audio, axis=-1),
            'preds_audio': jnp.argmax(logits_from_audio, -1),
            'logprobs_text': jax.nn.log_softmax(logits_from_text, axis=-1),
            'preds_text': jnp.argmax(logits_from_text, -1),
            }
    softmax_joint = jax.nn.softmax(logits_from_audio, axis=-1) + jax.nn.softmax(logits_from_text, axis=-1)
    out['preds_joint'] = jnp.argmax(softmax_joint, -1)
    return out
p_pred_step = jax.pmap(pred_step, axis_name='batch', donate_argnums=(1,))

out = {}
for split in ['val', 'test']:
    config['data']['val_fns'] = f"path_to_tvqa/{split}" + '{:03d}of008.tfrecord'
    val_iter = finetune_val_input_fn_builder(config, 'tvqa')

    for ids, batch in tqdm(val_iter):
        val_pred = p_pred_step(state, batch)
        preds_joint = val_pred['preds_joint'].reshape(-1).tolist()
        preds_audio = val_pred['preds_audio'].reshape(-1).tolist()
        preds_text = val_pred['preds_text'].reshape(-1).tolist()
        for (id_i, p_j, p_a, p_t) in zip(ids, preds_joint, preds_audio, preds_text):
            if id_i == 'pad':
                continue
            id_i = id_i.split('~')[0]
            out[(split, id_i)] = (p_t, p_a, p_j)

for sub_i, submission in enumerate(['text', 'audio', 'joint']):
    os.makedirs(submission, exist_ok=True)

    # Make prediction_val.json
    pred_dict = {id_idx: v[sub_i] for (split, id_idx), v in out.items() if split == 'val'}
    with open(os.path.join(submission, 'prediction_val.json'), 'w') as f:
        json.dump(pred_dict, f)

    # Make prediction_val.json
    pred_dict = {id_idx: v[sub_i] for (split, id_idx), v in out.items() if split == 'test'}
    with open(os.path.join(submission, 'prediction_test_public.json'), 'w') as f:
        json.dump(pred_dict, f)

    model_size = 'Base' if 'base' in args.pretrain_config_file.lower() else 'Large'
    model_suffix = {'text': '(subtitles)', 'audio': '(audio)', 'joint': '(subtitles and audio)'}[submission]
    meta = {'model_name': f'MerlotReserve-{model_size} {model_suffix}',
            'is_ensemble': False,
            'with_ts': True,
            'show_on_leaderboard': True,
            'author': 'Anonymous',
            'institution': 'Anonymous',
            'description': 'A {}-sized model, given {} at test time'.format(model_size, model_suffix.strip('()')),
            'paper_link': '', 'code_link': ''}
    with open(os.path.join(submission, 'meta.json'), 'w') as f:
        json.dump(meta, f)

    os.system(f'cd {submission} && zip ../{submission}.zip prediction_val.json prediction_test_public.json meta.json')