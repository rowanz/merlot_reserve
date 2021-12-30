from mreserve.modeling import *
from jax import lax
import numpy as np
from flax.training import train_state
from mreserve.checkpoint import f32_to_bf16, bf16_to_f32


class MerlotReservePretrainer(MerlotReserve):
    def _augment_video_src_idx(self, video_src_idx, prng_key):
        """
        Randomly split `video_src_idx` into two portions. basically this means that now we won't have some segments attending
        to other segments. Could be good if we want to often handle short clips of videos
        :param video_src_idx: [B, L] e.g.
          DeviceArray([[1, 1, 1, 1, 1, 1, 2, 2],
                       [2, 2, 2, 2, 2, 2, 2, 2],
                       [1, 1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1, 1]], dtype=int32)
        :return: [B, L]
        """
        B, L = video_src_idx.shape
        if L == 1:
            print("_augment_video_src_idx: L=1 so cant split", flush=True)
            return video_src_idx

        split_prob = self.config.get('_augment_video_src_idx_prob', 0.1)
        probs = [split_prob / (L - 1) for i in range(L - 1)] + [1 - split_prob]
        print("Augmenting video_src_idx {}x{} with probs {}".format(B, L, probs), flush=True)
        split_from_here = 1 + jax.random.choice(prng_key, a=L, shape=[B], p=np.array(probs))

        split_mask = split_from_here[:, None] <= jnp.arange(L)[None]

        # Offset by a big number, say 4L
        video_src_idx = lax.select(split_mask, video_src_idx + 4 * L, video_src_idx)
        return video_src_idx

    def __call__(self, batch):
        """
        Does a forward pass for pretraining
        :param batch: Everything from pretraining
        :return:
        """
        num_segment_groups = self.data['num_segment_groups']
        num_audio_subsegments = self.data['num_audio_subsegments']
        lang_seq_len = self.data['lang_seq_len']
        seq_len = self.data['seq_len']

        batch_size, num_segments_nvpatch0, pp3 = batch['images'].shape
        nvpatch0 = self.output_grid[0] * self.output_grid[1]
        num_segments = num_segments_nvpatch0 // nvpatch0
        num_segments_per_group = num_segments // num_segment_groups

        # Images is size [batch_size * num_segments, num_patch, H]
        imgs_enc = self.vision_encoder(batch['images'].reshape((batch_size * num_segments, nvpatch0, pp3)))

        nvpatch1 = nvpatch0 // (self.vit_pooling_ratio ** 2)
        imgs_seq = imgs_enc['seq_attnpool'].reshape(
            [batch_size, num_segment_groups, num_segments_per_group * nvpatch1, self.hidden_size])

        if self.config.get('no_vision', False):
            print("\nNO VISION\n\n\n!!!!\n\n\n", flush=True)
            imgs_seq *= 0.0

        vis_seq_length = imgs_seq.shape[-2]

        # Audio clips are provided as [batch_size, num_segments * num_audio_subsegments * audio_seq_len, num_mels]
        audio_enc = self.audio_encoder(batch['audio_clips'].reshape(
            (batch_size * num_segments * num_audio_subsegments, self.audio_seq_length, -1)))

        # Audio seq is now [batch_size, num_audio_spans, seq_len, H]
        num_audio_spans = num_segments * num_audio_subsegments
        audio_seq = audio_enc['seq_attnpool'].reshape(
            [batch_size, num_audio_spans, self.audio_token_length, self.config['hidden_size']])
        audio_cls = audio_enc['cls'].reshape([batch_size, num_audio_spans, self.config['hidden_size']])

        # Flatten text sequence
        for k1 in ['text2audio', 'audio2text']:
            for k2 in ['', '/audio_ptr', '/text_ptr']:
                k = k1 + k2
                batch[k] = batch[k].reshape((-1, lang_seq_len))

        for k in ['random_text', 'random_text/text_ptr', 'audio_text_matching', 'audio_text_matching/audio_ptr']:
            batch[k] = batch[k].reshape((-1, seq_len))

        batch['text_spans'] = batch['text_spans'].reshape((-1, self.text_span_length))

        ##############################################

        txt_embs = self.token_encoder(
            {k: batch[k] for k in ['text2audio', 'audio2text', 'audio_text_matching', 'text_spans', 'random_text']})

        batch['video_src_index'] = batch['video_src_index'].reshape(-1, num_segments_per_group)

        mm_inputs = {}
        prng_0 = batch['audio2text/text_ptr'].astype(jnp.uint32).sum()[None].repeat(2)
        prngs = jax.random.split(prng_0, num=3)

        num_audio2text_seqs = self.data['num_audio2text_seqs']
        mm_inputs['audio2text'] = self.prepare_multimodal_inputs(
            tokens=batch['audio2text'],
            token_segment_idx=(batch['audio2text/audio_ptr'] // num_audio_subsegments) % num_segments_per_group,
            token_embs=txt_embs['audio2text'],
            vision_input=jnp.tile(imgs_seq, [1, num_audio2text_seqs, 1, 1]).reshape(-1, vis_seq_length,
                                                                                    self.hidden_size),
            audio_spans=audio_seq.repeat(num_segment_groups * num_audio2text_seqs, axis=0),
            audio_pointers=batch['audio2text/audio_ptr'],
            padding_len=seq_len,
            video_src_idx=self._augment_video_src_idx(jnp.tile(batch['video_src_index'].reshape(
                batch_size, num_segment_groups, num_segments_per_group), [1, num_audio2text_seqs, 1]).reshape(-1,
                                                                                                              num_segments_per_group),
                                                      prngs[0]),
        )

        mm_inputs['audio_text_matching'] = self.prepare_multimodal_inputs(
            tokens=batch['audio_text_matching'],
            token_segment_idx=jnp.cumsum((batch['audio_text_matching'] == LTOVPOOL).astype(jnp.int32), -1),
            token_embs=txt_embs['audio_text_matching'],
            audio_spans=audio_seq,
            audio_pointers=batch['audio_text_matching/audio_ptr'],
            padding_len=seq_len,
        )

        num_text2audio_seqs = self.data['num_text2audio_seqs']
        mm_inputs['text2audio'] = self.prepare_multimodal_inputs(
            tokens=batch['text2audio'],
            token_segment_idx=(batch['text2audio/audio_ptr'] // num_audio_subsegments) % num_segments_per_group,
            token_embs=txt_embs['text2audio'],
            vision_input=jnp.tile(imgs_seq, [1, num_text2audio_seqs, 1, 1]).reshape(-1, vis_seq_length,
                                                                                    self.hidden_size),
            audio_pointers=batch['text2audio/audio_ptr'],
            padding_len=seq_len,
            video_src_idx=self._augment_video_src_idx(jnp.tile(batch['video_src_index'].reshape(
                batch_size, num_segment_groups, num_segments_per_group), [1, num_text2audio_seqs, 1]).reshape(-1,
                                                                                                              num_segments_per_group),
                                                      prngs[1]),
        )
        mm_inputs['random_text'] = self.prepare_multimodal_inputs(tokens=batch['random_text'], padding_len=seq_len)

        keys = sorted(mm_inputs.keys())
        x = jnp.concatenate([mm_inputs[k]['x'] for k in keys], 0)
        coords = jnp.concatenate([mm_inputs[k]['rotary_coords'] for k in keys], 0)
        attnmask = jnp.concatenate([mm_inputs[k]['attention_mask'] for k in keys], 0)
        real_bsizes = [mm_inputs[k]['x'].shape[0] for k in keys]

        if not self.config.get('do_rotary', True):
            print("NOT DOING ROTARY", flush=True)
            coords = None

        joint_enc = self.joint_transformer(x, rotary_coords=coords, attention_mask=attnmask)['seq']
        joint_enc = self.joint_proj(joint_enc)
        mm_outputs = {k: z for k, z in zip(keys, jnp.split(joint_enc, np.cumsum(real_bsizes), axis=0))}

        mm_outputs['text2audio'] = mm_outputs['text2audio'][:, :lang_seq_len]
        mm_outputs['audio2text'] = mm_outputs['audio2text'][:, :lang_seq_len]

        ################################################
        # Get everything needed
        # Vision to Audio
        is_pool = (batch['audio_text_matching'] == LTOVPOOL)
        v2a_cumulative_idx = jnp.cumsum(is_pool.astype(jnp.int32), -1) - 1
        a2v = one_hot_pool(is_pool,
                           idx=v2a_cumulative_idx,
                           v=mm_outputs['audio_text_matching'],
                           num_segments=num_segments)['x'].reshape((batch_size * num_segments, self.hidden_size))
        ################################################

        # Text to audio
        ################################################
        t2a_sel = one_hot_pool(
            do_pool=batch['text2audio'] == MASKAUDIO,
            idx=batch['text2audio/audio_ptr'],
            v=mm_outputs['text2audio'],
            num_segments=num_segments * num_audio_subsegments,
            real_bsize=batch_size,
        )
        # For text to audio, not all the audio is a "target" so don't include in one part of the loss
        num_audio_spans_trg = int(num_audio_spans * self.data['mask_rate']) * num_text2audio_seqs
        is_selected = t2a_sel['idx_oh'].sum(1)

        idx_sort = jnp.argsort(-is_selected, -1)

        best_idxs = idx_sort[:, :num_audio_spans_trg].reshape(batch_size * num_audio_spans_trg)
        batch_indexer = jnp.arange(batch_size).repeat(num_audio_spans_trg)
        t2a_sel = t2a_sel['x'][batch_indexer, best_idxs]
        a2t_sel = audio_cls[batch_indexer, best_idxs]

        extra_idxs = idx_sort[:, num_audio_spans_trg:].reshape(batch_size * (num_audio_spans - num_audio_spans_trg))
        batch_indexer = jnp.arange(batch_size).repeat(num_audio_spans - num_audio_spans_trg)
        a2t_extra = audio_cls[batch_indexer, extra_idxs]
        ################################################

        # Predict Text spans
        ################################################
        num_text_spans = txt_embs['text_spans'].shape[0] // batch_size
        t2sp = {}
        for k in ['audio2text', 'text2audio', 'random_text']:
            t2sp[k] = one_hot_pool(
                batch[k] == MASK,
                idx=batch[f'{k}/text_ptr'],
                v=mm_outputs[k],
                num_segments=num_text_spans,
                real_bsize=batch_size
            )
            t2sp[k]['count'] = t2sp[k].pop('idx_oh').sum(1)
        t2sp_sel = t2sp['text2audio']['x'] + t2sp['audio2text']['x'] + t2sp['random_text']['x']
        t2sp_ct = t2sp['text2audio']['count'] + t2sp['audio2text']['count'] + t2sp['random_text']['count']
        t2sp_src = jnp.stack([jnp.zeros_like(t2sp['text2audio']['count']), t2sp['text2audio']['count'],
                              t2sp['audio2text']['count'], t2sp['random_text']['count']], -1).argmax(-1) - 1

        # Exclude things that have length 0, or that got dropped from the input somehow
        is_valid = (batch['text_spans'] != PADDING).any(-1).reshape(batch_size, num_text_spans)
        is_valid &= (t2sp_ct > 0.0)
        is_valid = is_valid.astype(jnp.float32)

        # Select `num_text_spans_to_include` spans which is less than the number of spans total.
        # Use the `random choice without replacement` hack
        # Choose multimodal spans 4x as often
        prefer_multimodal = np.log(4)
        logits_for_pred = is_valid * 1e6 + prefer_multimodal * (
                    t2sp['text2audio']['count'] + t2sp['audio2text']['count'])
        z = -jnp.log(-jnp.log(jax.random.uniform(key=prngs[2], shape=[batch_size, num_text_spans],
                                                 dtype=jnp.float32, minval=0.0, maxval=1.0)))
        is_valid = logits_for_pred + z

        NUM_TO_INCLUDE = self.data['num_text_spans_to_include']
        assert NUM_TO_INCLUDE <= num_text_spans
        print(f"Including {NUM_TO_INCLUDE} / {num_text_spans} text spans per batch (bsize={batch_size}."
              f" but doing it across batches! so some might have more. that's OK though I think", flush=True)
        best_idxs = lax.top_k(is_valid.reshape(-1), k=NUM_TO_INCLUDE * batch_size)[1]

        # Text input
        t2sp_sel = t2sp_sel.reshape([batch_size * num_text_spans, self.hidden_size])[best_idxs]
        t2sp_src = t2sp_src.reshape([batch_size * num_text_spans])[best_idxs]
        sp2t_sel = self.span_encoder(x=txt_embs['text_spans'][best_idxs],
                                     x_isvalid=batch['text_spans'][best_idxs] != PADDING)
        #################################################################

        log_scales = jnp.clip(self.scale_params.astype(jnp.float32), a_max=np.log(100.0))
        outputs = {
            'imgs_to_audio': {'x': a2v, 'y': imgs_enc['cls'], 'log_scale': log_scales[0]},
            'text_to_audio': {'x': t2a_sel, 'y': a2t_sel, 'y_extra': a2t_extra, 'log_scale': log_scales[1]},
            'stuff_to_span': {'x': t2sp_sel, 'y': sp2t_sel, 'log_scale': log_scales[2], '_sources': t2sp_src},
        }

        for k in outputs:
            temp_to_use = jnp.exp(outputs[k].pop('log_scale') / 2.0)
            for k2 in 'xy':
                outputs[k][k2] = unit_normalize(outputs[k][k2]) * temp_to_use
                if self.use_bfloat16:
                    outputs[k][k2] = outputs[k][k2].astype(jnp.bfloat16)

                k2_extra = f'{k2}_extra'
                if k2_extra in outputs[k]:
                    outputs[k][k2_extra] = unit_normalize(outputs[k][k2_extra]) * temp_to_use
                    if self.use_bfloat16:
                        outputs[k][k2_extra] = outputs[k][k2_extra].astype(jnp.bfloat16)

        return outputs


def loss_fn_given_preds(preds):
    loss_info = {}

    if 'text_preds' in preds:
        # Special-case of mask LM loss
        text_preds = preds.pop('text_preds')
        logits = text_preds['logits']
        labels = jax.nn.one_hot(text_preds['labels'], num_classes=logits.shape[1], dtype=logits.dtype)
        logprobs = jax.nn.log_softmax(logits, axis=-1)
        mask = (text_preds['labels'] != 0).astype(logits.dtype)

        loss_info['audio2text'] = -(jnp.sum(logprobs * labels, axis=-1) * mask).sum() / mask.sum()

    for c_type, c_dict in preds.items():
        numer_logits = (c_dict['x'] * c_dict['y']).sum(-1)
        loss_info[c_type] = 0.0

        if '_sources' in c_dict:
            for k in ['text2audio', 'audio2text', 'random_text']:
                loss_info[f'_{c_type}_from_{k}'] = 0.0
        # For both directions (average the result)
        for k1, k2 in ['xy', 'yx']:
            x = c_dict[k1]
            y = c_dict[k2]

            # Add in extra things that are only valid as targets
            if f'{k2}_extra' in c_dict:
                y = jnp.concatenate([y, c_dict[f'{k2}_extra']])
            y_allgather = jax.lax.all_gather(y, 'batch').reshape(-1, x.shape[-1])

            print(f"{c_type} {k1}->{k2} dot product sim:  shaped [{x.shape}] -> [{y_allgather.shape}", flush=True)
            denom_logits = jnp.einsum('lh,vh->lv', x, y_allgather)
            denom_lse = jax.nn.logsumexp(denom_logits.astype(jnp.float32), axis=-1)
            loss_info[c_type] += (denom_lse - numer_logits).mean() / 2.0
            if '_sources' in c_dict:
                for i, type_i in enumerate(['text2audio', 'audio2text', 'random_text']):
                    does_match = (c_dict['_sources'] == i).astype(jnp.float32)
                    loss_match = ((denom_lse - numer_logits) * does_match).sum() / (does_match.sum() + 1e-5)
                    loss_info[f'_{c_type}_from_{type_i}'] += loss_match / 2.0

    loss = sum([v for k, v in loss_info.items() if not k.startswith('_')])
    return loss, loss_info


def train_step(state: train_state.TrainState, batch, use_bfloat16_grads=True):
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

    params = state.params
    if use_bfloat16_grads:
        params = f32_to_bf16(state.params)

    (loss, loss_info), grads = grad_fn(params)

    grads = jax.tree_map(lambda x: jnp.nan_to_num(x, copy=False), grads)
    grads = jax.lax.pmean(grads, axis_name='batch')

    # Cast up grads here (after the pmean) which reduces bandwidth maybe
    if use_bfloat16_grads:
        grads = bf16_to_f32(grads)

    # Average metrics over all replicas (maybe this isn't a great idea, idk)
    loss_info = jax.lax.pmean(loss_info, axis_name='batch')
    loss_info = bf16_to_f32(loss_info)

    new_state = state.apply_gradients(grads=grads)
    return new_state, loss_info
