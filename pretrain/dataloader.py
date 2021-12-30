"""
Pretraining dataloader
"""
import sys

sys.path.append('../')
import time
from pretrain.data_utils import resize_and_pad, get_shape_list, pad_to_fixed_size, \
    uniform_random_select, random_categorical_without_replacement, sample_bernoulli, batch_index_iterator, \
    sample_bernoullis, cumulative_maximum_int, encode_string
from mreserve.lowercase_encoder import get_encoder, START, END, PADDING, MASK, AUDIOSPAN, LTOVPOOL, MASKAUDIO
import math
import tensorflow as tf
import regex as re
import numpy as np
import tensorflow_datasets as tfds
import functools
from copy import deepcopy
import random
from collections import defaultdict
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    tf.config.experimental.set_visible_devices([], 'GPU')

logger = tf.get_logger()
encoder = get_encoder()
###################################
segment_k2f = {
    'image/encoded': tf.io.FixedLenFeature((), tf.string, default_value=''),
    'image/format': tf.io.FixedLenFeature((), tf.string, default_value='jpeg'),
    'image/key/sha256': tf.io.FixedLenFeature((), tf.string, default_value=''),
    'image/height': tf.io.FixedLenFeature((), tf.int64, 1),
    'image/width': tf.io.FixedLenFeature((), tf.int64, 1),

    'spectrogram/encoded': tf.io.FixedLenFeature((), tf.string, default_value=''),
    'spectrogram/format': tf.io.FixedLenFeature((), tf.string, default_value='jpeg'),
    'spectrogram/key/sha256': tf.io.FixedLenFeature((), tf.string, default_value=''),
    'spectrogram/height': tf.io.FixedLenFeature((), tf.int64, 1),
    'spectrogram/width': tf.io.FixedLenFeature((), tf.int64, 1),
    'spectrogram/magic_number': tf.io.FixedLenFeature((), tf.float32, 1),

    'youtube_id': tf.io.FixedLenFeature((), tf.string, default_value=''),
    'video_src_index': tf.io.FixedLenFeature((), tf.int64, 1),

    'title': tf.io.VarLenFeature(tf.int64),
    'tags': tf.io.VarLenFeature(tf.int64),
    'description': tf.io.VarLenFeature(tf.int64),
    'meta': tf.io.FixedLenFeature((), tf.string, default_value=''),

    'playback_speed': tf.io.VarLenFeature(tf.int64),
    'start_time': tf.io.FixedLenFeature((), tf.float32, 1),
    'end_time': tf.io.FixedLenFeature((), tf.float32, 1),

    'tok_ids': tf.io.VarLenFeature(tf.int64),
    'tok_start_times': tf.io.VarLenFeature(tf.float32),
    'tok_end_times': tf.io.VarLenFeature(tf.float32),
    'random_text': tf.io.VarLenFeature(tf.int64),
}


def load_and_resize_img(encoded_jpg, config):
    """
    Encoded JPG -> image patches
    :param encoded_jpg: string
    :return: [(H // P) * (W // P), P * P * 3] image
    """
    P = config['vit_patch_size']
    h1, w1 = config['output_grid']

    img = tf.image.decode_jpeg(encoded_jpg, channels=3)
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)

    img, this_image_info = resize_and_pad(img, (h1 * P, w1 * P),
                                          do_random_scale=config.get('do_random_scale', True),
                                          random_scale_max=config.get('random_scale_max', 1.1),
                                          random_scale_min=config.get('random_scale_min', 1.05),
                                          shrink_both_sides=config.get('shrink_both_sides', True),
                                          do_flip_if_vertical=config.get('do_flip_if_vertical', True),
                                          resize_method='random')

    img = tf.nn.space_to_depth(img[None], P, data_format='NHWC')
    img = tf.reshape(img, [h1 * w1, P * P * 3])
    return img


def load_audio(x, config):
    """
    :param x: A tuple with a string (of the encoded audio) and a magic number for inverse scaling
    :return: [num_audio_subsegments, audio_seq_length, num_mels],
    """
    encoded_audio, magic_number, playback_speed = x
    img = tf.image.decode_jpeg(encoded_audio, channels=1)
    img = tf.squeeze(img, 2)
    img.set_shape([config['num_mels'], config['spec_size']])
    img = tf.transpose(img)

    # Extract N sequences
    content_len = config['num_audio_subsegments'] * config['audio_seq_length']
    assert content_len < config['spec_size']
    paddings = tf.random.uniform([config['num_audio_subsegments'] + 1], minval=0, maxval=1.0)
    num_pad = config['spec_size'] - content_len
    paddings_int = tf.cast(num_pad * tf.cumsum(paddings / tf.reduce_sum(paddings)), dtype=tf.int32)
    start_idx = paddings_int[:config['num_audio_subsegments']] + tf.range(config['num_audio_subsegments']) * config[
        'audio_seq_length']
    audio_seqs = []
    for i in range(config['num_audio_subsegments']):
        audio_seqs.append(img[start_idx[i]:(start_idx[i] + config['audio_seq_length'])])

    audio_seqs = tf.stack(audio_seqs)
    audio_seqs = tf.cast(audio_seqs, dtype=tf.float32) / magic_number  # Don't use convert_image_dtype as that scales it
    # If we wanted to invert it completely
    # mel = tf.exp(log_mel) - config['spec_eps']

    # Add in the playback speed as an extra feature
    audio_seqs.set_shape([config['num_audio_subsegments'], config['audio_seq_length'], config['num_mels']])

    playback_speed_f32 = tf.cast(playback_speed, dtype=tf.float32)
    audio_seqs = tf.concat(
        [audio_seqs, tf.fill([config['num_audio_subsegments'], config['audio_seq_length'], 1], playback_speed_f32)], -1)

    fft_window = config['fft_window_size'] / config['sample_rate']
    fft_to_time_scale = config['fft_hop_length'] / config['sample_rate']
    audio_start_t = tf.cast(start_idx, dtype=tf.float32) * fft_to_time_scale - fft_window / 2.0
    audio_end_t = audio_start_t + config['audio_seq_length'] * fft_to_time_scale + fft_window

    return audio_seqs, audio_start_t, audio_end_t


def pad_tokens_to_fixed_size(tokens, padded_seq_len):
    """
    Pad, then truncate tokens to padded_seq_len
    :param tokens:
    :param padded_seq_len:
    :return:
    """
    missing_len = tf.maximum(padded_seq_len - get_shape_list(tokens, 2)[0], 0)
    dummy_row = tf.constant([0, -1, -1], dtype=tf.int32)
    tokens = tf.concat([tokens, tf.tile(dummy_row[None], [missing_len, 1])], 0)[:padded_seq_len]
    tokens = tf.reshape(tokens, [padded_seq_len, 3])
    return tokens


def _one_hot(idx, N):
    m = get_shape_list(idx, 1)[0]
    return tf.reduce_any(tf.equal(tf.range(N)[:, None], idx[None]), 1)


def select_tokens(tokens, padded_seq_len, num_segments):
    """
    Take out stuff from `tokens` without killing mask tokens

    We can cut out `audiospan' tokens on the RHS, not the left
    :param tokens:
    :param padded_seq_len:
    :return:
    """
    L = get_shape_list(tokens, 2)[0]
    amt_to_truncate = L - padded_seq_len

    is_mask = tf.cumsum(tf.cast((tokens[:, 0] == MASK) | (tokens[:, 0] == MASKAUDIO), dtype=tf.int32))
    is_audiospan = tf.cumsum(tf.cast(tokens[:, 0] == AUDIOSPAN, dtype=tf.int32))

    lhs_amt = tf.reduce_sum(tf.cast((is_mask == 0) & (is_audiospan == 0), dtype=tf.int32))
    rhs_amt = tf.reduce_sum(tf.cast(is_mask == is_mask[-1], dtype=tf.int32)) - 1

    # Truncate from both sides
    trunc_start = tf.minimum(amt_to_truncate // 2, lhs_amt)

    trunc_end = tf.minimum(amt_to_truncate - trunc_start, rhs_amt)
    trunc_start = tf.minimum(amt_to_truncate - trunc_end, lhs_amt)

    tokens0 = tokens[trunc_start:(L-trunc_end)]

    # We might need to more aggressively sample
    keep_logits = 1e7 * tf.cast(tf.equal(tokens0[:, 0], MASK) & (tokens0[:, 0] != AUDIOSPAN), dtype=tf.float32)

    # Try to keep the same segments together
    segment_to_score = tf.random.uniform(shape=[num_segments], minval=-1e5, maxval=1e5, dtype=tf.float32)
    keep_logits += tf.gather(segment_to_score, tokens0[:, 1])
    idx2 = tf.sort(random_categorical_without_replacement(keep_logits, padded_seq_len))
    tokens1 = tf.gather(tokens0, idx2)

    return tf.cond(
        get_shape_list(tokens0, 2)[0] > padded_seq_len,
        lambda: tokens1,
        lambda: tokens0,
    )


def mask_tokens(tokens_ragged, mask_idx, do_audio_span=None, audio_token_length=6, text_span_start_counter=0,
                num_groups=1, padded_seq_len=None, do_audio_mask=False):
    """
    Masks tokens in a ragged representation.
    :param tokens_ragged: a [N, L] representation of tokens
                          you can do this conversion by e.g.
                          tokens_ragged = tf.RaggedTensor.from_value_rowids(tokens, segment_idx)

    :param mask_idx_onehot: a [N] vector for the rows we should mask
    :param do_audio_span: Optional: a [N] vector for whether to turn the row into an audio span.
    :param text_span_start_counter: An integer >= 0. basically if we have other text spans from a different masker,
                                    this means we won't cross-over into them
    :param num_groups: How many partitions to split the tokens into
    :param padded_seq_len: Length to pad things to
    :param do_audio_mask: Whether to mask audio

    :return: A [sum(do_mask), L] representation of text spans, in ragged form
             A list of tokens of size [<=L, 3]. The columns are [token_id, audio_span, text_span]
    """
    N = tokens_ragged.bounding_shape()[0]
    mask_idx = tf.sort(mask_idx, 0)
    text_spans = tf.gather(tokens_ragged, mask_idx)
    mask_idx_onehot = _one_hot(mask_idx, N)

    if do_audio_span is not None:
        do_audio_span = tf.logical_and(do_audio_span, tf.logical_not(mask_idx_onehot))

        audio_span_full = tf.fill([N, audio_token_length], AUDIOSPAN)

        tokens_ragged = tf.compat.v1.where(do_audio_span, audio_span_full, tokens_ragged)

    # Replace with mask token
    mask_tok = tf.fill([N, 1], MASK)
    if do_audio_mask:
        mask_tok = tf.concat([mask_tok, tf.fill([N, 1], MASKAUDIO)], 1)

    tokens_ragged = tf.compat.v1.where(mask_idx_onehot, mask_tok, tokens_ragged)

    # Replace each token with the corresponding index into text_spans if it's a MASK token, else -1
    text_ptr = tf.cumsum(tf.cast(mask_idx_onehot, dtype=tf.int32)) - 1 + text_span_start_counter
    text_ptr = tf.where(mask_idx_onehot, text_ptr, tf.fill([N], -1))

    # split into groups
    grp_size = N // num_groups

    output_grouped = []
    for i in range(num_groups):
        tokens_ragged_i = tokens_ragged[i * grp_size:(i + 1) * grp_size]
        idxs_i = tf.cast(tf.where(tokens_ragged_i)[:, 0], dtype=tf.int32)

        audio_ptr_i = idxs_i + i * grp_size

        # text pointer -- grab all tokens if it's a mask token
        text_ptr_i = text_ptr[i * grp_size:(i + 1) * grp_size]
        text_ptr_i = tf.gather(text_ptr_i, idxs_i)

        # Do the formatting thing with the audio pointers and text pointers (for MASK tokens)
        output_i = tf.stack([tokens_ragged_i.values, audio_ptr_i, text_ptr_i], -1)
        if padded_seq_len is not None:

            is_over_budget = get_shape_list(output_i, 2)[0] > padded_seq_len
            output_i = tf.cond(is_over_budget,
                               lambda: select_tokens(output_i, padded_seq_len, num_segments=N),
                               lambda: pad_tokens_to_fixed_size(output_i, padded_seq_len))
        output_grouped.append(output_i)
    return text_spans, output_grouped


def shift_ragged_tokens_at_positions(tokens_ragged, positions, right_to_left=True):
    """
    Given a ragged tensor of size [N, L] and an index of positions, we shift those values one to the left, or one to the right
    :param tokens_ragged:
    :param positions:
    :return:
    """
    N = tokens_ragged.bounding_shape()[0]
    positions = tf.cast(positions, dtype=tf.int32)
    pos_onehot = _one_hot(positions, N)
    pos_onehot = tf.logical_and(pos_onehot, tf.greater(tokens_ragged.row_lengths(), 0))

    amt_to_take = tf.cast(pos_onehot, dtype=tf.int32)

    if right_to_left:
        amt_to_take = amt_to_take[1:]
        sub1 = tf.concat([[0], -amt_to_take], 0)
        add1 = tf.concat([amt_to_take, [0]], 0)
    else:
        amt_to_take = amt_to_take[:-1]
        sub1 = tf.concat([-amt_to_take, [0]], 0)
        add1 = tf.concat([[0], amt_to_take], 0)
    row_lengths = tokens_ragged.row_lengths() + sub1 + add1
    return tf.RaggedTensor.from_row_lengths(tokens_ragged.values, row_lengths)


def random_do_both_directions(f):
    # Decorator to do right than left, then left than right, or the other way around
    def _f(x, **kwargs):
        x_rtl0 = f(x, **kwargs, right_to_left=True)
        x_rtl1 = f(x_rtl0, **kwargs, right_to_left=False)

        x_ltr0 = f(x, **kwargs, right_to_left=False)
        x_ltr1 = f(x_ltr0, **kwargs, right_to_left=True)
        return tf.cond(sample_bernoulli(0.5), lambda: x_rtl1, lambda: x_ltr1)
    return _f


@random_do_both_directions
def reassign_empty_tokens(tokens_ragged, *, mask_idx, right_to_left: bool=True):
    """
    If there's something that's empty (and masked), steal one of the tokens

    :param tokens_ragged: Ragged Tensor of timesteps, [N rows, L]
    :param mask_idx: Index into length L, whether we mask that.
    :param right_to_left: Direction
    :return:
    """
    # 1. Reassign empty tokens
    N = tokens_ragged.bounding_shape()[0]
    mask_idx_onehot = _one_hot(mask_idx, N)

    row_lengths = tokens_ragged.row_lengths()
    needs_tokens = tf.logical_and(mask_idx_onehot, tf.equal(row_lengths, 0))
    can_donate = tf.logical_and(tf.logical_not(mask_idx_onehot), tf.greater_equal(row_lengths, 2))

    if right_to_left:
        positions = tf.where(tf.logical_and(can_donate[1:], needs_tokens[:-1]))[:, 0] + 1
        return shift_ragged_tokens_at_positions(tokens_ragged, positions)
    else:
        positions = tf.where(tf.logical_and(can_donate[:-1], needs_tokens[1:]))[:, 0]
        return shift_ragged_tokens_at_positions(tokens_ragged, positions, right_to_left=False)


@random_do_both_directions
def increase_textmask(tokens_ragged, *, mask_idx, tok_centroids_vals, audio_start_end, right_to_left: bool=True,
                      delta_thresh=0.1):
    """
    Increase text mask by 1 in places
    :param tokens_ragged:
    :param mask_idx:
    :param tok_centroids_vals: Values that go into a ragged tensor
    :param audio_start_end: [N, 2] coords.
    :param right_to_left: Direction
    :param delta_thresh: Threshold for assigning
    :return:
    """
    nrows_real = tokens_ragged.bounding_shape(axis=0)
    tok_centroids_expanded = tf.RaggedTensor.from_value_rowids(tok_centroids_vals, tokens_ragged.value_rowids() + 1,
                                                               nrows=nrows_real + 2, name='increase_textmask')

    # Don't let us increase at the expense of empty tokens
    nmask = get_shape_list(mask_idx, 1)[0]

    if right_to_left:
        # Move from the mini segment to our right, to us
        t_out_right = tf.reduce_min(tf.gather(tok_centroids_expanded, mask_idx + 2), -1)

        # only things at least length 1
        t_out_right = tf.where(tf.less_equal(tf.gather(tok_centroids_expanded.row_lengths(), mask_idx + 2), 1),
                               tf.fill([nmask], 10000.0), t_out_right)

        audio_boundary_r = tf.gather(audio_start_end[:, 1], mask_idx)

        delta_r = (t_out_right - audio_boundary_r)

        take_from_right = tf.less(delta_r, delta_thresh)
        right_is_masked = tf.reduce_any(tf.equal(mask_idx[:, None] + 1, mask_idx[None]), -1)
        take_from_right = tf.logical_and(take_from_right, tf.logical_not(right_is_masked))
        take_from_right = tf.logical_and(take_from_right, tf.less(mask_idx + 1, nrows_real))

        take_from_right_idx = tf.gather(mask_idx + 1, tf.where(take_from_right)[:, 0])

        return shift_ragged_tokens_at_positions(tokens_ragged, take_from_right_idx, right_to_left=True)
    else:
        t_out_left = tf.reduce_max(tf.gather(tok_centroids_expanded, mask_idx), -1)

        t_out_left = tf.where(tf.less_equal(tf.gather(tok_centroids_expanded.row_lengths(), mask_idx), 1),
                              tf.fill([nmask], -10000.0), t_out_left)

        audio_boundary_l = tf.gather(audio_start_end[:, 0], mask_idx)

        delta_l = (audio_boundary_l - t_out_left)
        take_from_left = tf.less(delta_l, delta_thresh)
        left_is_masked = tf.reduce_any(tf.equal(mask_idx[:, None] - 1, mask_idx[None]), -1)
        take_from_left = tf.logical_and(take_from_left, tf.logical_not(left_is_masked))
        take_from_left = tf.logical_and(take_from_left, tf.greater(mask_idx, 0))

        take_from_left_idx = tf.gather(mask_idx - 1, tf.where(take_from_left)[:, 0])

        return shift_ragged_tokens_at_positions(tokens_ragged, take_from_left_idx, right_to_left=False)

# is_valid = re.compile(r"^[ A-Za-z0-9\-$%&'+,./:?@\[\]_’]*$")
is_valid = re.compile(r"^[ A-Za-z0-9']*$")
TOKEN_IS_VALID = [(i > 10) and bool(is_valid.match(encoder.decode([i]))) for i in range(encoder.get_vocab_size())]
bad_tokens = [149, 4858, 9504, 15162, 22312, 22433, 32156]
for i in bad_tokens:
    TOKEN_IS_VALID[i] = False

def filter_out_tokens_not_in_youtube(spans_i, token_is_valid_tf=None):
    if token_is_valid_tf is None:
        token_is_valid_tf = tf.constant(TOKEN_IS_VALID, dtype=tf.bool)
    # Filter out tokens not seen in YouTube
    new_span_idx = tf.where(tf.gather(token_is_valid_tf, spans_i.values))[:, 0]
    spans_i = tf.RaggedTensor.from_value_rowids(tf.gather(spans_i.values, new_span_idx),
                                                tf.gather(spans_i.value_rowids(), new_span_idx),
                                                nrows=spans_i.bounding_shape(axis=0))
    return spans_i


def convert_rawtext_into_fake_segments(tokens, desired_len, span_budget, use_v1_stats=False):
    """
    :param tokens: Tokens that we will mask. I'm only going to mask alphanumeric characters
    :param desired_len: desired length of the tokens
    :param mask_rate: How much to mask
    :return A ragged list of tokens
    """
    # # I got this empirically to minimize KL divergence between lengths of this and audio-to-text and text-to-audio
    if use_v1_stats:
        logger.info("rawtext stats v1 -- should be for yttemporal 180m")
        weights = [0.0210583 , 0.03984984, 0.06506665, 0.09467365, 0.12138153,
           0.13305461, 0.12973022, 0.11296043, 0.09024, 0.06730134,
           0.04789645, 0.03232633, 0.02123288, 0.01397406, 0.00925371]
    else:
        logger.info("rawtext stats v2 -- should be for ytmega")
        weights = [0.03233136, 0.05236081, 0.08763368, 0.11757072, 0.13737426,
           0.13717706, 0.12541218, 0.10262764, 0.0771088 , 0.05364242,
           0.0342899 , 0.0203823 , 0.01177542, 0.00664939, 0.00366406]

    ev = sum(i * w_i for i, w_i in enumerate(weights)) + 1
    logger.info("mask weights ev={:.3f}, weights={}".format(ev, weights))
    # k masked tokens that cover an expected length of k * e
    # L - k non masked tokens
    # mask rate is then ek/(L-k+ek)
    # some algebra and then
    #####################

    # I'm going to be conservative here bc I don't want to have too many tokens
    L = desired_len + int((ev * 0.85 - 1) * span_budget)
    L = tf.minimum(L, get_shape_list(tokens, 1)[0])

    segm_lens = tf.squeeze(tf.random.categorical(tf.math.log([weights]), dtype=tf.int32, num_samples=L), 0) + 1

    # Truncate to the suggested length
    segm_lens_keep = tf.less_equal(tf.cumsum(segm_lens), L)
    segm_lens = tf.gather(segm_lens, tf.where(segm_lens_keep)[:, 0])

    # Randomly truncate tokens if it's really long
    l_sel = tf.reduce_sum(segm_lens)
    wiggle_room = get_shape_list(tokens, 1)[0] - l_sel
    random_offset = tf.random.uniform(shape=[], minval=0, maxval=tf.maximum(wiggle_room, 1), dtype=tf.int32)

    tokens_ragged = tf.RaggedTensor.from_row_lengths(tokens[random_offset:(random_offset + l_sel)], segm_lens)

    extra_lhs = tokens[:random_offset]
    extra_rhs = tokens[(random_offset+l_sel):]
    return tokens_ragged, extra_lhs, extra_rhs


def dataset_parser(record, config):
    """
    We are going to return the following things:

    * Images: [num_segments, H, W, 3]
    * audio: [num_segments, num_audio_spans, T, num_mels]

    :param record:
    :return:
    """
    num_segments = config['num_segments']

    keys_to_features = {f'c{i:02d}/{k}': v for i in range(num_segments) for k, v in segment_k2f.items()}
    parsed_features = tf.io.parse_single_example(record, keys_to_features)

    features = {}

    def _unsparsify(x):
        if isinstance(x, tf.SparseTensor):
            x = x.values
        if x.dtype == tf.int64:
            x = tf.cast(x, dtype=tf.int32)
        return x

    segment_list = [{k: _unsparsify(parsed_features.pop(f'c{i:02d}/{k}')) for k in segment_k2f} for i in
                    range(num_segments)]

    # Load images
    encodeds = tf.stack([x['image/encoded'] for x in segment_list])
    features['images'] = tf.map_fn(functools.partial(load_and_resize_img, config=config),
                                   elems=encodeds, fn_output_signature=tf.float32)
    if config.get('disable_imgs_dataloader', False):
        print("Disabling images from the dataloader level!!!", flush=True)
        features['images'] *= 0.0

    magic_numbers = tf.stack([x['spectrogram/magic_number'] for x in segment_list])
    encodeds = tf.stack([x['spectrogram/encoded'] for x in segment_list])
    playback_speeds = tf.squeeze(tf.stack([x['playback_speed'] for x in segment_list], 0), 1)
    features['audio_clips'], audio_start, audio_end = tf.map_fn(
        functools.partial(load_audio, config=config),
        elems=(encodeds, magic_numbers, playback_speeds),
        fn_output_signature=(tf.float32, tf.float32, tf.float32),
    )
    if config.get('disable_audio_dataloader', False):
        print("Disabling audio from the dataloader level!!!", flush=True)
        features['audio_clips'] *= 0.0

    ######################################################

    num_audio_spans = num_segments * config['num_audio_subsegments']
    num_audio_spans_trg = int(num_audio_spans * config['mask_rate'])
    num_text2audio_seqs = config['num_text2audio_seqs']
    num_audio2text_seqs = config['num_audio2text_seqs']

    segment_idx = []
    tok_centroids_all = []
    audio_start_end_all = []
    t_start = 0.0
    for i, segment_i in enumerate(segment_list):
        # Partition the tokens into the audio segments

        tok_centroids = (segment_i['tok_start_times'] + segment_i['tok_end_times']) / 2.0
        audio_centroids = (audio_start[i] + audio_end[i]) / 2.0
        tok_to_audio = tf.abs(tok_centroids[:, None] - audio_centroids[None])

        assignment = tf.cast(tf.argmin(tok_to_audio, 1), dtype=tf.int32)
        # Constrain to be non-negative (usually things are OK but ocasionally weird stuff happens)
        assignment = cumulative_maximum_int(assignment)

        segment_idx.append(assignment + i * config['num_audio_subsegments'])

        # Keep track of timesteps -- this is in case mulitple things are in the batch
        tok_centroids_all.append(tok_centroids + t_start)
        audio_start_end_all.append(tf.stack([audio_start[i], audio_end[i]], -1) + t_start)

        t_start += (segment_i['end_time'] - segment_i['start_time'])

    segment_idx = tf.concat(segment_idx, 0)

    tokens_ragged = tf.RaggedTensor.from_value_rowids(tf.concat([x['tok_ids'] for x in segment_list], 0),
                                                      segment_idx, nrows=num_audio_spans, name='ragged0')
    tok_centroids_vals = tf.concat(tok_centroids_all, 0)
    audio_start_end = tf.concat(audio_start_end_all, 0)

    # Use different segments for the targets
    audio_spans_trg_idx = uniform_random_select(n=num_audio_spans, num_samples=num_audio_spans_trg * (
            num_text2audio_seqs + num_audio2text_seqs), sort_idx=False)

    text_to_audio_idx = tf.reshape(audio_spans_trg_idx[:num_audio_spans_trg * num_text2audio_seqs],
                                   [num_text2audio_seqs, num_audio_spans_trg])

    audio_to_text_idx = tf.reshape(audio_spans_trg_idx[num_audio_spans_trg * num_text2audio_seqs:],
                                   [num_audio2text_seqs, num_audio_spans_trg])

    # First do text -> audio
    spans_all = []
    tokens_all = []
    for i in range(num_text2audio_seqs):
        # Mess with the alignments such that we mask more things of length 1,
        # and that audio targets are smaller than the text
        tokens_ragged_i = reassign_empty_tokens(tokens_ragged, mask_idx=text_to_audio_idx[i])

        # I tuned delta_thresh s.t. the probability of a span of length 1 or 2 is the same for both T2A and A2T
        tokens_ragged_i = increase_textmask(tokens_ragged_i, mask_idx=text_to_audio_idx[i],
                                            tok_centroids_vals=tok_centroids_vals,
                                            audio_start_end=audio_start_end,
                                            delta_thresh=0.125)

        spans, output_groups = mask_tokens(tokens_ragged_i, mask_idx=text_to_audio_idx[i],
                                           text_span_start_counter=i * num_audio_spans_trg,
                                           num_groups=config['num_segment_groups'],
                                           padded_seq_len=config['lang_seq_len'],
                                           do_audio_mask=True)
        spans_all.append(spans)
        tokens_all.extend(output_groups)

    # [num_groups * num_text2audio_seqs, L, 3]
    features['text2audio'] = tf.stack(tokens_all, 0)

    #######################################################
    # Now do audio -> text. will this be easier? hope so!
    audio_tokens_all = []
    for i in range(num_audio2text_seqs):
        audio_span_trg_idx = audio_to_text_idx[i]

        # Convert things to the LEFT or the RIGHT of a masked-out span into text, so that the prediction of
        # the missing text makes sense (and also hopefully such that bleeding is less important)
        one_hot_mask = _one_hot(audio_span_trg_idx, N=num_audio_spans)
        one_hot_mask_exp = tf.concat([[False], one_hot_mask, [False]], 0)

        should_textify = tf.logical_or(one_hot_mask_exp[2:], one_hot_mask_exp[:-2])
        should_textify = tf.logical_and(should_textify, tf.logical_not(one_hot_mask))
        should_textify = tf.logical_and(should_textify,
                                        sample_bernoullis(config.get('convert_extra_span_to_text_prob', 0.8),
                                                          N=num_audio_spans))

        spans, output_groups = mask_tokens(tokens_ragged, mask_idx=audio_span_trg_idx,
                                           do_audio_span=tf.logical_not(should_textify),
                                           audio_token_length=config['audio_token_length'],
                                           padded_seq_len=config['lang_seq_len'],
                                           text_span_start_counter=(i + num_text2audio_seqs) * num_audio_spans_trg,
                                           num_groups=config['num_segment_groups'])
        spans_all.append(spans)
        audio_tokens_all.extend(output_groups)

    features['audio2text'] = tf.stack(audio_tokens_all, 0)

    # here's how this works. all sequences get padded to seq_len at the end bc that's the size of the joint transformer
    # if you pass in max_text_seq_len we will ensure that only the first e.g. <=1024 tokens are valid,
    # the rest will be padded
    max_text_seq_len = config.get('max_text_seq_len', config['seq_len'])

    #####################################
    # For the audio -> image part
    use_audio_tokens = sample_bernoulli(config.get('use_audio_token_prob', 1.0))
    matching_toks = []
    for i, segment_i in enumerate(segment_list):
        matching_toks.append(tf.stack([LTOVPOOL, i * config['num_audio_subsegments'], -1])[None])

        audio_subseg = []
        for j in range(config['num_audio_subsegments']):
            new_subseg = tf.stack([AUDIOSPAN, j + i * config['num_audio_subsegments'], -1])[None]
            audio_subseg.append(tf.tile(new_subseg, [config['audio_token_length'], 1]))
        audio_subseg = tf.concat(audio_subseg, 0)

        # don't bother with alignment here bc floor dividing by num_audio_subsegments later
        text_subseg = tf.stack([
            segment_i['tok_ids'],
            tf.zeros_like(segment_i['tok_ids']) + i * config['num_audio_subsegments'],
            tf.zeros_like(segment_i['tok_ids']) - 1], 1)
        matching_toks.append(tf.cond(use_audio_tokens, lambda: audio_subseg, lambda: text_subseg))
    matching_toks = tf.concat(matching_toks, 0)

    aux_info = tf.concat([
        [START], encoder.encode('title:').ids, segment_list[0]['title'],
        [START], encoder.encode('description:').ids, segment_list[0]['description'],
        [START] + encoder.encode('tags:').ids, segment_list[0]['tags'], [END],
    ], 0)
    aux_info = tf.stack([aux_info, tf.zeros_like(aux_info) - 1, tf.zeros_like(aux_info) - 1], 1)

    extra_space_for_desc = tf.maximum(max_text_seq_len - get_shape_list(matching_toks, 2)[0], 0)
    aux_info = aux_info[:extra_space_for_desc]
    matching_toks = tf.concat([aux_info, matching_toks], 0)

    features['audio_text_matching'] = pad_tokens_to_fixed_size(matching_toks, config['seq_len'])

    ####################### Random text

    num_text_seqs_in_record = config['num_text_seqs_in_record']
    random_text = tf.cast(
        tf.stack([x['random_text'] for i, x in enumerate(segment_list) if i < config['num_text_seqs_in_record']]),
        dtype=tf.int32)

    assert config['num_text_seqs'] <= num_text_seqs_in_record
    random_inds = uniform_random_select(num_text_seqs_in_record, config['num_text_seqs'])
    random_text = tf.gather(random_text, random_inds)

    random_text_l = []
    counter = num_audio_spans_trg * (num_audio2text_seqs + num_text2audio_seqs)

    token_is_valid_tf = tf.constant(TOKEN_IS_VALID, dtype=tf.bool)

    for i in range(config['num_text_seqs']):
        # span_budget = int(desired_len / (ev / mask_rate - ev + 1))
        _ev = 5.5
        if 'text_span_budget' in config:
            span_budget = config['text_span_budget']
        else:
            span_budget = int(max_text_seq_len / (_ev / config['mask_rate'] - _ev + 1.0))
        print(f"Using span budget of {span_budget}", flush=True)
        tokens_ragged_i, extra_lhs, extra_rhs = convert_rawtext_into_fake_segments(random_text[i],
                                                                                   desired_len=max_text_seq_len,
                                                                                   span_budget=span_budget,
                                                                                   use_v1_stats='ytt180m' in config['train_fns'])

        # 4x as often, pick something that only has characters we see in YouTube
        want_to_mask = tf.gather(token_is_valid_tf, tokens_ragged_i)
        mask_w = 0.2 + 0.8 * tf.cast(tf.reduce_all(want_to_mask, -1), dtype=tf.float32)
        do_mask_i = random_categorical_without_replacement(logits=tf.math.log(mask_w), num_samples=span_budget)
        do_mask_i = tf.sort(do_mask_i)
        spans_i, tokens_i = mask_tokens(tokens_ragged_i, do_mask_i, text_span_start_counter=counter, num_groups=1)

        # Add in extra LHS and extra RHS if under max len
        tokens_i = tokens_i[0]
        amt_needed = tf.maximum(max_text_seq_len - get_shape_list(tokens_i, 2)[0], 0)
        extra_lhs_len = get_shape_list(extra_lhs, 1)[0]
        amt_lhs = tf.minimum(extra_lhs_len, amt_needed // 2)

        extra_lhs = tf.stack([extra_lhs[(extra_lhs_len - amt_lhs):], tf.zeros([amt_lhs], dtype=tf.int32), tf.zeros([amt_lhs], dtype=tf.int32)-1], 1)

        extra_rhs_len = get_shape_list(extra_rhs, 1)[0]
        amt_rhs = tf.minimum(extra_rhs_len, (amt_needed+1) // 2)
        extra_rhs = tf.stack([extra_rhs[:amt_rhs], tokens_i[-1, 1] + tf.ones([amt_rhs], dtype=tf.int32), tf.zeros([amt_rhs], dtype=tf.int32)-1], 1)
        tokens_i = tf.concat([extra_lhs, tokens_i, extra_rhs], 0)

        # OK now we pad to the length of the joint transformer
        tokens_i = pad_tokens_to_fixed_size(tokens_i, padded_seq_len=config['seq_len'])

        # Filter out tokens not seen in YouTube
        spans_i = filter_out_tokens_not_in_youtube(spans_i, token_is_valid_tf=token_is_valid_tf)

        counter += span_budget
        random_text_l.append(tokens_i)
        spans_all.append(spans_i)

    features['text_spans'] = tf.concat(spans_all, 0).to_tensor()
    features['text_spans'] = pad_to_fixed_size(features['text_spans'], PADDING,
                                               output_shape=[get_shape_list(features['text_spans'], 2)[0],
                                                             config['text_span_length']], truncate=True, axis=1)

    if config['num_text_seqs'] > 0:
        features['random_text'] = tf.stack(random_text_l, 0)

    # Video src idx per segment
    features['video_src_index'] = tf.cast(tf.stack([x['video_src_index'] for x in segment_list]), dtype=tf.int32)
    features['meta'] = segment_list[0]['meta']
    features['youtube_id'] = segment_list[0]['youtube_id']

    if config.get('encode_meta', False):
        features['youtube_id'] = encode_string(features['youtube_id'], 11)
        features['meta'] = encode_string(features['meta'], 256)


    return features


def handle_duplicate_text_spans(text_spans, shape_prefix):
    """
    Make it so that if two text spans are equal, only one shows up
    :param text_spans:
    :return:
    """
    batch_size, num_text_segments, span_len = get_shape_list(text_spans, 3)
    all_ts = tf.reshape(text_spans, [batch_size * num_text_segments, span_len])
    is_eq = tf.reduce_all(all_ts[:, None] == all_ts[None, :], -1)
    random_perm_idx = uniform_random_select(batch_size * num_text_segments, batch_size * num_text_segments,
                                            sort_idx=False)
    overlay_lt_mask = (random_perm_idx[:, None] < random_perm_idx[None])

    to_kill = tf.reduce_any(is_eq & overlay_lt_mask, 0)
    all_ts = tf.where(to_kill[:, None], x=tf.fill([batch_size * num_text_segments, span_len], PADDING), y=all_ts)
    return tf.reshape(all_ts, shape_prefix + [num_text_segments, span_len])

def handle_batch(batched_tensor, num_devices=None, use_bfloat16=False):
    """
    Deal with the fact that for a batched tensor, the pointers are off
    nvm i'm just not going to worry about that and make the pointers only valid in-batch since we never
    link to anything outside of the batch
    :param batched_tensor:
    :return:
    """
    # Mask batch
    logger.info("BEFORE HANDLING BATCH")
    for k, v in batched_tensor.items():
        logger.info("{}: {}".format(k, v.shape))

    batch_size, num_segments, hw, ppthree_ = get_shape_list(batched_tensor['images'], 4)

    if num_devices is not None:
        assert num_devices <= batch_size
        assert batch_size % num_devices == 0
        shape_prefix = [num_devices, batch_size // num_devices]
        logger.info("{} devices: shape prefix is {}".format(num_devices, shape_prefix))
    else:
        logger.info("No devices, batch size is just {}".format(batch_size))
        shape_prefix = [batch_size]

    batched_tensor['images'] = tf.reshape(batched_tensor['images'], shape_prefix + [num_segments * hw, ppthree_])

    batch_size_, num_segments_, num_audio_subsegments, audio_seq_length, num_mels = get_shape_list(
        batched_tensor['audio_clips'], 5)
    batched_tensor['audio_clips'] = tf.reshape(batched_tensor['audio_clips'],
                                               shape_prefix + [num_segments * num_audio_subsegments * audio_seq_length,
                                                               num_mels])

    # batched_tensor['text_spans'] = handle_duplicate_text_spans(batched_tensor['text_spans'], shape_prefix=shape_prefix)
    batch_size, num_text_segments, span_len = get_shape_list(batched_tensor['text_spans'], 3)
    batched_tensor['text_spans'] = tf.reshape(batched_tensor['text_spans'], shape_prefix + [num_text_segments, span_len])
    batched_tensor['video_src_index'] = tf.reshape(batched_tensor['video_src_index'], shape_prefix + [num_segments])

    # The hidden order is always [(batch, sub-batch, mask idx)]
    for k in ['text2audio', 'audio2text', 'audio_text_matching', 'random_text']:
        if k in batched_tensor:
            x_shape = get_shape_list(batched_tensor[k])
            x2 = tf.reshape(batched_tensor[k], shape_prefix + [int(np.prod(x_shape[1:-2])), x_shape[-2], 3])
            batched_tensor[k] = x2[..., 0]
            batched_tensor[k + '/audio_ptr'] = x2[..., 1]
            batched_tensor[k + '/text_ptr'] = x2[..., 2]

    # Delete if not in debug mode
    for k in ['meta', 'youtube_id']:
        if (num_devices is not None) and (batched_tensor[k].dtype == tf.string):
            batched_tensor.pop(k, None)
        else:
            old_shape = get_shape_list(batched_tensor[k])
            batched_tensor[k] = tf.reshape(batched_tensor[k], shape_prefix + old_shape[1:])

    if use_bfloat16:
        batched_tensor['images'] = tf.cast(batched_tensor['images'], dtype=tf.bfloat16)
        batched_tensor['audio_clips'] = tf.cast(batched_tensor['audio_clips'], dtype=tf.bfloat16)
    return batched_tensor


def _debug_print_tokens(tokens: np.ndarray, do_print=True):
    """
    :param tokens:
    :return:
    """
    if not isinstance(tokens, np.ndarray):
        tokens = tokens.numpy()

    if tokens.ndim == 4:
        tokens = tokens.reshape([-1] + list(tokens.shape[2:]))
    elif tokens.ndim == 2:
        tokens = tokens[None]
    outs = []
    len_out = []
    for b, tokens_b in enumerate(tokens):
        tokens_b = tokens_b[tokens_b[:, 0] > 0]

        out = []
        audio_to_merge = set()
        for x in tokens_b:
            token_id, audio_src, text_src = x.tolist()

            if token_id == AUDIOSPAN:
                if audio_src not in audio_to_merge:
                    out.append('<|AI{:02d}|>'.format(audio_src))
                    audio_to_merge.add(audio_src)
            elif token_id == MASK:
                out.append('<|MASK text={:02d},audio={:02d}|>'.format(text_src, audio_src))
            else:
                out.append(encoder.decode([token_id], skip_special_tokens=False))
        out = ''.join(out)
        if do_print:
            print("{:02d}) (len={}) {}".format(b, tokens_b.shape[0], ''.join(out)), flush=True)
        outs.append(out)
        len_out.append(tokens_b.shape[0])
    return outs, len_out


def tokens_to_segments(tokens: np.ndarray, num_audio_segments):
    """
    Similar to _debug_print_tokens, i'm going to create a dataframe at the segment level...
    :param tokens:
    :return:
    """
    assert tokens.ndim == 2
    # need valid audio segment
    tokens = tokens[(tokens[:, 1] != -1) & (tokens[:, 0] > 0)]
    out = []
    for i in range(num_audio_segments):
        tokens_i = tokens[tokens[:, 1] == i]
        out.append(_debug_print_tokens(tokens_i, do_print=False)[0][0])
    return out


def _debug_invert_imgpatches(img, h, w, patch_size):
    """
    Inverts a sequence of patches [H//P * W//P, P * P * 3] into the image
    :param img:
    :param h: how many patches in height
    :param w: how many patches in width
    :param patch_size: Int for the patch size
    :return:
    """
    *leading_dims, nseq, pp3 = img.shape
    assert pp3 == (3 * patch_size * patch_size)
    assert nseq == (h * w)
    img = img.reshape(list(leading_dims) + [h, w, patch_size, patch_size, 3])
    img = img.swapaxes(-4, -3)
    img = img.reshape(list(leading_dims) + [h * patch_size, w * patch_size, 3])
    return img


def make_dataset(config, fns, batch_size, num_devices=None, is_training=True):
    """
    Create tf.data dataset for a single tfrecord, or a few. I'm splitting this up because ocassionally I get DNS issues when accessing
    google cloud, even while on google cloud. idk why

    :param merged_config:
    :param fns:
    :param batch_size:
    :param num_devices:
    :param is_training:
    :return:
    """
    merged_config = deepcopy(config['data'])
    merged_config.update(config['model'])

    num_parallel_reads = config['device'].get('num_parallel_reads', 4)
    num_parallel_reads = min(len(fns), num_parallel_reads) if isinstance(fns, list) else None
    if not is_training:
        num_parallel_reads = 1
    print(f"Constructing TFRecord Input FN over {fns}\n\n{num_parallel_reads} reads in parallel", flush=True)

    dataset = tf.data.TFRecordDataset(fns, num_parallel_reads=num_parallel_reads)

    options = tf.data.Options()
    options.threading.private_threadpool_size = 48
    options.threading.max_intra_op_parallelism = 1
    options.experimental_optimization.map_parallelization = True
    options.experimental_deterministic = (not is_training)
    dataset = dataset.with_options(options)

    if is_training:
        dataset = dataset.shuffle(buffer_size=config['device'].get('shuffle_buffer_size', 256))

    dataset = dataset.map(functools.partial(dataset_parser, config=merged_config),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
    dataset = dataset.map(functools.partial(handle_batch, num_devices=num_devices,
                                            use_bfloat16=merged_config['use_bfloat16']))
    return dataset


def input_fn_builder(config, make_dataset_fn=make_dataset):
    """
    Get input fn for TPU use -- for training
    :param config:
    :param is_training:
    :param as_numpy_iter:
    :return:
    """
    import jax
    from flax import jax_utils

    current_host = jax.process_index()
    num_hosts = jax.process_count()
    num_devices = jax.local_device_count()
    batch_size = config['device']['batch_size'] // num_hosts
    # random.seed(123456 + current_host)
    # non-determinism for reloading...
    random.seed(int(time.time()))
    tf.random.set_seed(int(time.time()))

    matching_fns = []
    for i in range(config['data']['num_train_files']):
        if i % num_hosts == current_host:
            matching_fns.append(config['data']['train_fns'].format(i))
    assert len(matching_fns) > 0

    def _multi_iterator0():
        n_fns_per_cycle = min(config['device'].get('n_fns_per_cycle', 32), len(matching_fns))
        while len(matching_fns) % n_fns_per_cycle != 0:
            print(f"!!!Truncating n_fns_per_cycle {n_fns_per_cycle} -> {n_fns_per_cycle - 1} so it fits")
            n_fns_per_cycle -= 1

        n_epochs = 0
        while True:
            fns_shuff = [x for x in matching_fns]
            random.shuffle(fns_shuff)
            print(f"Now on epoch {n_epochs}")
            for s, e in batch_index_iterator(len(fns_shuff), batch_size=n_fns_per_cycle, skip_end=True):
                print(f"Resetting iterator, epoch={n_epochs}, batch of fns={s}:{e} /{len(fns_shuff)}", flush=True)
                try:
                    dataset = make_dataset_fn(config, fns=fns_shuff[s:e], batch_size=batch_size,
                                              num_devices=num_devices, is_training=True)
                    for item in dataset:
                        item = jax.tree_map(lambda x: x._numpy(), item)
                        yield item
                # except tf.errors.FailedPreconditionError as e:
                except Exception as e:
                    print(str(e), flush=True)
                    time.sleep(5)
            n_epochs += 1

    if config['device'].get('prefetch_size', 1) > 0:
        return jax_utils.prefetch_to_device(_multi_iterator0(), size=config['device'].get('prefetch_size', 1))
    return _multi_iterator0()


if __name__ == '__main__':

    # NOTE: This is some debugging code that may or may not be helpful for analyzing the data

    import yaml

    with open('configs/base.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    merged_config = deepcopy(config['data'])
    merged_config.update(config['model'])
    config = merged_config


    dataset = tf.data.TFRecordDataset(['train00000of32800.tfrecord'])
    # # For eager debugging
    # for record in dataset:
    #     assert False
        # x = dataset_parser(record, config)

    dataset = dataset.map(lambda x: dataset_parser(x, config))
    B = 8
    dataset = dataset.batch(batch_size=B, drop_remainder=True)
    dataset = dataset.map(handle_batch)
    start = time.time()

    sizes = []
    # Debug
    lens = []
    w2c = defaultdict(int)
    span_lens_by_pos = []
    char2count = defaultdict(int)
    tok_to_count = np.zeros([encoder.get_vocab_size()], dtype=np.int32)
    tok_to_count_text = np.zeros([encoder.get_vocab_size()], dtype=np.int32)

    for nei, next_element in enumerate(dataset):
        print("Done in {:.3f}".format(time.time() - start), flush=True)

        span_lens_by_pos.append((next_element['text_spans'].numpy() != 0).sum(-1))

        for tok_i in next_element['text_spans'].numpy()[:, :36].reshape((-1)):
            tok_to_count[tok_i] += 1
        for tok_i in next_element['text_spans'].numpy()[:, 36:].reshape((-1)):
            tok_to_count_text[tok_i] += 1

        for b in range(B):
            ts_dec = encoder.decode_batch(next_element['text_spans'][b], skip_special_tokens=False)
            print("\n\n\n TEXT SPANS\n-----\n")
            for i, ts_i in enumerate(ts_dec):
                print("{:02d}) {}".format(i, ts_i.replace('<|PAD|>', '')), flush=True)
                w2c[ts_i.replace('<|PAD|>', '')] += 1
                if i < 36:
                    for c in ts_i.replace('<|PAD|>', ''):
                        char2count[c] += 1

            print("\n\n\n TEXT TO AUDIO TOKENS\n-----\n")
            _, len_t2a = _debug_print_tokens(np.stack([next_element['text2audio'][b].numpy(),
                                                       next_element['text2audio/audio_ptr'][b].numpy(),
                                                       next_element['text2audio/text_ptr'][b].numpy(),
                                                       ], -1))
            print("\n\n\n AUDIO TO TEXT TOKENS\n-----\n")
            _, len_a2t = _debug_print_tokens(np.stack([next_element['audio2text'][b].numpy(),
                                                       next_element['audio2text/audio_ptr'][b].numpy(),
                                                       next_element['audio2text/text_ptr'][b].numpy(),
                                                       ], -1))

            print("\n\n\n AUDIO-TEXT MATCHING")
            _, len_atm = _debug_print_tokens(np.stack([next_element['audio_text_matching'][b].numpy(),
                                                       next_element['audio_text_matching/audio_ptr'][b].numpy(),
                                                       next_element['audio_text_matching/text_ptr'][b].numpy(),
                                                       ], -1))
            print("\n\n\n RANDOM TEXT TOKENS\n-----\n")
            _, len_rt = _debug_print_tokens(np.stack([next_element['random_text'][b].numpy(),
                                                      next_element['random_text/audio_ptr'][b].numpy(),
                                                      next_element['random_text/text_ptr'][b].numpy(),
                                                      ], -1))
            lens.append({'t2a': len_t2a[0], 'a2t': len_a2t[0], 'atm': len_atm[0], 'rt': len_rt[0],
                         'tsdec': len([ts_i for ts_i in ts_dec if len(ts_i.replace('<|PAD|>', '')) > 0])})

    import pandas as pd

    lens = pd.DataFrame(lens)
    print(lens.mean(0))
    print("99% len: {}".format(lens.quantile(0.99)))
    print("95% len: {}".format(lens.quantile(0.95)))
    print("90% len: {}".format(lens.quantile(0.90)))

    # To get a good value for tsdec (since it's batched)    df['token'] = df['token'].apply(encoder.id_to_token)
    # Assuming the batch size is 4
    lens['tsdec'].values.reshape(-1, B).mean(-1).min()

    df = pd.DataFrame(np.stack([next_element['text2audio'][b, 0].numpy(),
                                next_element['text2audio/audio_ptr'][b, 0].numpy(),
                                next_element['text2audio/text_ptr'][b, 0].numpy(),
                                ], -1), columns=['token', 'audio_ptr', 'text_ptr'])

    span_lens_by_pos = np.concatenate(span_lens_by_pos, 0)
    numer = span_lens_by_pos.sum(0)
    denom = (span_lens_by_pos > 0).sum(0)
    span_lens_by_pos_mean = numer / denom
    print("Text to audio: {:.3f}".format(span_lens_by_pos_mean[:12].mean()))
    print("Audio to text: {:.3f}".format(span_lens_by_pos_mean[12:24].mean()))
    print("Random text: {:.3f}".format(span_lens_by_pos_mean[24:].mean()))


    def _calc_span_dist(span_lens):
        lens = np.zeros([15], dtype=np.int32)
        lens_l = []
        for x in span_lens.reshape(-1).tolist():
            if x > 0:
                lens[x - 1] += 1
                lens_l.append(x)
        return lens / (lens.sum() + 1e-5)


    print("Text to audio: {}".format(_calc_span_dist(span_lens_by_pos[:, :12])))
    t2a = _calc_span_dist(span_lens_by_pos[:, :12])
    a2t = _calc_span_dist(span_lens_by_pos[:, 12:24])
    rt = _calc_span_dist(span_lens_by_pos[:, 24:])

    print("KL divergence T2A -> A2T: {}".format((t2a * np.log(t2a / a2t)).sum()), flush=True)
    print(t2a)
    print("KL divergence T2A -> RT: {}".format((t2a * np.log(t2a / rt)).sum()), flush=True)
    print(a2t)
    print("KL divergence A2T -> RT: {}".format((a2t * np.log(a2t / rt)).sum()), flush=True)

    probs = np.maximum(t2a, a2t)

    gamma = 0.5
    probs_i = probs.copy()
    for i in range(14):
        probs_i = np.concatenate([[0.0], probs_i[:-1]/probs_i.sum()], 0)
        probs += (probs_i * np.power(gamma, i+1))
    probs = probs / probs.sum()

    print("ev: {}, desired ev {}".format(((np.arange(probs.shape[0]) + 1) * probs).sum(), 5.4 * 5.4 / 5.0))


    # get KL on tokens level
    # tok_to_count_float = tok_to_count.astype(np.float64) + 0.1
    # tok_to_count_text_float = tok_to_count_text.astype(np.float64) + 0.1
    #
    # youtube_dist = tok_to_count_float / tok_to_count_float.sum()
    # text_dist = tok_to_count_text_float / tok_to_count_text_float.sum()
    #
    # count_kl = text_dist * np.log(text_dist / youtube_dist)
    # bad_tokens = []
    # funny_g = encoder.id_to_token(32156)[0]
    # for j, i in enumerate(np.argsort(-count_kl)):
    #     kl_i = count_kl[i]
    #     tok_i = encoder.id_to_token(i)
    #     youtube_count = tok_to_count[i]
    #     text_count = tok_to_count_text[i]
    #     print(f"{tok_i:<20s} Token {i:05d}: KL {kl_i:.6f} (YoutubeCT {youtube_count}, TextCT {text_count})", flush=True)
    #     if (youtube_count == 0) and len(tok_i.replace(funny_g, '')) == 0:
    #         bad_tokens.append(i)