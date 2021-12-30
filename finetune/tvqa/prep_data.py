"""
Convert TVQA into tfrecords
"""
import sys

sys.path.append('../../')
import argparse
import hashlib
import io
import json
import os
import random
import numpy as np
from tempfile import TemporaryDirectory
from copy import deepcopy

from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
from google.cloud import storage
from sacremoses import MosesDetokenizer
import regex as re
from tqdm import tqdm
import pandas as pd
from finetune.common_data_utils import *
from collections import defaultdict
import colorsys
import hashlib
import tempfile
import subprocess
from scipy.io import wavfile
from mreserve.preprocess import make_spectrogram, invert_spectrogram
from mreserve.lowercase_encoder import START
import pysrt
from unidecode import unidecode
import ftfy


parser = create_base_parser()
parser.add_argument(
    '-data_dir',
    dest='data_dir',
    default='/home/rowan/datasets3/tvqa/',
    type=str,
    help='Image directory.'
)
"""
Must set things up like this in the data_dir
drwxr-xr-x 1 rowan rowan    1155072 Aug 19  2018 tvqa_subtitles
drwxr-xr-x 1 rowan rowan       4096 Aug 27  2018 tvqa_qa_release
drwxr-xr-x 1 rowan rowan       4096 Jan 18  2020 tvqa_plus_annotations_with_test
drwxrwxr-x 1 rowan rowan       4096 Sep 14 08:06 tvqa_frames
-rw-rw-r-- 1 rowan rowan 4294967296 Nov 12  2018 tvqa_video_frames_fps3_hq.tar.gz.aa
-rw-rw-r-- 1 rowan rowan 4294967296 Nov 12  2018 tvqa_video_frames_fps3_hq.tar.gz.ab
-rw-rw-r-- 1 rowan rowan 4294967296 Nov 12  2018 tvqa_video_frames_fps3_hq.tar.gz.ac
-rw-rw-r-- 1 rowan rowan 4294967296 Nov 12  2018 tvqa_video_frames_fps3_hq.tar.gz.ad
-rw-rw-r-- 1 rowan rowan 4294967296 Nov 12  2018 tvqa_video_frames_fps3_hq.tar.gz.ae
-rw-rw-r-- 1 rowan rowan 4294967296 Nov 12  2018 tvqa_video_frames_fps3_hq.tar.gz.af
-rw-rw-r-- 1 rowan rowan 4294967296 Nov 12  2018 tvqa_video_frames_fps3_hq.tar.gz.ag
-rw-rw-r-- 1 rowan rowan 4294967296 Nov 12  2018 tvqa_video_frames_fps3_hq.tar.gz.ah
-rw-rw-r-- 1 rowan rowan 4294967296 Nov 12  2018 tvqa_video_frames_fps3_hq.tar.gz.ai
-rw-rw-r-- 1 rowan rowan 4294967296 Nov 12  2018 tvqa_video_frames_fps3_hq.tar.gz.aj
-rw-rw-r-- 1 rowan rowan 4294967296 Nov 12  2018 tvqa_video_frames_fps3_hq.tar.gz.ak
-rw-rw-r-- 1 rowan rowan 4294967296 Nov 12  2018 tvqa_video_frames_fps3_hq.tar.gz.al
-rw-rw-r-- 1 rowan rowan 4294967296 Nov 12  2018 tvqa_video_frames_fps3_hq.tar.gz.am
-rw-rw-r-- 1 rowan rowan 4294967296 Nov 12  2018 tvqa_video_frames_fps3_hq.tar.gz.an
-rw-rw-r-- 1 rowan rowan 4294967296 Nov 12  2018 tvqa_video_frames_fps3_hq.tar.gz.ao
-rw-rw-r-- 1 rowan rowan 4294967296 Nov 12  2018 tvqa_video_frames_fps3_hq.tar.gz.ap
-rw-rw-r-- 1 rowan rowan 4294967296 Nov 12  2018 tvqa_video_frames_fps3_hq.tar.gz.aq
-rw-rw-r-- 1 rowan rowan 4294967296 Nov 12  2018 tvqa_video_frames_fps3_hq.tar.gz.ar
-rw-rw-r-- 1 rowan rowan 4294967296 Nov 12  2018 tvqa_video_frames_fps3_hq.tar.gz.as
-rw-rw-r-- 1 rowan rowan 4294967296 Nov 12  2018 tvqa_video_frames_fps3_hq.tar.gz.at
-rw-rw-r-- 1 rowan rowan 4294967296 Nov 12  2018 tvqa_video_frames_fps3_hq.tar.gz.au
-rw-rw-r-- 1 rowan rowan 4294967296 Nov 12  2018 tvqa_video_frames_fps3_hq.tar.gz.av
-rw-rw-r-- 1 rowan rowan 4294967296 Nov 12  2018 tvqa_video_frames_fps3_hq.tar.gz.aw
-rw-rw-r-- 1 rowan rowan 4294967296 Nov 12  2018 tvqa_video_frames_fps3_hq.tar.gz.ax
-rw-rw-r-- 1 rowan rowan 4294967296 Nov 12  2018 tvqa_video_frames_fps3_hq.tar.gz.ay
-rw-rw-r-- 1 rowan rowan 4294967296 Nov 12  2018 tvqa_video_frames_fps3_hq.tar.gz.az
-rw-rw-r-- 1 rowan rowan 4294967296 Nov 12  2018 tvqa_video_frames_fps3_hq.tar.gz.ba
-rw-rw-r-- 1 rowan rowan 4294967296 Nov 12  2018 tvqa_video_frames_fps3_hq.tar.gz.bb
-rw-rw-r-- 1 rowan rowan 4294967296 Nov 12  2018 tvqa_video_frames_fps3_hq.tar.gz.bc
-rw-rw-r-- 1 rowan rowan 4294967296 Nov 12  2018 tvqa_video_frames_fps3_hq.tar.gz.bd
-rw-rw-r-- 1 rowan rowan 4294967296 Nov 12  2018 tvqa_video_frames_fps3_hq.tar.gz.be
-rw-rw-r-- 1 rowan rowan 4294967296 Nov 12  2018 tvqa_video_frames_fps3_hq.tar.gz.bf
-rw-rw-r-- 1 rowan rowan 4294967296 Nov 12  2018 tvqa_video_frames_fps3_hq.tar.gz.bg
-rw-rw-r-- 1 rowan rowan 4294967296 Nov 12  2018 tvqa_video_frames_fps3_hq.tar.gz.bh
-rw-rw-r-- 1 rowan rowan  761207798 Nov 12  2018 tvqa_video_frames_fps3_hq.tar.gz.bi
-rw-rw-r-- 1 rowan rowan       2450 Nov 25  2018 tvqa_video_frames_fps3_hq.checksum.txt
-rw-r--r-- 1 rowan rowan    4061313 Apr 22  2019 tvqa_plus_val.json
-rw-r--r-- 1 rowan rowan   31270388 Apr 22  2019 tvqa_plus_train.json
-rw-rw-r-- 1 rowan rowan 4294967296 Apr 29  2019 tvqa_audios.tar.gz.aa
-rw-rw-r-- 1 rowan rowan 4294967296 Apr 29  2019 tvqa_audios.tar.gz.ab
-rw-rw-r-- 1 rowan rowan 4294967296 Apr 29  2019 tvqa_audios.tar.gz.ac
-rw-rw-r-- 1 rowan rowan 4294967296 Apr 29  2019 tvqa_audios.tar.gz.ad
-rw-rw-r-- 1 rowan rowan 4294967296 Apr 29  2019 tvqa_audios.tar.gz.ae
-rw-rw-r-- 1 rowan rowan 4294967296 Apr 29  2019 tvqa_audios.tar.gz.af
-rw-rw-r-- 1 rowan rowan 4294967296 Apr 29  2019 tvqa_audios.tar.gz.ag
-rw-rw-r-- 1 rowan rowan 2750112255 Apr 29  2019 tvqa_audios.tar.gz.ah
-rw-rw-r-- 1 rowan rowan        448 Apr 29  2019 tvqa_audios.checksum.txt
-rw-rw-r-- 1 rowan rowan   15495443 Jul 23  2019 tvqa_subtitles.tar.gz
-rw-rw-r-- 1 rowan rowan   14474003 Jul 23  2019 tvqa_qa_release.tar.gz
-rw-rw-r-- 1 rowan rowan    6718915 Jul 23  2019 tvqa_plus_annotations.tar.gz
-rw-rw-r-- 1 rowan rowan       6899 Nov 11  2019 tvqa_dl_instructions.txt
-rw-rw-r-- 1 rowan rowan   47577821 Nov 11  2019 subs.pkl
-rw-rw-r-- 1 rowan rowan    7323926 Jan 19  2020 tvqa_plus_annotations_preproc_with_test.tar.gz
"""

args = parser.parse_args()
random.seed(args.seed)

out_fn = os.path.join(args.base_fn, 'tvqa', '{}{:03d}of{:03d}.tfrecord'.format(args.split, args.fold, args.num_folds))

split_fn = {
    'train': 'tvqa_train.jsonl',
    'val': 'tvqa_val.jsonl',
    'test': 'tvqa_test_public.jsonl',
}[args.split]
split_fn = os.path.join(args.data_dir, 'tvqa_qa_release', split_fn)

data = []
with open(split_fn, 'r') as f:
    for idx, l in enumerate(f):
        if idx % args.num_folds != args.fold:
            continue
        item = json.loads(l)
        item['ts'] = tuple([float(x) for x in item['ts'].split('-')])
        assert len(item['ts']) == 2
        if np.any(np.isnan(item['ts'])):
            item['ts'] = (0, 9999.0)
        data.append(item)

ts_lens = [x['ts'][1] - x['ts'][0] for x in data]
max_end = max([x['ts'][1] for x in data])

def parse_item(item):
    qa_item = {'qa_query': item.pop('q'), 'qa_choices': [item.pop(f'a{i}') for i in range(5)],
               'qa_label': item.get('answer_idx', 0),
               'id': '{:06d}~{}'.format(item.pop('qid'), item['vid_name'])}

    show_shortname = {
        'Grey\'s Anatomy': 'grey',
        "How I Met You Mother": 'met',
        "Friends": 'friends',
        'The Big Bang Theory': 'bbt',
        'House M.D.': 'house',
        'Castle': 'castle',
    }[item['show_name']]
    frames_path = os.path.join(args.data_dir, 'tvqa_frames', 'frames_hq', f'{show_shortname}_frames',
                            item['vid_name'])

    max_frame_no = max([int(x.split('.')[0]) for x in os.listdir(frames_path)])
    max_time = (max_frame_no - 1) / 3.0

    ts0, ts1 = item.pop('ts')
    ts0 = max(ts0, 0)
    ts1 = min(ts1, max_time)
    segment_size = 4.6666667 # this differs a tiny bit from pretraining. basically i'm using denser frames here
                             # to avoid needing to cut off any audio

    # Midpoint will be the middle of the (middle) chunk, so round it to the nearest 1/3rd
    # because that's when frames were extracted
    midpoint = (ts0 + ts1) / 2.0
    midpoint = round(midpoint * 3) / 3

    t_start = midpoint - segment_size * 0.5
    t_end = midpoint + segment_size * 0.5

    # Try to extend by 3 segments in either direction of the middle
    times_used0 = [{'start_time': t_start, 'end_time': t_end}]
    for i in range(6):
        for delta in [-segment_size, segment_size]:
            t0 = t_start + delta * (i+1)
            t1 = t_end + delta * (i+1)

            t0 = round(t0 * 3) / 3
            t1 = round(t1 * 3) / 3

            if t1 < 0:
                continue
            if t0 > max_time:
                continue
            if len(times_used0) < 7:
                times_used0.append({'start_time': t0, 'end_time': t1})
    times_used0 = sorted(times_used0, key=lambda x: x['start_time'])

    ###
    frames = []
    times_used = []
    for trow in times_used0:
        t_midframe = (trow['start_time'] + trow['end_time']) / 2.0
        t_mid_3ps_idx = int(round(t_midframe * 3.0)) + 1
        t_mid_3ps_idx = max(t_mid_3ps_idx, 1)
        t_mid_3ps_idx = min(t_mid_3ps_idx, max_frame_no)

        fn = os.path.join(frames_path, f'{t_mid_3ps_idx:05d}.jpg')
        if os.path.exists(fn):
            image = Image.open(fn)
            image = resize_image(image, shorter_size_trg=450, longer_size_max=800)
            frames.append(image)
            times_used.append(trow)
        else:
            print(f"{fn} doesn't exist")

    ### idk why this is the case...
    show_audioname = show_shortname if show_shortname != 'bbt' else 'bbt_new'


    audio_fn_mp3 = os.path.join(args.data_dir, 'tvqa_frames', 'audios', show_audioname,
                            f'{show_shortname}_audios', item['vid_name'] + '.mp3')
    # Start the process
    temp_folder = tempfile.TemporaryDirectory()
    audio_fn = os.path.join(temp_folder.name, 'audio.wav')

    # Before we were sampling at 22050, and we had 188 mel windows for 5 sec.
    # now we want exactly 180 windows from 4.6667 sec.
    # 4.66667 * sr / 180 = 5 * 22050 / 188

    ffmpeg_process = subprocess.Popen(['ffmpeg', '-y', '-i', audio_fn_mp3, '-ac', '1', '-ar', '22620',
                                       audio_fn], stdout=-1, stderr=-1, text=True)
    try:
        stdout, stderr = ffmpeg_process.communicate(None, timeout=5.0)
    except subprocess.TimeoutExpired:
        ffmpeg_process.kill()
        stdout, stderr = subprocess.TimeoutExpired.communicate()
        raise ValueError("couldnt convert in time")
    except:  # Keyboardinterrupt
        ffmpeg_process.kill()
        raise
    if not os.path.exists(audio_fn):
        import ipdb
        ipdb.set_trace()
    ffmpeg_process.kill()
    sr, waveform = wavfile.read(audio_fn, mmap=False)
    waveform = waveform.astype('float32')
    waveform /= max(np.abs(waveform).max(), 1.0)

    # Pad to max time just in case
    desired_final_frame = int(sr * max([t['end_time'] for t in times_used]))
    if waveform.size < desired_final_frame:
        waveform = np.concatenate([waveform, np.zeros(desired_final_frame - waveform.size, dtype=np.float32)], 0)

    # Process each segment. here i'm always using a playback_speed of 1 (aka no fast forwarding).
    spectrograms = []
    for ts_group in times_used:
        start_idx = int(sr * ts_group['start_time'])
        end_idx = int(sr * ts_group['end_time'])

        if start_idx < 0:
            # i have to add 1 here because casting to int floors "up" rather than "down" if start time is negative.
            wav_ts = np.concatenate([np.zeros(1-start_idx, dtype=np.float32), waveform[:end_idx]], 0)
        else:
            wav_ts = waveform[start_idx:end_idx]
        spectrograms.append(make_spectrogram(wav_ts, playback_speed=1, sr=22050, pad_size=0))
    temp_folder.cleanup()

    # Get subtitles
    #############################################################
    show_subname = item['vid_name']
    sub_fn = os.path.join(args.data_dir, 'tvqa_subtitles', show_subname + '.srt')
    if not os.path.exists(sub_fn):
        import ipdb
        ipdb.set_trace()

    def _parse_ts(ts):
        sec = ts.hours * 3600 + ts.minutes * 60 + ts.seconds + ts.milliseconds / 1000.0
        return sec
    for ts in times_used:
        ts['sub'] = []

    bounds = np.array([x['start_time'] for x in times_used] + [times_used[-1]['end_time']])
    for sub_item in pysrt.open(sub_fn):
        start_time = _parse_ts(sub_item.start)
        end_time = _parse_ts(sub_item.end)
        mid_time = (start_time + end_time) / 2.0
        pos = np.searchsorted(bounds, mid_time)
        if (pos > 0) and (pos <= len(times_used)):
            times_used[pos-1]['sub'].append(sub_item.text)

    for ts in times_used:
        ts['sub'] = ' '.join(ts['sub'])
        ts['sub'] = unidecode(ftfy.ftfy(ts['sub'])).replace('\n', ' ')

    # Figure out the relative position of the annotation
    my_duration = times_used0[-1]['end_time'] - times_used[0]['start_time']
    rel_localized_tstart = (ts0 - times_used[0]['start_time']) / my_duration
    rel_localized_tend = (ts1 - times_used[0]['start_time']) / my_duration
    qa_item['rel_localization'] = (rel_localized_tstart, rel_localized_tend)

    qa_item['num_frames'] = len(frames)
    qa_item['magic_number'] = 255.0 / max(np.percentile(np.stack(spectrograms).reshape(-1, 65), 99), 1.0)
    qa_item['_mp3_fn'] = audio_fn_mp3
    qa_item['_frames_path'] = frames_path
    qa_item['_time_interval'] = [ts0, ts1]


    # Pad to 7
    for i in range(7 - len(frames)):
        frames.append(frames[-1])
        spectrograms.append(spectrograms[-1])
        times_used.append({'start_time': -1, 'end_time': -1, 'sub': ''})

    return qa_item, frames, spectrograms, times_used

num_written = 0
max_len = 0
with GCSTFRecordWriter(out_fn, auto_close=False) as tfrecord_writer:
    for item in data:
        qa_item, frames, specs, subs = parse_item(item)

        # Tack on the relative position of the localized timestamp, plus a START token for separation
        query_enc = encoder.encode(qa_item['qa_query']).ids
        ts_enc = encoder.encode('{} to {}'.format(int(qa_item['rel_localization'][0] * 100),
                                                  int(qa_item['rel_localization'][1] * 100),
                                                  )).ids + [START]
        query_enc = ts_enc + query_enc

        feature_dict = {
            'id': bytes_feature(qa_item['id'].encode('utf-8')),
            'magic_number': float_list_feature([qa_item['magic_number']]),
            'qa_query': int64_list_feature(query_enc),
            'qa_label': int64_feature(qa_item['qa_label']),
            'num_frames': int64_feature(qa_item['num_frames']),
        }

        max_query = 0
        for i, choice_i in enumerate(encoder.encode_batch(qa_item['qa_choices'])):
            feature_dict[f'qa_choice_{i}'] = int64_list_feature(choice_i.ids)
            max_query = max(len(choice_i.ids) + len(query_enc), max_query)

        for i, (frame_i, spec_i, subs_i) in enumerate(zip(frames, specs, subs)):
            feature_dict[f'c{i:02d}/image_encoded'] = bytes_feature(pil_image_to_jpgstring(frame_i))

            compressed = np.minimum(spec_i.reshape(-1, 65) * qa_item['magic_number'], 255.0).astype(np.uint8)
            assert compressed.shape == (180, 65)
            feature_dict[f'c{i:02d}/spec_encoded'] = bytes_feature(pil_image_to_jpgstring(Image.fromarray(compressed)))

            feature_dict[f'c{i:02d}/sub'] = int64_list_feature(encoder.encode(subs_i['sub']).ids)
            max_query += len(feature_dict[f'c{i:02d}/sub'].int64_list.value)
        max_len = max(max_len, max_query)

        if num_written < 4:
            print(f"~~~~~~~~~~~ Example {num_written} {qa_item['id']} ~~~~~~~~")
            print(encoder.decode(feature_dict['qa_query'].int64_list.value, skip_special_tokens=False), flush=True)
            for i in range(5):
                toks = feature_dict[f'qa_choice_{i}'].int64_list.value
                toks_dec = encoder.decode(toks, skip_special_tokens=False)
                lab = ' GT' if i == qa_item['qa_label'] else '   '
                print(f'{i}{lab}) {toks_dec}     ({len(toks)}tok)', flush=True)
            #
            # # Debug image
            # os.makedirs('debug', exist_ok=True)
            # for i in range(7):
            #     with open(f'debug/ex{num_written}_img{i}.jpg', 'wb') as f:
            #         f.write(feature_dict[f'c{i:02d}/image_encoded'].bytes_list.value[0])
            #
            #     jpgstr = feature_dict[f'c{i:02d}/spec_encoded'].bytes_list.value[0]
            #     inv = Image.open(io.BytesIO(jpgstr))
            #     inv_np = np.asarray(inv).astype(np.float32) / qa_item['magic_number']
            #     inv_np = inv_np[:, :64].reshape(3, 60, 64) # remove playback speed feature
            #     for ii, spec_ii in enumerate(inv_np):
            #         y = invert_spectrogram(spec_ii)
            #         wavfile.write(f'debug/ex{num_written}_audio{i}_{ii}.wav', rate=22050, data=y)
            #
            # # Get the ground truth
            # mp3_orig = qa_item['_mp3_fn']
            # print("time interval {}".format(qa_item['_time_interval']), flush=True)
            # os.system(f'cp {mp3_orig} debug/ex{num_written}_audio_raw.mp3')
            # frames_path = qa_item['_frames_path']
            # os.system(f'cp -r {frames_path} debug/ex{num_written}_frames')
            # # assert False

        example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
        tfrecord_writer.write(example.SerializeToString())
        num_written += 1
        if num_written % 100 == 0:
            print("Have written {} / {}".format(num_written, len(data)), flush=True)
    tfrecord_writer.close()

print(f'Finished writing {num_written} questions; max len = {max_len}', flush=True)
