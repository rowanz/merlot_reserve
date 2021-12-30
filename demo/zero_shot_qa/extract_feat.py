"""
Converting the validation and test data for MSRVTT-QA.
"""
import sys
sys.path.append('../')
import pdb
import json
import os
from mreserve.preprocess import video_to_segments_zero_shot, preprocess_video, encoder, MASK
from mreserve.modeling import PretrainedMerlotReserve
import csv
import datetime
import pickle
import jax
import jax.numpy as jnp
import pickle
import glob
import subprocess
from tqdm import tqdm

grid_size = (18, 32)
time_interval = 3.0  # 2.0
# average video length: 15.17965927218444

split = 'test'
factor = 6
print(f'{split}_infill.json.{factor}')

segment_path = '/net/nfs2.mosaic/mreserve/msrvtt-qa/cache/segments_msrvtt_t{}'.format(time_interval)
if not os.path.exists(segment_path):
    os.mkdir(segment_path)

def get_length(filename):
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    return float(result.stdout)


video_path = '/net/nfs2.corp/mosaic/yj/msr-vtt/video/TestVideo/video%s.mp4' #TrainValVideo

narration_id_to_se_time = {}
with open(f'MSRVTT-QA/{split}_infill.json.{factor}') as f:
    ds = json.load(f)
    for item in tqdm(ds):
        if 'bad_template' in item:
            # if template is bad, automatically count it as wrong
            continue

        try:
            annot_id = item["id"]
            video_id = item["video_id"]
            video_file = video_path % video_id
            video_length = get_length(video_file)

            times = []
            num_chunk = int(video_length // time_interval)
            cut_length = num_chunk * time_interval
            st = round((video_length - cut_length) / 2, 2)
            for i in range(num_chunk):
                et = st + time_interval
                times.append({'start_time': st, 'end_time': et, 'mid_time': round((st + et) / 2.0, 2)})
                st += time_interval

            video_segments = video_to_segments_zero_shot(video_file, time_interval=time_interval, times=times)

            assert len(video_segments) == num_chunk, f'{video_id} has wrong num of chunks'

            # Set up a fake classification task.
            video_segments[-1]['text'] = item['question']
            video_segments[-1]['use_text_as_input'] = True
            for i in range(num_chunk - 1):
                video_segments[i]['use_text_as_input'] = False

            sample = {'video_id': video_id, 'annot_id': annot_id, 'video_length': video_length, 'times': times,
                     'question': item['question'], 'answer': item['answer']}
            segments = {'video_segments': video_segments, 'info': sample}
            pickle.dump(segments, open(os.path.join(segment_path, f'{annot_id}.pkl'), 'wb'))

        except:
            print(f'skip question {annot_id} with video {video_id}')
            continue
