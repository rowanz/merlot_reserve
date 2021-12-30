"""
Converting the validation and test data for EPICK KITCHEN.
"""
import sys
sys.path.append('../')
import pdb
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
import torch
import numpy as np
from tqdm import tqdm

# this is fixed for action anticipation tasks.
tau_a = 1.0
# this has to be divided by 5.
num_observed_segments = 8
time_interval = 2.0 # 2.0
tau_o = time_interval * num_observed_segments# 12.0

# ablations we want to try.
# num_observed_segments = 8 vs 6.
# time_interval = 1.0, 2.0, 5.0
# with audio / without audio
# different prompt (noun the verb / noun verb) / training noun verb stats.
assert num_observed_segments * time_interval == tau_o
segment_path = 'cache/segments_ek100_n{}_t{}'.format(num_observed_segments, time_interval)
print('loading from %s' %(segment_path))
if not os.path.exists(segment_path):
    os.mkdir(segment_path)

def convert_datetime_to_second(t):
    date_time = datetime.datetime.strptime(t, "%H:%M:%S.%f")
    a_timedelta = date_time - datetime. datetime(1900, 1, 1)
    seconds = a_timedelta.total_seconds()
    return seconds

narration_id_to_se_time = {}
with open('data/epic-kitchens-100-annotations/EPIC_100_validation.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for _, row in enumerate(spamreader):
        if _ == 0:
            continue
        
        narration_id = row[0]
        se_time = [convert_datetime_to_second(row[4]), convert_datetime_to_second(row[5])]
        narration_id_to_se_time[narration_id] = se_time

noun_to_ids = {}
verb_to_ids = {}
action_to_ids = {}

# candidate action and noun.
with open('data/ek100/actions.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for _, row in enumerate(spamreader):
        if _ == 0:
            continue

        verb, noun = row[3].split(' ')
        if ':' in noun:
            noun_split = noun.split(':')
            noun = ' '.join(noun_split[::-1])

        action = verb + ' ' + noun
        action_to_ids[action] = row[0]
        verb_to_ids[verb] = row[1]
        noun_to_ids[noun] = row[2]

print('number of noun is: %d' %(len(noun_to_ids)))
print('number of verb is: %d' %(len(verb_to_ids)))

ids_to_noun = {v:k for k, v in noun_to_ids.items()}
ids_to_verb = {v:k for k, v in verb_to_ids.items()}
ids_to_action = {v:k for k, v in action_to_ids.items()}

samples = []
# get the validation dataset.
with open('data/ek100/validation.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for _, row in enumerate(spamreader):
        narration_id = row[0]
        action_se_time = narration_id_to_se_time[narration_id]
        anticipation_se_time = [max(action_se_time[0] - tau_a - tau_o, 0), max(action_se_time[0] - tau_a, 0)]
        video_id = row[1]
        action = [row[4], row[5], row[6]]

        samples.append({'narration_id': narration_id, 'video_id': video_id, 'action_se': action_se_time, 
                        'anticipation_se': anticipation_se_time, 'action': action})


grid_size = (18, 32)
model = PretrainedMerlotReserve.from_pretrained(model_name='large', image_grid_size=grid_size)

# extract the features for epic kitchen.
# iterate all videos:
epic_55_path = '/net/nfs2.prior/datasets/epickitchens/epic-kitchens-55-all/videos/test/%s/%s.MP4'
epic_100_path = '/net/nfs2.prior/datasets/epickitchens/epic-kitchens-100-all/%s/videos/%s.MP4'

top_1_correct_action = 0
top_5_correct_action = 0
top_1_correct_verb = 0
top_5_correct_verb = 0
top_1_correct_noun = 0
top_5_correct_noun = 0

# if there exist in the cache, get the latest one.
cache_list = glob.glob(segment_path+'/*.pkl')
total_cache_num = len(cache_list)

count = 0
with tqdm(range(total_cache_num)) as titer:
    for idx in titer:
        data = pickle.load(open(cache_list[idx], 'rb'))
        sample = data['info']
        video_segments = data['video_segments']

        for i in range(0,7):
            video_segments[i]['use_text_as_input'] = False

        if num_observed_segments == 6:
            video_segments[6]['frame'] *= 0
            video_segments[7]['frame'] *= 0

        video_segments[7]['text'] = 'My next action is <|MASK|>'
        video_segments[7]['use_text_as_input'] = True

        # mask the audio.
        # for i in range(0, 8):
            # video_segments[i]['spectrogram'] *=0
            # video_segments[i]['use_text_as_input'] = True

        video_pre = preprocess_video(video_segments, output_grid_size=grid_size, verbose=False)
        # # Now we embed the entire video and extract the text. result is  [seq_len, H]. we extract a hidden state for every
        # # MASK token
        out_h = model.embed_video(**video_pre)
        out_h = out_h[video_pre['tokens'] == MASK]
        action_list = [v for _, v in ids_to_action.items()]
        label_space = model.get_label_space(action_list)
        logits = 100.0 * jnp.einsum('bh,lh->bl', out_h, label_space)
        idx = jnp.argsort(-logits)

        top_1_predict_action = [action_list[idx[0,0]]]
        top_5_predict_action = [action_list[idx[0,i]] for i in range(5)]
        top_1_predict_noun = [top_1_predict_action[0].split(' ')[1]]
        top_1_predict_verb = [top_1_predict_action[0].split(' ')[0]]
        
        top_5_predict_noun = []
        i = 0
        while len(top_5_predict_noun) < 5:
            prediction = action_list[idx[0,i]]
            noun = prediction.split(' ')[1]
            if noun not in top_5_predict_noun:
                top_5_predict_noun.append(noun)
            i += 1

        top_5_predict_verb = []
        i = 0
        while len(top_5_predict_verb) < 5:
            prediction = action_list[idx[0,i]]
            verb = prediction.split(' ')[0]
            if verb not in top_5_predict_verb:
                top_5_predict_verb.append(verb)
            i += 1

        gt_action =  ids_to_action[sample['action'][2]]
        if gt_action in top_1_predict_action:
            top_1_correct_action += 1
        
        if gt_action in top_5_predict_action:
            top_5_correct_action += 1
        
        gt_verb =  ids_to_verb[sample['action'][0]]
        if gt_verb in top_1_predict_verb:
            top_1_correct_verb += 1
        
        if gt_verb in top_5_predict_verb:
            top_5_correct_verb += 1

        gt_noun =  ids_to_noun[sample['action'][1]]
        if gt_noun in top_1_predict_noun:
            top_1_correct_noun += 1
        
        if gt_noun in top_5_predict_noun:
            top_5_correct_noun += 1

        count += 1
        titer.set_postfix(
            a_1 = top_1_correct_action / count,
            a_5 = top_5_correct_action / count,
            n_1 = top_1_correct_noun / count, 
            n_5 = top_5_correct_noun / count, 
            v_1 = top_1_correct_verb / count,
            v_5 = top_5_correct_verb / count,
        )

    top_1_correct_action = top_1_correct_action / total_cache_num
    top_5_correct_action = top_5_correct_action / total_cache_num

    top_1_correct_verb = top_1_correct_verb / total_cache_num
    top_5_correct_verb = top_5_correct_verb / total_cache_num

    top_1_correct_noun = top_1_correct_noun / total_cache_num
    top_5_correct_noun = top_5_correct_noun / total_cache_num


    print('loading from %s' %(segment_path))

    print('action', top_1_correct_action, top_5_correct_action)
    print('verb', top_1_correct_verb, top_5_correct_verb)
    print('noun', top_1_correct_noun, top_5_correct_noun)

pdb.set_trace()