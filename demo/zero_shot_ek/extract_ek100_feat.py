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
# this is fixed for action anticipation tasks.
tau_a = 1.0
# this has to be divided by 5.
num_observed_segments = 6
time_interval = 5.0 # 2.0
tau_o = time_interval * num_observed_segments# 12.0

# ablations we want to try.
# num_observed_segments = 8 vs 6.
# time_interval = 1.0, 2.0, 5.0
# with audio / without audio
# different prompt (noun the verb / noun verb) / training noun verb stats.

assert num_observed_segments * time_interval == tau_o

segment_path = 'cache/segments_ek100_n{}_t{}'.format(num_observed_segments, time_interval)
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

        action = verb + ' the ' + noun
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

top_1_correct = 0
top_5_correct = 0

# if there exist in the cache, get the latest one.
cache_list = glob.glob(segment_path+'/*.pkl')
cache_id = len(cache_list)

outputs = []
for id, sample in enumerate(samples):
    if id <= cache_id:
        continue
    # action antifipation.
    start_time = sample['anticipation_se'][0]
    end_time = sample['anticipation_se'][1]
    path1 = epic_55_path %(sample['video_id'].split('_')[0], sample['video_id'])
    path2 = epic_100_path %(sample['video_id'].split('_')[0], sample['video_id'])

    if os.path.exists(path1):
        video_path = path1
    else:
        video_path = path2

    times = []
    et = sample['anticipation_se'][1]
    # get the segment for observed video. 
    for i in range(num_observed_segments):
        st = round(max(et - time_interval, 0),2)
        times.insert(0, {'start_time': st, 'end_time': et, 'mid_time': round((st + et) / 2.0, 2)})
        et = st

    if num_observed_segments == 6:
        st = round(sample['anticipation_se'][1],2)
        et = round(sample['action_se'][0],2)
        times.append({'start_time': st, 'end_time': et, 'mid_time': round((st + et) / 2.0, 2)})

        st = round(sample['action_se'][0],2)
        et = round(sample['action_se'][1],2)
        times.append({'start_time': st, 'end_time': et, 'mid_time': round((st + et) / 2.0,2)})  
    
    video_segments = video_to_segments_zero_shot(video_path, time_interval=time_interval, times=times)

    for i in range(0,7):
        video_segments[i]['use_text_as_input'] = False

    if num_observed_segments == 6:
        video_segments[6]['frame'] *= 0
        video_segments[7]['frame'] *= 0

    video_segments[7]['text'] = '<|MASK|>'
    video_segments[7]['use_text_as_input'] = True
    
    # save the segments.
    segments = {'video_segments':video_segments, 'info':sample}
    pickle.dump(segments, open(os.path.join(segment_path, '%05d.pkl' %id), 'wb'))

    print(id)