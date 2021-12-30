"""
Demo for doing interesting things with a video
"""
import os
import sys
import json
import pickle
from tqdm import tqdm
sys.path.append('../')

from mreserve.preprocess import video_to_segments, preprocess_video, encoder, MASK
from mreserve.modeling import PretrainedMerlotReserve
import jax
import jax.numpy as jnp

# This handles loading the model and getting the checkpoints.
grid_size = (18, 32)
time_interval = 3.0
model = PretrainedMerlotReserve.from_pretrained(model_name='large', image_grid_size=grid_size)

split = 'test'
factor = 0
print(f'{split}_infill.json.{factor}')
vocab = json.load(open('MSRVTT-QA/vocab_1k.json', 'r'))

segment_path = '/net/nfs2.mosaic/mreserve/msrvtt-qa/cache/segments_msrvtt_t{}'.format(time_interval)

ds = json.load(open(f'MSRVTT-QA/{split}_infill.json.{factor}', 'r'))

top1_acc, top5_acc, top10_acc = [], [], []
for item in tqdm(ds):
    annot_id = item["id"]
    feature_file = os.path.join(segment_path, f'{annot_id}.pkl')

    if not os.path.exists(feature_file):
        # if no cached feature, automatically count it as wrong
        top1_acc.append(0)
        top5_acc.append(0)
        top10_acc.append(0)
        continue

    try:
        segments = pickle.load(open(feature_file, 'rb'))
        video_segments, info = segments['video_segments'], segments['info']
        video_segments[-1]['text'] = video_segments[-1]['text'].replace(' <|MASK|>', '<|MASK|>')
        answer = info['answer']

        video_pre = preprocess_video(video_segments, output_grid_size=grid_size, verbose=False)

        # Now we embed the entire video and extract the text. result is  [seq_len, H]. we extract a hidden state for every
        # MASK token
        out_h = model.embed_video(**video_pre)
        out_h = out_h[video_pre['tokens'] == MASK]

        options = vocab
        # the following is all the labels from activitynet. why not! some of them don't make sense grammatically though.
        label_space = model.get_label_space(options)

        # Dot product the <|MASK|> tokens and the options together
        logits = 100.0 * jnp.einsum('bh,lh->bl', out_h, label_space)
        idx = jnp.argsort(-logits)

        top_1_predict = [options[idx[0, 0]]]
        top_5_predict = [options[idx[0, i]] for i in range(5)]
        top_10_predict = [options[idx[0, i]] for i in range(10)]

        top1_acc.append(int(answer in top_1_predict))
        top5_acc.append(int(answer in top_5_predict))
        top10_acc.append(int(answer in top_10_predict))
    except:
        continue


print(f'top1-acc: {round(sum(top1_acc) / len(top1_acc) * 100., 3)}')
print(f'top5-acc: {round(sum(top5_acc) / len(top5_acc) * 100., 3)}')
print(f'top10-acc: {round(sum(top10_acc) / len(top10_acc) * 100., 3)}')

json.dump({'top1-acc': top1_acc, 'top5-acc': top5_acc, 'top10-acc': top10_acc}, open(f'acc.json.{factor}', 'w'))
