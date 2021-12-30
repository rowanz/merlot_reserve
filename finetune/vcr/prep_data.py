"""
Turn the merged data into some tfrecord files.
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

parser = create_base_parser()
parser.add_argument(
    '-data_dir',
    dest='data_dir',
    default='/home/rowan/datasets2/vcr1',
    type=str,
    help='Image directory.'
)
"""
Must set things up like this
(vcr test is darpa)
drwxrwxr-x 2337 rowan rowan     122880 Dec  3  2018 vcr1images
drwxr-xr-x    3 rowan rowan       4096 Jul  7  2020 vcr-test
-rw-rw-r--    1 rowan rowan  392317321 Dec  3  2018 train.jsonl
-rw-rw-r--    1 rowan rowan   48969062 Dec  3  2018 val.jsonl
-rw-rw-r--    1 rowan rowan   34097024 Dec  3  2018 test.jsonl
"""

args = parser.parse_args()
random.seed(args.seed)

out_fn = os.path.join(args.base_fn, 'vcr', '{}{:03d}of{:03d}.tfrecord'.format(args.split, args.fold, args.num_folds))
detokenizer = MosesDetokenizer(lang='en')

def draw_boxes_on_image(image, metadata, tokenl_to_names, flip_lr=False):
    """
    Draw boxes on the image
    :param image:
    :param metadata:
    :param tokenl_to_names:
    :return:
    """
    #####################################################
    # last draw boxes on images
    image_copy = deepcopy(image)
    scale_factor = image.size[0] / metadata['width']

    boxes_to_draw = sorted(set([z for x in tokenl_to_names.keys() for z in x]))
    font_i = ImageFont.truetype(font='/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf', size=17)

    for i in boxes_to_draw:
        name_i = tokenl_to_names[tuple([i])]
        box_i = np.array(metadata['boxes'][i][:4]) * scale_factor
        color_hash = int(hashlib.sha256(name_i.encode('utf-8')).hexdigest(), 16)

        # Hue between [0,1],
        hue = (color_hash % 1024) / 1024
        sat = (color_hash % 1023) / 1023

        # luminosity around [0.5, 1.0] for border
        l_start = 0.4
        l_offset = ((color_hash % 1025) / 1025)
        lum = l_offset * (1.0 - l_start) + l_start
        txt_lum = l_offset * 0.1

        color_i = tuple((np.array(colorsys.hls_to_rgb(hue, lum, sat)) * 255.0).astype(np.int32).tolist())
        txt_colori = tuple((np.array(colorsys.hls_to_rgb(hue, txt_lum, sat)) * 255.0).astype(np.int32).tolist())

        x1, y1, x2, y2 = box_i.tolist()
        if flip_lr:
            x1_tmp = image_copy.width - x2
            x2 = image_copy.width - x1
            x1 = x1_tmp

        shape = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]

        draw = ImageDraw.Draw(image_copy, mode='RGBA')
        # draw.line(shape, fill=color_i, width=3)
        draw.rectangle([(x1, y1), (x2, y2)], fill=color_i + (32,), outline=color_i + (255,), width=2)
        txt_w, txt_h = font_i.getsize(name_i)

    return image_copy


def iterate_through_examples():
    if args.split not in ('train', 'val', 'test'):
        raise ValueError("unk split")
    with open(os.path.join(args.data_dir, args.split + '.jsonl'), 'r') as f:
        for idx, l in enumerate(f):
            if idx % args.num_folds != args.fold:
                continue
            if (idx // args.num_folds) % 100 == 0:
                print(f'On image {idx}')
            item = json.loads(l)

            with open(os.path.join(args.data_dir, 'vcr1images', item['metadata_fn']), 'r') as f:
                metadata = json.load(f)

            image = Image.open(os.path.join(args.data_dir, 'vcr1images', item['img_fn']))
            image = resize_image(image, shorter_size_trg=450, longer_size_max=800)

            ######################################################################
            # Tie tokens with names
            # the metadata file has the names only, ie
            # ['person', 'person', 'person', 'car']

            # questions refer to this through an index, ie
            # 2 for person3
            tokenl_to_names = {}
            type_to_ids_globally = defaultdict(list)
            object_count_idx = []

            for obj_id, obj_type in enumerate(metadata['names']):
                object_count_idx.append(len(type_to_ids_globally[obj_type]))
                type_to_ids_globally[obj_type].append(obj_id)

            for obj_type, obj_ids in type_to_ids_globally.items():
                if len(obj_ids) == 1:
                    # Replace all mentions with that single object ('car') with "the X"
                    tokenl_to_names[tuple(obj_ids)] = f'the {obj_type}'

            def get_name_from_idx(k):
                """
                If k has a length of 1: we're done
                otherwise, recurse and join
                :param k: A tuple of indices
                :return:
                """
                if k in tokenl_to_names:
                    return tokenl_to_names[k]

                if len(k) == 1:
                    obj_type = metadata['names'][k[0]]
                    obj_idx = object_count_idx[k[0]]
                    name = '{} {}'.format(obj_type.capitalize(), obj_idx+1)
                    tokenl_to_names[k] = name
                    return name

                names = [get_name_from_idx(tuple([k_sub])) for k_sub in k]

                if len(names) <= 2:
                    names = ' and '.join(names)
                else:
                    # who gives a fuck about an oxford comma
                    names = ' '.join(names[:-2]) + ' ' + ' and '.join(names[-2:])

                tokenl_to_names[k] = names
                return names

            def fix_token(tok):
                """
                Fix token that's either a list (of object detections) or a word
                :param tok:
                :return:
                """
                if not isinstance(tok, list):
                    # just in case someone said `Answer:'. unlikely...
                    if tok != 'Answer:':
                        return tok.replace(':', ' ')
                    return tok
                return get_name_from_idx(tuple(tok)[:2])

            def fix_tokenl(token_list):
                out = detokenizer.detokenize([fix_token(tok) for tok in token_list])
                out = re.sub(" n't", "n't", out)
                out = re.sub("n' t", "n't", out)

                # remove space before some punctuation
                out = re.sub(r'\s([\',\.\?])', r'\1', out)
                # fix shit like this: `he' s writing.`"
                out = re.sub(r'\b\'\ss', "'s", out)

                # kill some punctuation
                out = re.sub(r'\-\;', ' ', out)

                # remove extra spaces
                out = re.sub(r'\s+', ' ', out.strip())
                return out

            qa_query = fix_tokenl(item['question'])
            qa_choices = [fix_tokenl(choice) for choice in item['answer_choices']]
            qar_choices = [fix_tokenl(choice) for choice in item['rationale_choices']]

            img_boxes = draw_boxes_on_image(image, metadata, tokenl_to_names)
            if 'answer_label' in item:
                qar_query = '{} Answer: {}'.format(qa_query, qa_choices[item['answer_label']])

                # flip boxes
                everything = ' '.join(qa_choices + qar_choices + [qa_query])
                if ('right' in everything) or ('left' in everything):
                    img_lr = img_boxes
                else:
                    img_lr = draw_boxes_on_image(image, metadata, tokenl_to_names, flip_lr=True)

                yield {'qa_query': qa_query, 'qa_choices': qa_choices, 'qa_label': item['answer_label'],
                       'qar_query': qar_query, 'qar_choices': qar_choices, 'qar_label': item['rationale_label'],
                       'id': str(item['annot_id']), 'image': img_boxes, 'image_fliplr': img_lr,
                       }
            else:
                # Test set
                assert len(item['answer_choices']) == 4
                for i, qa_choice_i in enumerate(qa_choices):
                    qar_query = '{} Answer: {}'.format(qa_query, qa_choice_i)
                    yield {'qa_query': qa_query, 'qa_choices': qa_choices, 'qa_label': 0,
                           'qar_query': qar_query, 'qar_choices': qar_choices, 'qar_label': 0,
                           'id': '{}-qar-conditioned_on_a{}'.format(item['annot_id'], i), 'image': img_boxes
                           }

num_written = 0
max_len = 0
with GCSTFRecordWriter(out_fn, auto_close=False) as tfrecord_writer:
    for j, ex in enumerate(iterate_through_examples()):
        feature_dict = {
            'image': bytes_feature(pil_image_to_jpgstring(ex['image'])),
            'image_fliplr': bytes_feature(pil_image_to_jpgstring(ex.get('image_fliplr', ex['image']))),
            'id': bytes_feature(ex['id'].encode('utf-8')),
        }

        for prefix in ['qa', 'qar']:
            query_enc = encoder.encode(ex[f'{prefix}_query']).ids
            feature_dict[f'{prefix}_query'] = int64_list_feature(query_enc)
            for i, choice_i in enumerate(encoder.encode_batch(ex[f'{prefix}_choices'])):
                feature_dict[f'{prefix}_choice_{i}'] = int64_list_feature(choice_i.ids)
                max_len = max(len(choice_i.ids) + len(query_enc), max_len)
            feature_dict[f'{prefix}_label'] = int64_feature(ex[f'{prefix}_label'])

            if j < 20:
                print(f"~~~~~~~~~~~ Example {prefix} {j} {ex['id']} ~~~~~~~~")
                print(encoder.decode(feature_dict[f'{prefix}_query'].int64_list.value, skip_special_tokens=False), flush=True)
                for i in range(4):
                    toks = feature_dict[f'{prefix}_choice_{i}'].int64_list.value
                    toks_dec = encoder.decode(toks, skip_special_tokens=False)
                    lab = ' GT' if i == ex[f'{prefix}_label'] else '   '
                    print(f'{i}{lab}) {toks_dec}     ({len(toks)}tok)', flush=True)
                # ex['image'].save(f'ex-{j}.jpg', quality=95)

        example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
        tfrecord_writer.write(example.SerializeToString())
        num_written += 1
    tfrecord_writer.close()

print(f'Finished writing {num_written} questions; max len = {max_len}', flush=True)
