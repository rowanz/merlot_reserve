import sys
import os

os.environ['TOKENIZERS_PARALLELISM'] = 'False'
sys.path.append('../')

import argparse
import csv
import tempfile
import hashlib
import json
import numpy as np
import time
from data.video_utils import extract_all_frames_from_video, extract_frames_from_video, clean_description, \
    clean_subtitle_tuples, align_using_dtw, make_spectrogram, make_jpg_spectrograms, _invert_jpg_spectrogram
from data.data_utils import *
import string
import pandas as pd
import shutil
import atexit
import gzip
from data.youtube_utils import read_vtt_text
from mreserve.lowercase_encoder import get_encoder
import subprocess
import librosa
import scipy.signal.windows
from data.offset_model import predict_offsets, get_features
from scipy.io import wavfile
import torchvision.models as models
import torch
import torchvision.transforms as transforms
import zstandard
import io
import regex as re
from data.clean_text import clean_text
from google.cloud import storage


parser = argparse.ArgumentParser(description='Convert downloaded files to TFRecord format')
parser.add_argument(
    '-bucket_name',
    dest='bucket_name',
    type=str,
    help='Bucket name to use.'
)
parser.add_argument(
    '-fold',
    dest='fold',
    default=0,
    type=int,
    help='which fold we are on'
)
parser.add_argument(
    '-num_folds',
    dest='num_folds',
    default=32768,
    type=int,
    help='Number of folds (corresponding to both the number of training files and the number of testing files)',
)
parser.add_argument(
    '-ids_fn',
    dest='ids_fn',
    default='test_ids_fn.csv',
    type=str,
    help='We will use these IDs. you probably should filter them to mkae sure they all at least have the right files. can start with gs://'
)
parser.add_argument(
    '-out_folder',
    dest='out_folder',
    default="./",
    type=str,
    help='Output folder to use. You can start this with gs:// and we\'ll put it on google cloud.'
)
parser.add_argument(
    '-shuffle_fns',
    type=bool,
    default=False,
    help='Shuffle the filenames that we load'
)
parser.add_argument(
    '-num_chunks',
    dest='num_chunks',
    default=16,
    type=int,
    help='Number of chunks in each tfrecord',
)
parser.add_argument(
    '-split_name',
    dest='split_name',
    default='train',
    type=str,
    help='train or val'
)
parser.add_argument(
    '-seed',
    dest='seed',
    default=123456,
    type=int,
    help='Number of chunks in each tfrecord',
)
parser.add_argument(
    '-log_folder',
    dest='log_folder',
    default="./",
    type=str,
    help='Log folder to use. You can start this with gs:// and we\'ll put it on google cloud.'
)
parser.add_argument(
    '-ckpt',
    dest='ckpt',
    default='mobilenetv2_filter_model_coco_82ptacc.pth.tar',
    type=str,
    help='checkpoint location. The checkpoint we used is at gs://merlot/video_filter_cnn/mobilenetv2_filter_model_coco_82ptacc.pth.tar - you might want to download that first'
)
parser.add_argument(
    '-max_acs',
    dest='max_acs',
    default=0.85,
    type=float,
    help='Maximum average cosine similarity',
)
parser.add_argument(
    '-min_nco',
    dest='min_nco',
    default=1.0,
    type=float,
    help='Min num coco objects',
)
parser.add_argument(
    '-num_text_seqs',
    dest='num_text_seqs',
    default=2,
    type=int,
    help='Number of text sequences. Must be <= num_chunks, also tune this such that we never run out',
)
parser.add_argument(
    '-text_len',
    dest='text_len',
    default=1536,
    type=int,
    help='Length per text',
)

args = parser.parse_args()

gclient = storage.Client()
bucket = gclient.get_bucket(args.bucket)
encoder = get_encoder()

NUM_CHUNKS = args.num_chunks
NUM_MELS = 64

###########################################
# MEGA_WINDOW_SIZE = 10.0
# # Let's say we want a 10 second mega-window and 7 chunks. The extra 1.25sec can be missing for
# # data augmentation purposes (random crop?) or we can do 8 chunks, that's good too
# # So the small size is 1.25
# # need 1 + (22050 * t_delta) / num_hops = 64
# # So then  (22050 * t_delta) / 63 = num_hops
# NUM_HOPS = 437
# NUM_FFT = 1280  # Try around 2.5x NUM_HOPS but if you round to around a power of 2 it goes faster
# random.seed(args.seed)
# # Consider merging if fewer than this many tokens in a 12 sec window
# MIN_TOKS_WINDOW = 10
# OK_TOKS_MULTIWINDOW = 30  # If N windows would have this many tokens, then break (yielding a short window)
############################################

MEGA_WINDOW_SIZE = 5.0
# Let's say we want a 5 second mega-window and 3 chunks. Take out some 0.2sec as padding

# Need 1 + (22050 * t_delta) / num_hops = 60
# (22050 * t_delta) / 60 = num_hops
# IT WORKS PERFECTLY
NUM_HOPS=588
NUM_FFT=1536 # This sounds better

# # Consider merging if fewer than this many tokens in a 12 sec window
MIN_TOKS_WINDOW = 8
OK_TOKS_MULTIWINDOW = 16 # If N windows have this many tokens then break


if args.ckpt is not None:
    # Load mobilenet model
    model = models.MobileNetV2(num_classes=81)
    model.load_state_dict({k[7:]: v for k, v in torch.load(args.ckpt,
                                                           map_location=torch.device('cpu'))['state_dict'].items()})
    model.features[0][0].padding = (0, 0)
    model.features[0][0].stride = (1, 1)  # Now it expects [114, 114] images
    model.eval()


STORAGE_DIR = tempfile.mkdtemp()


def _cleanup():
    if os.path.exists(STORAGE_DIR):
        shutil.rmtree(STORAGE_DIR)


atexit.register(_cleanup)


def load_video(video_id):
    """
    Loads video from GCS
    :param video_id:
    :return: a video OR none
    """
    start = time.time()
    try:
        info_fn = os.path.join(STORAGE_DIR, 'info.json.gz')
        iblob = bucket.blob(f'youtube_dump/{video_id}/{video_id}.v2.info.json.gz')
        if not iblob.exists():
            return None
        iblob.download_to_filename(info_fn)
        with gzip.open(info_fn, 'r') as f:
            item = json.load(f)

        if 'title' not in item:
            raise ValueError(f"'title' not in item \n{item}")

        # Get transcript - not using Grover for now
        if 'transcripts' not in item:
            return None
        transcripts = {}
        for k, v in item['transcripts'].items():
            try:
                ts_k = read_vtt_text(v.splitlines(), skip_if_no_timing_info=True)
                if ts_k is not None:
                    transcripts[k] = clean_subtitle_tuples(ts_k)
            except (ValueError, KeyError, AttributeError) as e:
                print(str(e))
        if 'en' not in transcripts:
            raise ValueError(f"'en' not in item['transcripts'] \n{item}")
        item['transcripts'] = transcripts

        vtt = pd.DataFrame(item['transcripts']['en'])
        if (vtt.shape[0] == 0) or ('word' not in vtt.columns):
            raise ValueError(f"'Word' not in item['transcripts'] \n{item}")

        # A few times we failed to download automatic subtitles, or downloaded manual ones instead, due to a bug in the script
        # they usually suck, e.g. https://www.youtube.com/watch?v=DqqzX-3bW6A, let's take out the bad ones
        def _token_is_good(tok):
            if len(tok) > 1 and tok.isupper():
                return False
            if '\n' in tok:
                return False
            if ' ' in tok:
                return False
            return True
        tok_is_good = vtt['word'].apply(_token_is_good)
        if tok_is_good.mean() < 0.6:
            raise ValueError("{} has jarbled tokens".format(item['id']))
        len_variance = vtt['word'].apply(len).var()
        if len_variance > 10.0:
            raise ValueError("{} has a length variance of {:.3f}".format(item['id'], len_variance))
        item['transcript_vtt'] = vtt

        video_fn = os.path.join(STORAGE_DIR, 'video.mp4')
        vblob = bucket.blob(f'youtube_dump/{video_id}/{video_id}.mp4')
        if not vblob.exists():
            return None
        vblob.download_to_filename(video_fn)

        # Make sure if we have audio
        stream_txt = subprocess.run(f'ffprobe -i {video_fn} -show_streams -select_streams a -loglevel error',
                                    capture_output=True, shell=True, text=True).stdout
        if len(stream_txt) == 0 or 'codec_type=audio' not in stream_txt:
            return None
        item['_te'] = time.time() - start
        return item
    except (Exception, StopIteration) as e:
        print(str(e), flush=True)
        return None


def video_iterator():
    channels_video_ids = []
    print("LOADING IDS", flush=True)
    with tf.io.gfile.GFile(args.ids_fn, 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i % args.num_folds == args.fold:
                channels_video_ids.append(row['video_id'])
    if args.shuffle_fns:
        random.shuffle(channels_video_ids)
    print("GOT THE VIDEO IDS - {} in total".format(len(channels_video_ids)), flush=True)
    if len(channels_video_ids) == 0:
        time.sleep(5.0)  # race condition? idk
        raise ValueError("Couldnt load video ids")
    for video_id in channels_video_ids:
        video = load_video(video_id)
        if video is not None:
            yield video

def get_librosa_params(sr, playback_speed):
    params = {
        'sr': sr,
        'n_mels': 64,
        'n_fft': NUM_FFT * playback_speed,
        'hop_length': NUM_HOPS * playback_speed,
        'window': scipy.signal.windows.hann,
        'fmin': 20.0,
        'fmax': 11025.0,  # Half the sample rate
        'eps': 1e-1,
    }
    return params


def split_video_into_chunks(item):
    """
    :param item
    :return:
    """
    vtt = item['transcript_vtt']

    vtt['encoded'] = [x.ids for x in encoder.encode_batch(vtt['word'].tolist())]
    get_features(vtt)

    # [n_rows, (offset on start, offset on end)]
    offsets = predict_offsets(vtt)

    # Make the offsets globally consistent
    deltas = np.concatenate([[offsets[0, 0]], (offsets[1:, 0] + offsets[:-1, 1]) / 2.0, [offsets[-1, 1]]], 0)
    deltas = np.clip(deltas, a_min=-0.5, a_max=0.5)

    ##################
    vtt['start'] += deltas[:-1]
    vtt['end'] += deltas[1:]
    vtt['center'] = (vtt['start'] + vtt['end']) / 2.0

    ###############################
    # Perform a sliding window over MEGA_WINDOW_SIZE
    # Anything in the window that is too slow we will increase rate by 2x or 3x
    audio_chunks = []
    start_time = max(vtt.iloc[0]['start'] - 0.5 * random.random() * MEGA_WINDOW_SIZE, 0.0)
    start_time = round(start_time, 2)
    max_time = item['duration'] - 1
    idx = 0
    while (idx < vtt.shape[0]) and (start_time + MEGA_WINDOW_SIZE) <= max_time:

        # 1. See how many things are in start_time + Delta
        for playback_speed in range(1, 4):
            delta = MEGA_WINDOW_SIZE * playback_speed
            t_end = start_time + delta
            inds = (vtt['center'].values < t_end) & (np.arange(vtt.shape[0]) >= idx)
            inds = np.where(inds)[0]

            # Case 1: have enough tokens
            if inds.size >= MIN_TOKS_WINDOW:
                break

            # Case 2: We are at the end
            if (t_end + MEGA_WINDOW_SIZE) > max_time:
                break

            # Check if the next window has enough words
            inds_2d = (vtt['center'].values < (t_end + MEGA_WINDOW_SIZE)) & (np.arange(vtt.shape[0]) >= idx)
            if np.sum(inds_2d) >= OK_TOKS_MULTIWINDOW:
                break

            # Case 3: randomly break
            if random.random() > 0.9:
                break

        end_time = round(start_time + delta, 2)
        current_audio_chunk = {
            'start_time': start_time,
            'end_time': end_time,
            'playback_speed': playback_speed,
            'rows': inds.tolist(),
        }
        audio_chunks.append(current_audio_chunk)
        start_time = end_time
        if len(inds) > 0:
            idx = int(inds[-1]) + 1

    if len(audio_chunks) == 0:
        raise ValueError('chunks empty!')

    nwords = [len(x['rows']) for x in audio_chunks]
    if args.debug:
        print('duration = {:.3f}. {} audio chunks. #words mean: {:.3f} words max {:2d} words std {:.3f}'.format(
            vtt.iloc[-1]['end'], len(audio_chunks), np.mean(nwords), max(nwords), np.std(nwords)), flush=True)
        for i, c in enumerate(audio_chunks):
            # Get the mean timestep, rounded to an int.
            txt = '{:03d} [{:.1f}, {:.1f}] {}'.format(i, c['start_time'], c['end_time'],
                                                      ' '.join(vtt.loc[c['rows'], 'word']))
            print(txt, flush=True)
        print('----', flush=True)
    return audio_chunks, vtt


def video_chunk_iterator():
    for item in video_iterator():
        try:
            chunks, vtt = split_video_into_chunks(item)
        except (ValueError, KeyError) as e:
            print('{}\n{}'.format(str(e), item), flush=True)
            continue

        # Load audio in background
        audio_fn = os.path.join(STORAGE_DIR, 'audio.wav')
        video_fn = os.path.join(STORAGE_DIR, 'video.mp4')
        ffmpeg_process = subprocess.Popen(['ffmpeg', '-y', '-i', video_fn, '-ac', '1', '-ar', '22050',
                                           audio_fn,
                                           ],
                                          stdout=-1, stderr=-1, text=True)

        timesteps = [(x['start_time'] + x['end_time']) / 2.0 for x in chunks]

        # Extract frames at each chunk
        frames = extract_frames_from_video(video_file=os.path.join(STORAGE_DIR, 'video.mp4'),
                                           times=timesteps, use_multithreading=True, info=item)
        if frames is None:
            print("Couldn't extract frames from video {}".format(item['id']), flush=True)
            continue
        trg_size = get_size_for_resize((frames.shape[2], frames.shape[1]), shorter_size_trg=288,
                                       longer_size_max=512)
        for i, frame_i in enumerate(frames):
            img_i = Image.fromarray(frame_i, mode='RGB')
            if trg_size != img_i.size:
                img_i = img_i.resize(trg_size, resample=Image.BICUBIC)
            chunks[i]['frame'] = img_i

        ############################
        # Now load audio
        # # Extract audio frames
        audio_fn = os.path.join(STORAGE_DIR, 'audio.wav')
        try:
            stdout, stderr = ffmpeg_process.communicate(None, timeout=5.0)
        except subprocess.TimeoutExpired:
            ffmpeg_process.kill()
            stdout, stderr = subprocess.TimeoutExpired.communicate()
            raise ValueError("couldnt convert in time")
        except:  # Keyboardinterrupt
            ffmpeg_process.kill()
            raise
        ffmpeg_process.kill()

        sr, waveform = wavfile.read(audio_fn, mmap=True)
        waveform = waveform.astype('float32')
        waveform /= max(np.abs(waveform).max(), 1.0)

        # Pad to max time just in case
        desired_final_frame = int(sr * max([x['end_time'] for x in chunks]))
        if waveform.size < desired_final_frame:
            waveform = np.concatenate([waveform, np.zeros(desired_final_frame-waveform.size, dtype=np.float32)], 0)

        # Avoid annoying float roundoff
        delta = int(sr * MEGA_WINDOW_SIZE)
        waveforms = []
        for x in chunks:
            start_idx = int(sr * x['start_time'])
            end_idx = start_idx + delta * x['playback_speed']
            waveforms.append(waveform[start_idx:end_idx])

        params_list = [get_librosa_params(sr, playback_speed=chunk['playback_speed']) for chunk in chunks]

        spec_size = int((params_list[0]['sr'] * MEGA_WINDOW_SIZE * chunks[0]['playback_speed']) / (
            params_list[0]['hop_length'])) + 1
        specs = make_jpg_spectrograms(waveforms, params_list, use_multithreading=True,
                                      expected_size=spec_size)
        for i, (spec_i, spectrogram_magic_number) in enumerate(specs):
            chunks[i]['spectrogram'] = spec_i
            chunks[i]['spectrogram_width'] = spec_size
            chunks[i]['spectrogram_magic_number'] = spectrogram_magic_number

        # Get everything needed for chunks to work on their own
        # dict_keys(['start_time', 'end_time', 'playback_speed', 'rows', 'frame', 'spectrogram', 'spectrogram_width'])

        description = encoder.encode(item['description']).ids
        title = encoder.encode(item['title']).ids
        tags = encoder.encode(', '.join(item['tags'])).ids

        meta_info = {k: item[k] for k in ['channel_id', 'view_count', 'average_rating',
                                          '_avg_cosine_sim', '_num_coco_objects_expectation', 'upload_date',
                                          'categories', '_ids_fn'] if k in item}

        for i, chunk in enumerate(chunks):
            df = vtt.iloc[chunk.pop('rows')]

            start_times = []
            end_times = []
            bpe_tokens = []
            for _, row in df.iterrows():
                st = (row['start'] - chunk['start_time']) / chunk['playback_speed']
                et = (row['end'] - chunk['start_time']) / chunk['playback_speed']
                for tok in row['encoded']:
                    start_times.append(st)
                    end_times.append(et)
                    bpe_tokens.append(tok)

            chunk['tok_start_times'] = start_times
            chunk['tok_end_times'] = end_times
            chunk['tok_ids'] = bpe_tokens

            chunk['meta'] = meta_info
            chunk['youtube_id'] = item['id']
            chunk['description'] = description
            chunk['title'] = title
            chunk['tags'] = tags
        yield chunks


def grouped_iterator(iterator, group_size, max_items=100, pop_from_front_prob=0.8):
    """
    Try to group together short sequences
    :param iterator: Iterator returning sequences
    :param group_size:
    :param max_items:
    :return:
    """
    buffer = {}

    def _random_slice(list_to_slice, amount):
        if pop_from_front_prob > random.random():  # 80% of the time pop from front
            piece = list_to_slice[:amount]
            return piece, list_to_slice[amount:]
        else:
            piece = list_to_slice[-amount:]
            return piece, list_to_slice[:-amount]

    def _pop():
        # Prioritize longest
        k_to_len = {k: len(c) for k, c in buffer.items()}
        keys_in_order = sorted(k_to_len.items(), key=lambda x: -x[1])
        # print(f"Time to pop, keys={keys_in_order}", flush=True)
        # Start us off
        k0, l0 = keys_in_order[0]

        # Pop biggest and that's enough - probably this won't happen
        if l0 > group_size:
            to_yield, buffer[k0] = _random_slice(buffer[k0], group_size)
            return to_yield

        # print(f"Popping the TOP one ({k0}, {l0})")
        to_yield = buffer.pop(k0)

        # See if we can scoop up smaller things
        for k1, l1 in keys_in_order[1:]:
            if l1 <= (group_size - len(to_yield)):
                # print(f"len ={len(to_yield)}. Scooping up ({k1}, {l1})")
                to_yield += buffer.pop(k1)

        # If needed destroy something at random
        while len(to_yield) < group_size:
            if len(buffer) == 0:
                # print("Empty buffer! exit", flush=True)
                return None

            random_k = random.choice(sorted(buffer.keys()))
            random_l = len(buffer[random_k])
            l_needed = min(group_size - len(to_yield), random_l)

            # print(f"len ={len(to_yield)}. partially popping ({random_k}, {random_l})")
            piece, buffer[random_k] = _random_slice(buffer[random_k], l_needed)
            to_yield += piece
        return to_yield

    for c_i, chunk in enumerate(iterator()):
        while len(chunk) >= group_size:
            to_yield, chunk = _random_slice(chunk, group_size)
            yield to_yield
        if len(chunk) > 0:
            buffer[c_i] = chunk

        while len(buffer) > max_items:
            x = _pop()
            if x is not None:
                yield x
            else:
                print(f"WARNING: BUFFER with max_items={max_items} MIGHT NOT BE BIG ENOUGH", flush=True)

    while len(buffer) > 0:
        x = _pop()
        if x is not None:
            yield x

if args.ckpt is not None:
    my_transform = transforms.Compose([
        transforms.Resize((90, 120)),
        transforms.CenterCrop((82, 114)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

def _allpairs_cosine_similarity(x):
    """ for a matrix of size [n, d] we will compute all pairs cosine similarity and get [n,n]"""
    pairwise_numerator = x @ x.t()
    denominator_elems = torch.sqrt(torch.diag(pairwise_numerator))
    denominator = denominator_elems[None] * denominator_elems[:, None]
    cosine_sim = pairwise_numerator / denominator
    return cosine_sim

def text_iterator(num_seqs = 4, text_len=512):
    """
    This is for downloading the pile, jointly with the rest
    if not using the pile you don't need this function
    :param num_seqs:
    :param text_len:
    :return:
    """
    zst_fn = os.path.join(STORAGE_DIR, 'txt.jsonl.zst')
    file_id = args.fold % 16410

    NUM_SKIPEVERY = args.num_folds // 16410 + 1

    skip_every = (args.fold // 16410) % NUM_SKIPEVERY
    blob = bucket.blob(f'thepile/fold{file_id:05d}of16410.jsonl.zst')
    blob.download_to_filename(zst_fn)

    def sub_iterator():
        current = []
        ok_sources = set(['Pile-CC', 'FreeLaw', 'StackExchange', 'PubMed Abstracts', 'OpenWebText2', 'Wikipedia (en)',
                      'HackerNews', 'NIH ExPorter', 'USPTO Backgrounds', 'OpenSubtitles', 'Books3', 'Gutenberg (PG-19)',
                      'BookCorpus2'])

        with open(zst_fn, 'rb') as fh:
            dctx = zstandard.ZstdDecompressor()
            with dctx.stream_reader(fh, read_size=16384) as reader:
                text_stream = io.TextIOWrapper(reader, encoding='utf-8', errors='ignore')
                for j, line in enumerate(text_stream):
                    if (j % NUM_SKIPEVERY) == skip_every:
                        try:
                            X = json.loads(line)
                        except json.decoder.JSONDecodeError:
                            print("ERROR JSON DECODE", flush=True)
                            continue

                        # Options ['Pile-CC', 'FreeLaw', 'StackExchange', 'YoutubeSubtitles', 'Github',
                        # 'PubMed Abstracts', 'PubMed Central', 'OpenWebText2', 'Wikipedia (en)', 'HackerNews',
                        # 'NIH ExPorter', 'USPTO Backgrounds', 'ArXiv', 'Enron Emails', 'DM Mathematics',
                        # 'OpenSubtitles', 'Books3', 'Gutenberg (PG-19)', 'Ubuntu IRC', 'EuroParl', 'PhilPapers',
                        # 'BookCorpus2']

                        # for k, vs in story_by_meta.items():
                        #     print(k + '\n=========\n')
                        #     for v_i, v in enumerate(vs[:10]):
                        #         print(f"{v_i}) {clean_text(v)[:128]}", flush=True)
                        #     print('\n\n')

                        # story_by_meta[X['meta']['pile_set_name']].append(X['text'])
                        if X['meta']['pile_set_name'] not in ok_sources:
                            continue

                        text = clean_text(X['text'])

                        x_enc = [encoder.token_to_id('<|START|>')] + encoder.encode(text).ids
                        x_enc.append(encoder.token_to_id('<|END|>'))
                        current.extend(x_enc)

                        while len(current) >= text_len:
                            yield current[:text_len]
                            current = current[text_len:]

                        if len(current) <= (text_len // 8):
                            current = []

    buffer = []
    for seq in sub_iterator():
        buffer.append(seq)
        if len(buffer) == num_seqs:
            yield buffer
            buffer = []

    raise ValueError("Consumed text iterator too early")

def buffered_chunk_iterator():
    for chunk_group in grouped_iterator(video_chunk_iterator, group_size=NUM_CHUNKS, max_items=NUM_CHUNKS * 10):
        # Simple img recognizer
        if args.ckpt is not None:
            if random.random() > 0.9:
                with torch.no_grad():
                    imgs = torch.stack([my_transform(x['frame']) for x in chunk_group[::2]], 0)
                    features = model.features(imgs).mean([2,3])
                    cosine_sim = _allpairs_cosine_similarity(features).numpy()
                    objects = torch.sigmoid(model.classifier(features)).numpy()
                    avg_cosine_sim = float(np.tril(cosine_sim, -1).sum()) / (len(imgs) * (len(imgs) - 1.0) / 2.0)
                    youtube_id = chunk_group[0]['youtube_id']
                    if avg_cosine_sim > args.max_acs:
                        print(f"breaking ACS is {avg_cosine_sim} on {youtube_id}", flush=True)
                        continue
                    num_coco_objects_expectation = objects.max(0)
                    num_coco_objects_expectation = float(
                        num_coco_objects_expectation[num_coco_objects_expectation > 0.3].sum())
                    if num_coco_objects_expectation < args.min_nco:
                        print(f"breaking NCO is {num_coco_objects_expectation} on {youtube_id}", flush=True)
                        continue
        yield chunk_group

train_file = os.path.join(args.out_folder,
                          '{}{:05d}of{:05d}.tfrecord'.format(args.split_name, args.fold, args.num_folds))

num_written = 0
video_set = set()
tokens_written = []
st = time.time()
with GCSTFRecordWriter(train_file, buffer_size=10000, auto_close=False) as train_writer:
    for chunks, txt in zip(buffered_chunk_iterator(), text_iterator(num_seqs=args.num_text_seqs, text_len=args.text_len)):
        feats = {}
        video_idx = -1
        for i, c_i in enumerate(chunks):
            video_set.add(c_i['youtube_id'])
            is_first = i == 0 or (c_i['youtube_id'] != chunks[i - 1]['youtube_id'])
            if is_first:
                video_idx += 1

            image_encoded = pil_image_to_jpgstring(c_i['frame'], quality=75)
            tokens_written.append(len(c_i['tok_ids']))
            current_feats = {
                'image/encoded': bytes_feature(image_encoded),
                'image/height': int64_feature(c_i['frame'].height),
                'image/width': int64_feature(c_i['frame'].width),
                'image/key/sha256': bytes_feature(hashlib.sha256(image_encoded).hexdigest().encode('utf-8')),
                'image/format': bytes_feature('jpeg'.encode('utf-8')),

                'spectrogram/encoded': bytes_feature(c_i['spectrogram']),
                'spectrogram/height': int64_feature(NUM_MELS),
                'spectrogram/width': int64_feature(c_i['spectrogram_width']),
                'spectrogram/key/sha256': bytes_feature(hashlib.sha256(c_i['spectrogram']).hexdigest().encode('utf-8')),
                'spectrogram/format': bytes_feature('jpeg'.encode('utf-8')),
                'spectrogram/magic_number': float_list_feature([c_i['spectrogram_magic_number']]),

                'youtube_id': bytes_feature(c_i['youtube_id'].encode('utf-8')),
                'video_src_idx': int64_feature(video_idx),

                'title': int64_list_feature(c_i['title'] if is_first else []),
                'tags': int64_list_feature(c_i['tags'] if is_first else []),
                'description': int64_list_feature(c_i['description'] if is_first else []),
                'meta': bytes_feature(json.dumps(c_i['meta']).encode('utf-8') if is_first else b''),

                'playback_speed': int64_feature(c_i['playback_speed']),
                'start_time': float_list_feature([c_i['start_time']]),
                'end_time': float_list_feature([c_i['end_time']]),

                'tok_ids': int64_list_feature(c_i['tok_ids']),
                'tok_start_times': float_list_feature(c_i['tok_start_times']),
                'tok_end_times': float_list_feature(c_i['tok_end_times']),

                'random_text': int64_list_feature(txt[i] if i < args.num_text_seqs else []),
            }
            for k, v in current_feats.items():
                feats[f'c{i:02d}/{k}'] = v

        example = tf.train.Example(features=tf.train.Features(feature=feats))
        train_writer.write(example.SerializeToString())
        num_written += 1
        if num_written % 10 == 0:
            te = time.time() - st
            tokens_sum = sum(tokens_written)
            tokens_max = max(tokens_written)
            tokens_90perc = int(np.percentile(tokens_written, 90))
            tokens_95perc = int(np.percentile(tokens_written, 95))
            num_videos = len(video_set)
            tokens_mean = tokens_sum / len(tokens_written)
            print(
                f"Wrote {num_written} in {te:.3f}; num_videos={num_videos}, num_tokens={tokens_sum}, max_tokens_chunk={tokens_max}, tokens_mean={tokens_mean:.2f}, tokens_95perc={tokens_95perc}",
                flush=True)
    te = time.time() - st
    num_videos = len(video_set)
    print(f"Wrote {num_written} in {te:.3f}; {num_videos} videos", flush=True)
    train_writer.close()

with open('log.csv', 'w') as f:
    fieldnames = ['video_id']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for x in video_set:
        writer.writerow({'video_id': x})

log_file_out = os.path.join(args.log_folder,
                          '{}{:05d}of{:05d}.csv'.format(args.split_name, args.fold, args.num_folds))
if log_file_out.startswith('gs://' + args.bucket_name):
    blob_fn = '/'.join(log_file_out.split('/')[3:])
    print(f"Uploading to {blob_fn}", flush=True)
    bucket.blob(blob_fn).upload_from_filename('log.csv')
