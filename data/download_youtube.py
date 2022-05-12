"""
A script for downloading a video from YouTube.
Mainly this just calls YouTube-DL and does a bunch of filtering. If you have your own favored YouTube-DL setup you should use that instead.

A few notes:
* This script has a bunch of bells and whistles that we used to filter videos. If you're just trying to download YT-Temporal-180M or YT-Temporal-1B, you can comment those pieces out.
* The script downloads everything and puts them on google cloud storage right now. You'll need to provide your own bucket to do that.
* The storage might be kind of excessive (600 TiB for 20M videos). Right now we default to downloading at 360p, but maybe downloading at 240p would be okay too.
* Be careful with how you set `sleep`. If you don't wait enough between videos, you might get blocked by YouTube. If you see things like "ERROR 403" then that means you probably need to cool it with how fast you're downloading.
* You'll note we're filtering with e.g. the MobileNet CNN here, along with in `process.py`. That's because we used a stricter threshhold there. If you were really efficient, you could just use the stricter threshold here too.
* We filtered out all gaming videos and used CLD3 to detect English text. Though, lots of non-english videos snuck in anyways to our dataset. Some had non-English subtitles but were translated to English in the end. That's just how things are I guess.
* The way we're currently downloading subtitles isn't ideal. Essentially with this we always try to download the automatic subtitles as they contain finegrained timing information. What we should have probably done is try to download *BOTH* the automatic and manual subs, and then force-align the manual subs to the automatic ones, so as to hallucinate timing information there but also get better quality too. To fix this might require digging in a little bit into how youtube-dl downloads those subtitles.
* All of this is based on undocumented parts of the YouTube backend that could change at any time, you have been warned!

"""
import sys

sys.path.append('../')
import argparse

import torch.utils.data.distributed
import torchvision.models as models
import torchvision.transforms as transforms
import requests
import time
import concurrent.futures
from PIL import Image
from io import BytesIO
import os
import numpy as np
import json
from youtube_utils import download_transcript, download_video
from google.cloud import storage
import socket
import urllib3
from google.oauth2 import service_account
from google.api_core.exceptions import ServiceUnavailable
import csv
import random
import glob
import gzip
import re

parser = argparse.ArgumentParser(description="download")
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
    default=1000,
    type=int,
    help='Number of folds (corresponding to both the number of training files and the number of testing files)',
)

parser.add_argument(
    '-ids_fn',
    dest='ids_fn',
    type=str,
    help='Provide a CSV file of video IDs'
)

parser.add_argument(
    '-ckpt',
    dest='ckpt',
    default='mobilenetv2_filter_model_coco_82ptacc.pth.tar',
    type=str,
    help='checkpoint location. The checkpoint we used is at gs://merlot/video_filter_cnn/mobilenetv2_filter_model_coco_82ptacc.pth.tar - you might want to download that first'
)

parser.add_argument(
    '-local_cache_path',
    dest='local_cache_path',
    default='videos',
    type=str,
    help='Where to cache videos and things locally.'
)

parser.add_argument(
    '-bucket_name',
    dest='bucket_name',
    default='YOURBUCKETNAMEHERE',
    type=str,
    help='We will save things to gs://{bucket_name}/youtube_dump/{id}/'
)

parser.add_argument(
    '-redownload',
    dest='redownload',
    action='store_true',
    help='Whether to redownload a vide that we have already downloaded'
)

parser.add_argument(
    '-shuffle',
    dest='shuffle',
    action='store_true',
    help = 'If True, we download the videos from `ids_fn` in shuffled order'
)

parser.add_argument(
    '-sleep',
    dest='sleep',
    default=0,
    type=int,
    help='Wait this many seconds between downloads',
)

parser.add_argument(
    '-max_acs',
    dest='max_acs',
    default=0.9,
    type=float,
    help='Maximum average cosine similarity between thumbnail frames',
)
parser.add_argument(
    '-min_nco',
    dest='min_nco',
    default=0.9,
    type=float,
    help='Min num coco objects in the thumbnails',
)
parser.add_argument(
    '-skip_gaming',
    dest='skip_gaming',
    action='store_true'
)
parser.add_argument(
    '-use_cld3',
    dest='use_cld3',
    action='store_true'
)
parser.add_argument(
    '-nofilter',
    dest='nofilter',
    action='store_true',
    help='Dont perform any kind of filtering. '
         'If youre going to download YT-Temporal-1B or AudioSet or something like that, '
         'you might want to use this flag.',
)
parser.add_argument(
    '-skip_video',
    dest='skip_video',
    action='store_true',
    help='DON\'T download the videos, just the metadata.'
)
parser.add_argument(
    '-max_duration',
    dest='max_duration',
    default=20,
    type=int,
    help='max duration in minutes',
)
args = parser.parse_args()

if args.use_cld3:
    import gcld3
    lang_detector = gcld3.NNetLanguageIdentifier(min_num_bytes=0, max_num_bytes=1024)
else:
    lang_detector = None
# Low memory
# channels = pd.read_csv(args.ids_fn)
# channels = channels[channels.index % args.num_folds == args.fold]

gclient = storage.Client()
bucket = gclient.get_bucket(args.bucket_name)

if not os.path.exists(args.local_cache_path):
    os.mkdir(args.local_cache_path)

if args.ids_fn.startswith('gs://'):
    assert args.ids_fn.startswith('gs://{}/'.format(args.bucket_name))
    blob = bucket.blob('/'.join(args.ids_fn.split('/')[3:]))
    ids_fn = os.path.join(args.local_cache_path, 'ids.csv')
    blob.download_to_filename(ids_fn)
else:
    ids_fn = args.ids_fn

channels_video_ids = []
with open(ids_fn, 'r') as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        if i % args.num_folds == args.fold:
            channels_video_ids.append(row['video_id'])

if args.shuffle:
    random.shuffle(channels_video_ids)

# Load mobilenet model
model = models.MobileNetV2(num_classes=81)
model.load_state_dict({k[7:]: v for k, v in torch.load(args.ckpt,
                                                       map_location=torch.device('cpu'))['state_dict'].items()})
model.features[0][0].padding = (0, 0)
model.features[0][0].stride = (1, 1)  # Now it expects [114, 114] images
model.eval()

# retry_params = gclient.RetryParams
def _upload(fn, video_id):
    """Uploads the filename to the gcs bucket"""
    for i in range(3):
        try:
            blob = bucket.blob(f'youtube_dump/{video_id}/' + os.path.basename(fn))
            blob.upload_from_filename(fn)
            return True
        except (requests.exceptions.ConnectionError, socket.timeout, ServiceUnavailable, urllib3.exceptions.ProtocolError) as e:
            print(str(e))
            time.sleep(3*(i+1))
    return False


def _video_id_exists(video_id):
    """ Check whether we've downloaded the video already."""
    try:
        # TODO: Do something more interesting here
        return bucket.blob(f'youtube_dump/{video_id}/{video_id}.thumb.jpg').exists()
    except ServiceUnavailable as e:
        print(str(e))
        time.sleep(3600)
        return False

def load_thumbnails(id):
    """
    Given a video ID, download and load into torch all four thumbnails
    :param id: Video ID
    :return: A torch tensor that's of shape [4, 3, height, width]
    """
    thumbnails_to_dl = [0, 1, 2, 3]
    thumbnails = [None for x in thumbnails_to_dl]
    raw_thumbs = [None for x in thumbnails_to_dl]
    def _dl(i):
        resp = requests.get(f'https://i.ytimg.com/vi/{id}/{i}.jpg')
        img = Image.open(BytesIO(resp.content))
        img = transforms.Resize((90, 120))(img)  # Resize to 90 x 120 always

        # Everything is assumed to already be [90 x 120] or something with that ratio
        transform = transforms.Compose([
            transforms.CenterCrop((82, 114)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        return i, transform(img), np.asarray(img)

    time1 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        submitted_threads = (executor.submit(_dl, i) for i in thumbnails_to_dl)
        for future in concurrent.futures.as_completed(submitted_threads):
            try:
                i, img, raw_img = future.result()
                thumbnails[i] = img
                raw_thumbs[i] = raw_img
            except Exception as exc:
                print("Oh no {}".format(str(type(exc))))

    print("Downloading thumnails took {:.3f}".format(time.time() - time1), flush=True)
    thumbnails = [(torch.zeros(3,82,114) if x is None else x) for x in thumbnails]

    return torch.stack(thumbnails, 0), np.stack(raw_thumbs, 0)


def _inverse_transform(img_batch):
    """ Transform an image back into PIL format for debugging"""
    with torch.no_grad():
        # Normalize does ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
        stdev = torch.as_tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=img_batch.device)
        mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=img_batch.device)
        x_inv = img_batch.clone().mul_(stdev[None, :, None, None]).add_(mean[None, :, None, None])
        x_inv = x_inv.transpose(0, 1).reshape((3, img_batch.shape[0] * img_batch.shape[2], img_batch.shape[3]))
    x_pil = transforms.ToPILImage()(x_inv)
    return x_pil


def _allpairs_cosine_similarity(x):
    """ for a matrix of size [n, d] we will compute all pairs cosine similarity and get [n,n]"""
    pairwise_numerator = x @ x.t()
    denominator_elems = torch.sqrt(torch.diag(pairwise_numerator))
    denominator = denominator_elems[None] * denominator_elems[:, None]
    cosine_sim = pairwise_numerator / denominator
    return cosine_sim


def download_id(video_id, cache_path, verbose=False, debug=False):
    """
    Downloads the video ID + does stuff with it. If we fail, return None.
    We will save debugging printouts in videos/
    :param video_id: id to download!
    :param cache_path: Where to store things LOCALLY
    :return:
    """

    def _print(x):
        if verbose:
            print(x, flush=True)

    start = time.time()
    transcript, info = download_transcript(video_id, cache_path=cache_path)
    t1 = time.time()

    # We will return this
    uploadables = {}

    info_fn = os.path.join(cache_path, video_id + '.v2.info.json')
    if not os.path.exists(info_fn):
        info_fn = None

    # Delete old expensive stuff
    info.pop('automatic_captions', None)
    info.pop('formats', None)
    info.pop('url', None)
    info.pop('http_headers', None)
    # Save new stuff
    info['_server_used'] = os.uname()[1]
    info['_ids_fn'] = args.ids_fn

    if (not args.nofilter) and ((not transcript) or (not info_fn)):
        _print("Downloading info (no transcript): {:.1f}s".format(t1 - start))

        # Sometimes this means we were too greedy
        if args.sleep > 0:
            time.sleep(args.sleep)
        return uploadables, info

    # Get the raw transcripts
    info['transcripts'] = {}
    transcript_fns = glob.glob(os.path.join(cache_path, video_id + '.v2.*.vtt'))
    for transcript_fn in transcript_fns:
        code = transcript_fn.split(video_id + '.v2.')[1][:-len('.vtt')]
        with open(transcript_fn, 'r') as f:
            info['transcripts'][code] = f.read()
    if (not args.nofilter) and (len(info['transcripts']) == 0):
        _print("Downloading info (no transcript): {:.1f}s".format(t1 - start))
        # Sometimes this means we were too greedy
        if args.sleep > 0:
            time.sleep(args.sleep)
        return uploadables, info

    duration = info.get('duration', None)
    if (not args.nofilter) and ((duration is None) or duration > (args.max_duration * 60)):
        _print("Breaking bc video {} is too long at {}".format(video_id, duration))
        info['_failreason'] = 'video too long'
        return uploadables, info

    if args.skip_gaming:
        assert (not args.nofilter)
        for cat in ['Gaming']:
            if cat in info['categories']:
                _print(f"Skipping {cat} video")
                info['_failreason'] = 'gaming'
                return uploadables, info

    if len(transcript) > 0:
        timestamps = np.array([x[1] for x in transcript])
        words_per_30s, _ = np.histogram(timestamps, bins=30 * np.arange(timestamps[-1] // 30 + 2))
        _print("Words per 30s for {} are {}".format(video_id, words_per_30s))
        info['_words_per_30s'] = words_per_30s.tolist()

        if max(words_per_30s) < 50:
            _print(
                "Breaking {} words per 30s are {}. took {:.1f}s".format(video_id, words_per_30s, time.time() - start))
            info['_failreason'] = 'word density too low'
            return uploadables, info
    else:
        words_per_30s = [0]
        info['_words_per_30s'] = [0]


    if lang_detector is not None:
        # Lang detect
        try:
            text = info['title']

            if info['description'] is not None:
                desc_tok = re.split(r'\s+', info['description'])
                desc_tok = [x for x in desc_tok if (not '#' in x) and (not 'http' in x) and (not '@' in x) and (not 'www' in x)]
                desc_tok = desc_tok[:100]
                text += '\n' + ' '.join(desc_tok)
            text += '\n' + ' '.join([x[0] for x in transcript])
        except Exception as e:
            raise ValueError(str(e))

        res = lang_detector.FindLanguage(text=text)
        info['_cld3_lang_prob'] = res.probability
        info['_cld3_lang'] = str(res.language)
        if (res.language != 'en') or (res.probability < 0.8):
            info['_failreason'] = 'maybe not english'
            _print("Skipping bc langdetect: lang is {} p={:.3f}".format(res.language, res.probability))
            return uploadables, info

    # last but not least get thumbnails
    x, thumbs_raw = load_thumbnails(video_id)
    with torch.no_grad():
        features = model.features(x).mean([2, 3])
        cosine_sim = _allpairs_cosine_similarity(features).numpy()
        objects = torch.sigmoid(model.classifier(features)).numpy()

    thumbs_pil = Image.fromarray(np.reshape(thumbs_raw, [4 * 90, 120, 3]))
    uploadables['thumb_fn'] = os.path.join(cache_path, video_id + '.thumb.jpg')
    thumbs_pil.save(uploadables['thumb_fn'], format='JPEG', quality=90)


    avg_cosine_sim = float(np.tril(cosine_sim, -1).sum()) / 6
    info['_avg_cosine_sim'] = avg_cosine_sim

    if (not args.nofilter) and (avg_cosine_sim > args.max_acs):
        _print("Breaking {} ACS is {:.2f} took {:.1f}s".format(video_id, avg_cosine_sim, time.time() - start))
        info['_failreason'] = 'ACS too low'
        return uploadables, info

    t2 = time.time()

    num_coco_objects_expectation = objects.max(0)
    num_coco_objects_expectation = float(num_coco_objects_expectation[num_coco_objects_expectation > 0.3].sum())
    info['_num_coco_objects_expectation'] = num_coco_objects_expectation

    if (not args.nofilter) and (num_coco_objects_expectation < args.min_nco):
        _print("Breaking {} num coco objects are {:.1f}. took {:.1f}s".format(video_id, num_coco_objects_expectation,
                                                                              time.time() - start))
        info['_failreason'] = 'NCO too low'
        return uploadables, info
    # OK now download the video

    if args.skip_video and bucket.blob(f'youtube_dump/{video_id}/{video_id}.mp4').exists():
        del uploadables['thumb_fn']
        return uploadables, info

    video_fn = download_video(video_id, cache_path=cache_path)
    uploadables['video_fn'] = video_fn
    _print("Success! ON {} w/30s={} nco={:.1f} acs={:.2f}. Took dl={:.1f} thumb={:.1f} video={:.1f}".format(
        video_id, words_per_30s, num_coco_objects_expectation, avg_cosine_sim, t1 - start, t2 - t1, time.time() - t2,
    ))
    return uploadables, info


start_ = time.time()
num_uploaded = 0
for video_id in channels_video_ids:
    if args.redownload or (not _video_id_exists(video_id)):
        try:
            uploadables, info = download_id(video_id, cache_path=args.local_cache_path, verbose=True)
        except ValueError as e:
            print(str(e))
            continue

        info.pop('thumbnails', None)
        info.pop('thumbnail', None)
        info.pop('playlist', None)
        info.pop('is_live', None)
        info.pop('subtitles', None)
        info.pop('protocol', None)
        info.pop('playlist_index', None)
        info.pop('extractor_key', None)
        info.pop('extractor', None)

        info_path = os.path.join(args.local_cache_path, video_id + '.v2.info.json.gz')
        with gzip.open(info_path, 'w') as f:
            f.write(json.dumps(info).encode('utf-8'))
        uploadables['info'] = info_path

        if args.sleep > 0:
            time.sleep(args.sleep)
        # Upload the files
        success = {}
        for k, local_fn in uploadables.items():
            if (local_fn is not None) and os.path.exists(local_fn):
                success[k] = _upload(local_fn, video_id)
                try:
                    os.remove(local_fn)
                except Exception as e:
                    pass
        # DID IT
        if len(success) == 3:
            num_uploaded += 1


print("DONE with {} videos in {:.1f}sec".format(num_uploaded, time.time()-start_), flush=True)
