# We will write videos to
import sys
sys.path.append('../')
from youtube_dl import YoutubeDL, DownloadError
from youtube_dl.utils import subtitles_filename, ExtractorError, encodeFilename
import io
from bs4 import BeautifulSoup
from mreserve.preprocess import video_to_segments, preprocess_video, encoder, MASK


from google.cloud import storage
import tempfile
import os
import regex as re
import json
import glob
import pandas as pd
import ftfy

def ts_to_sec(ts):
    """
    Splits a timestamp of the form HH:MM:SS.MIL
    :param ts:  timestamp of the form HH:MM:SS.MIL
    :return: seconds
    """
    rest, ms = ts.split('.')
    hh, mm, ss = rest.split(':')
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + float('.{}'.format(ms))


def sec_to_ts(sec):
    """
    seconds to timestamp
    :param sec:  number of seconds
    :return: timestamp of the form HH:MM:SS.MIL
    """
    ms = '{:.3f}'.format(sec).split('.')[-1]
    int_time = int(sec)
    ss = int_time % 60
    int_time = int_time // 60
    mm = int_time % 60
    hh = int_time // 60
    return '{:0>2d}:{:0>2d}:{:0>2d}.{}'.format(hh, mm, ss, ms)


def _read_part(stuff, start_ts, stop_ts):
    """
    Reads a part of a VTT subtitles file

    My current understanding is that the important stuff looks like this

    00:00:00.030 --> 00:00:02.060 align:start position:0%

    hello<00:00:00.450><c> everyone</c><00:00:00.840><c> and</c><00:00:01.140><c> welcome</c><00:00:01.199><c> in</c><00:00:01.800><c> this</c><00:00:01.890><c> video</c>

    so Hello started at 00:00:00.030 and goes to 00:00:00.450, everyone started right after and goes to 00:00:00.840

    We will extract the start and stop timesteps

    :param stuff: VTT text between two timestamps.
    :param ts: Initial timestamp.
    :return:
     (word, start time, end time, start time of chunk, end time of chunk, distance from left, distance from right)
    """
    # print("Start: {} stop: {} \n stuff {} \n\n".format(start, stop, stuff), flush=True)
    matching_lines = re.findall(r'^(.+<\d\d:\d\d:\d\d\.\d\d\d>.+)$', '\n'.join(stuff), flags=re.MULTILINE)

    start_time = ts_to_sec(start_ts)
    end_time = ts_to_sec(stop_ts)

    if len(matching_lines) == 0:
        # EXCEPTION: IF there is a single word, that is on the second line, and we didn't add anything
        if len(stuff) >= 3 and len(stuff[1].strip()) > 0 and ('<c>' not in stuff[1]) and len(stuff[1].strip().split(' ')) > 0:
            return [(stuff[1].strip(), start_time, end_time)]
        else:
            return []
    if not len(matching_lines) == 1:
        raise ValueError("WTF? VTT subtitles not well formed:\n{}".format('\n'.join(stuff)))

    # stuff0 = '<{}>{}<{}>'.format(start, matching_lines[0], stop)
    stuff1 = re.sub(r'(c.color\S\S\S\S\S\S)', lambda x: 'c color="{}"'.format(x[0][-6:]), matching_lines[0])
    stuff2 = re.sub(r'(<\d\d:\d\d:\d\d\.\d\d\d>)', lambda x: '<timestamp t="{}"></timestamp>'.format(x[0][1:-1]),
                    stuff1)

    # We need to attach all the CSS tags uptop
    stuff3 = re.sub(r'(c.color\S\S\S\S\S\S)', lambda x: 'c color="{}"'.format(x[0][-6:]),
                    ''.join(re.findall(r'(</?c.*?>)', stuff[0]))) + stuff2

    soup = BeautifulSoup(stuff3, 'lxml')
    children = next(next(soup.children).children)

    words = []
    timesteps = []
    cur_start = start_ts
    conf = 'CCCCCC'
    conf_meanings = {'CCCCCC': 0, 'E5E5E5': 1}

    for child in children.recursiveChildGenerator():
        name = getattr(child, "name", None)
        if name == 'timestamp':
            cur_start = child.attrs['t']
        elif name == 'c':
            if 'color' in child.attrs:
                conf = child.attrs['color']
        elif child.isspace is not None and not child.isspace():
            words.append(child.strip())
            timesteps.append(cur_start)
    timesteps.append(stop_ts)
    buffer = []
    for w_i, word in enumerate(words):
        buffer.append((word, ts_to_sec(timesteps[w_i]), ts_to_sec(timesteps[w_i+1])))
    return buffer


def ydl_download(id, ydl_opts):
    """
    Downloads from YDL but with error handling and shit
    :param ydl_opts:
    :return: True if success!
    """
    for i in range(2):
        try:
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(id, download=True, ie_key='Youtube')

                # Download manual subs
                if len(info.get('subtitles', [])) > 0 and ydl_opts['writeautomaticsub']:
                    ydl.params['writesubtitles'] = True
                    # Hack to get manual subs too
                    ps = ydl.process_subtitles(id, normal_subtitles=info['subtitles'], automatic_captions=None)

                    ie = ydl.get_info_extractor(info['extractor_key'])
                    for lang_raw, sub_info in ps.items():
                        sub_lang = f'{lang_raw}-manual'
                        sub_format = sub_info['ext']
                        filename = ydl.prepare_filename(info)
                        sub_filename = subtitles_filename(filename, sub_lang, sub_format, info.get('ext'))
                        try:
                            sub_data = ie._request_webpage(
                                    sub_info['url'], info['id'], note=False).read()
                            with io.open(encodeFilename(sub_filename), 'wb') as subfile:
                                subfile.write(sub_data)
                        except (ExtractorError, IOError, OSError, ValueError) as err:
                            print(str(err), flush=True)
                            continue
            return True
        except DownloadError as e:
            if "Too Many Requests" in str(e):
                return False
            elif "requested format not available" in str(e):
                print("Oh no! {}  -- RETRYING IN A SEC.\n".format(str(e)), flush=True)
                return False
            else:
                print("Oh no! Problem \n\n{}\n".format(str(e)))
                return False
        except Exception as e:
            print("Misc exception: {}".format(str(e)))
            return False
    return False

def read_uploaded_vtt(stuff):
    """
    Reads in a user uploaded VTT file
    :param stuff: list of things
    :return:
    """
    start = None
    stop = None
    buffer = []
    everything = []

    def _pop_buffer(start, stop):
        # We have to guess word level alignments, now the buffer looks like
        # ['MALE SPEAKER: And your hand', "shakes from Parkinson's?", '']
        clean_buffer = re.sub(r'<.*?>', '', ' '.join(buffer))

        clean_buffer = [x.strip() for x in clean_buffer.split(' ')]
        clean_buffer = [x for x in clean_buffer if len(x) > 0]

        start_sec = ts_to_sec(start)
        end_sec = ts_to_sec(stop)

        # The timestamps are when the word STARTS which is why I do the +1 thingy'
        # TOTALLY NOT TESTED
        timestamps = np.linspace(start=start_sec, stop=end_sec, num=len(clean_buffer) + 1)
        for b_i, t_s, t_e in zip(clean_buffer, timestamps[:-1], timestamps[1:]):
            everything.append((b_i, t_s, t_e))

    for line in stuff:
        # Sometimes things end with "line:0%" or they have HTML in them
        possible_re_match = re.findall(r'^(.+) --> ([^\s]+)', line)
        if len(possible_re_match) == 1:
            if (start is not None) and (stop is not None):
                _pop_buffer(start, stop)
            # Do it again, on the trimmed line
            possible_re_match = re.findall(r'^(.+) --> (.+)', line[:len("00:00:17.683 --> 00:00:19.285")])
            start, stop = possible_re_match[0]
            buffer = []
        else:
            buffer.append(line)

    if (len(buffer) >= 0) and (start is not None) and (stop is not None):
        _pop_buffer(start, stop)
    return everything

def read_vtt_text(stuff, skip_if_no_timing_info=False):
    """
    Reads in a VTT (as text), split into lines
    :param stuff: LIST of strings
    :param skip_if_no_timing_info: If we can't find any timing information -- skip
    :return: List of tuples
    """

    if skip_if_no_timing_info:
        if '<c>' not in ''.join(stuff):
            return None

    start = None
    stop = None
    buffer = []
    everything = []
    for line in stuff:
        possible_re_match = re.findall(r'^(.+) --> (.+) align:start position:0%', line)
        if len(possible_re_match) == 1:
            if (start is not None) and (stop is not None):
                part = _read_part(buffer, start, stop)
                everything.extend(part)

            start, stop = possible_re_match[0]
            buffer = []
        else:
            buffer.append(line)

    # Add in a missing line
    if len(buffer) > 0:
        try:
            everything.extend(_read_part(buffer, start, stop))
        except (ValueError, KeyError, AttributeError) as e:
            print("Missing line error {}: {}".format(buffer, str(e)), flush=True)
    # The above reads in Google's format. OTHERWISE we need to read a user uploaded format,
    # which is unfortunately differnt
    if (len(everything) == 0) and (len(buffer) > 0) and (buffer[0] == 'WEBVTT'):
        if skip_if_no_timing_info:
            return None
        return read_uploaded_vtt(buffer)
    return everything


def read_vtt(fn):
    """
    Reads in a VTT file and produces a list of tuples, each one containing (word, confidence (0 or 1) and timestamp).
    :param fn: VTT filename
    :return: List of tuples
    """
    with open(fn, 'r') as f:
        stuff = f.read().splitlines()
    return read_vtt_text(stuff)


def clean_subtitle_tuples(subtitle_tuples):
    """
    :param subtitle_tuples: e.g. ('and', 0, 242.06). also get rid of middle one
    :return:
    """
    new_subs = []
    for (word, ts0, ts1) in subtitle_tuples:
        if word.startswith('&') or word.endswith(';'):
            continue
        fixed_word = ftfy.ftfy(word)
        if len(fixed_word) == 0:
            continue
        new_subs.append({'word': word, 'start': ts0, 'end': ts1})
    return new_subs


def download_transcript(id, cache_path):
    """
    Given an ID, download and read JUST the transcript + info

    :param id: Video id
    :param cache_path: Where to download the transcript
    :return: Transcript (list of tuples), Info (json)
    """
    if not os.path.exists(os.path.join(cache_path, f'{id}.v2.info.json')):
        ydl_opts = {
            'writedescription': False,
            'writeinfojson': True,
            'write_all_thumbnails': False,
            'writeautomaticsub': True,
            'writesubtitles': False,
            'subtitlesformat': 'vtt',
            'cachedir': cache_path,
            'format': 'best[height=360]',
            'outtmpl': os.path.join(cache_path, '%(id)s.v2.%(ext)s'),
            'skip_download': True,
            'subtitleslangs': ['en','EN','eng','ENG','en-gb','en-us','en-GB','en-US','EN-GB','EN-US','english','English','ENGLISH'],
            'source_address': '0.0.0.0',
            'ratelimit': 10000,
            'no_warnings': True,
        }
        if not ydl_download(id, ydl_opts):
            ydl_opts['writeautomaticsub'] = False
            if not ydl_download(id, ydl_opts):
                return {'transcript': []}

    transcript = []
    if os.path.exists(os.path.join(cache_path, '{}.v2.en.vtt'.format(id))):
        try:
            transcript = read_vtt(os.path.join(cache_path, '{}.v2.en.vtt'.format(id)))
        except KeyError as e:
            transcript = []
            print(f"Oh no error in read_vtt! {id} {e}", flush=True)

    transcript = clean_subtitle_tuples(transcript)

    with open(os.path.join(cache_path, f'{id}.v2.info.json'), 'r') as f:
        info = json.load(f)

    # Delete old expensive stuff
    info.pop('automatic_captions', None)
    info.pop('formats', None)
    info.pop('url', None)
    info.pop('http_headers', None)
    info['transcript'] = transcript
    return info

def download_video(id, cache_path):
    """
    Given an ID, download the video using HQ settings. (might need to turn this down, idk)

    :param id: Video id
    :param cache_path: Where to download the video
    :return: The file that we downloaded things to
    """
    ydl_opts = {
        'writedescription': False,
        'writeinfojson': False,
        'write_all_thumbnails': False,
        'writeautomaticsub': False,
        'cachedir': cache_path,
        'format': 'best[height<=360][ext=mp4]',
        'outtmpl': os.path.join(cache_path, '%(id)s.%(ext)s'),
        'retries': 3,
        'ignoreerrors': True,
        'sub_lang': 'en',
        'source_address': '0.0.0.0',
    }
    if ydl_download(id, ydl_opts):
        return os.path.join(cache_path, f'{id}.mp4')
    return None

def load_video_info(video_id):
    """
    Loads the youtbe metadata and caches it
    :param video_id:
    :return:
    """
    os.makedirs('cache', exist_ok=True)
    cache_fn = os.path.join('cache', f'{video_id}.json')

    info = download_transcript(video_id, cache_path='cache')
    with open(cache_fn, 'w') as f:
        json.dump(info, f)

    # Clean up the timestamps
    return info


def load_video_mp4(video_id, start_time=0.0):
    """
    Load the video and convert it into segments
    :param video_id:
    :return:
    """
    cache_path = tempfile.TemporaryDirectory()
    cache_fn = os.path.join('cache', f'{video_id}.mp4')

    download_video(video_id, cache_path='cache')

    # Now grab the video!

    video_segments = video_to_segments(cache_fn, time_interval=5.0, segment_start_time=start_time, num_segments_max=8)
    cache_path.cleanup()
    return video_segments

def load_video(video_id, start_time=0.0):
    video_info = load_video_info(video_id)
    transcript = video_info['transcript']
    for x in transcript:
        x['middle_time'] = (x['start'] + x['end']) / 2.0
    video_segments = load_video_mp4(video_id, start_time=start_time)

    # Add in subtitles!
    for x in video_segments:
        x['sub'] = [t['word'] for t in transcript if t['middle_time'] >= x['start_time'] and t['middle_time'] < x['end_time']]
        x['sub'] = ' '.join(x['sub'])
    return video_segments

