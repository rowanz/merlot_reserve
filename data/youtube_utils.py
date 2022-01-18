"""
Simple library for reading youtube VTT files
"""
import json
import os
import re

import googleapiclient.discovery
import numpy as np
from bs4 import BeautifulSoup
from googleapiclient.http import HttpError
from youtube_dl import YoutubeDL, DownloadError
from youtube_dl.utils import subtitles_filename, ExtractorError, encodeFilename
import io
import time

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


def channel_to_video_ids(channel_id):
    """
    Converts youtube channel ID to a list of video IDs
    :param channel_id:
    :return:
    """
    ydl_opts = {
        'extract_flat': True,
        'dump_single_json': True,
        'quiet': True,
    }
    try:
        with YoutubeDL(ydl_opts) as ydl:
            hidden_playlist = ydl.extract_info(f'https://www.youtube.com/channel/{channel_id}/videos/')
            if 'entries' not in hidden_playlist:
                raise DownloadError("entries not in hidden playlist")
            # items = ydl.extract_info(hidden_playlist['url'])
    except DownloadError as e:
        print("Oh no! Got " + str(e))
        return []
    # NOTE: I could even get the titles from here!
    return hidden_playlist['entries']


def ydl_download(id, ydl_opts):
    """
    Downloads from YDL but with error handling and shit
    :param ydl_opts:
    :return: True if success!
    """
    # This dosent help
    # user_agents = ['Mozilla/5.0 (X11; Linux i686; rv:82.0) Gecko/20100101 Firefox/82.0',
    #                'Mozilla/5.0 (Linux x86_64; rv:82.0) Gecko/20100101 Firefox/82.0',
    #                 'Mozilla/5.0 (X11; Ubuntu; Linux i686; rv:82.0) Gecko/20100101 Firefox/82.0',
    #                 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:82.0) Gecko/20100101 Firefox/82.0',
    #                 'Mozilla/5.0 (X11; Fedora; Linux x86_64; rv:82.0) Gecko/20100101 Firefox/82.0',
    #                ]
    # youtube_dl.utils.std_headers['User-Agent'] = user_agents[int(np.random.choice(len(user_agents)))]
    #
    # if os.path.exists('/home/rowan/cookies.txt'):
    #     ydl_opts['cookiefile'] = '/home/rowan/cookies.txt'

    for i in range(2):
        try:
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(id, download=True, ie_key='Youtube')

                # Download manual subs
                if len(info['subtitles']) > 0 and ydl_opts['writeautomaticsub']:
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
                num_hours = 24 + float(np.random.random(1)*24)
                print(f"Hit a too many requests. Sleeping {num_hours:.3f} hrs", flush=True)
                time.sleep(num_hours*60.0*60.0)
                # There is no recovering from a toomanyrequests error
                # raise e
                # print("sleeping bc toomanyrequests", flush=True)
                # time.sleep(random.randint(1+2*i, 5+5*i))
                return False
            elif "requested format not available" in str(e):
                print("Oh no! {}  -- RETRYING IN A SEC.\n".format(str(e)), flush=True)
                time.sleep(5)
                return False
            else:
                print("Oh no! Problem \n\n{}\n".format(str(e)))
                time.sleep(5)
                return False
        except Exception as e:
            print("Misc exception: {}".format(str(e)))
            return False
    return False


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
            return [], {}

    transcript = []
    if os.path.exists(os.path.join(cache_path, '{}.v2.en.vtt'.format(id))):
        try:
            transcript = read_vtt(os.path.join(cache_path, '{}.v2.en.vtt'.format(id)))
        except KeyError as e:
            print(f"Oh no error in read_vtt! {id} {e}", flush=True)

    with open(os.path.join(cache_path, f'{id}.v2.info.json'), 'r') as f:
        info = json.load(f)
    return transcript, info


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
