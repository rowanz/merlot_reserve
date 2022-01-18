import sys
import numpy as np
import skvideo.io
import concurrent.futures
import time
import string
import editdistance
import tslearn.metrics
import demoji
import regex as re
import ftfy
import librosa
from PIL import Image
import io
sys.path.append('../')
from data.data_utils import pil_image_to_jpgstring

def _detect_black_bars_from_video(frames, blackbar_threshold=16, max_perc_to_trim=.2):
    """
    :param frames: [num_frames, height, width, 3]
    :param blackbar_threshold: Pixels must be this intense for us to not trim
    :param max_perc_to_prim: Will trim 20% by default of the image at most in each dimension
    :return:
    """
    # Detect black bars####################
    has_content = frames.max(axis=(0, -1)) >= blackbar_threshold
    h, w = has_content.shape

    y_frames = np.where(has_content.any(1))[0]
    if y_frames.size == 0:
        print("Oh no, there are no valid yframes")
        y_frames = [h // 2]

    y1 = min(y_frames[0], int(h * max_perc_to_trim))
    y2 = max(y_frames[-1] + 1, int(h * (1 - max_perc_to_trim)))

    x_frames = np.where(has_content.any(0))[0]
    if x_frames.size == 0:
        print("Oh no, there are no valid xframes")
        x_frames = [w // 2]
    x1 = min(x_frames[0], int(w * max_perc_to_trim))
    x2 = max(x_frames[-1] + 1, int(w * (1 - max_perc_to_trim)))
    return y1, y2, x1, x2


def extract_all_frames_from_video(video_file, blackbar_threshold=32, max_perc_to_trim=0.2,
                                  every_nth_frame=1, verbosity=0):
    """
    Same as exact_frames_from_video but no times meaning we grab every single frame
    :param video_file:
    :param r:
    :param blackbar_threshold:
    :param max_perc_to_trim:
    :return:
    """
    reader = skvideo.io.FFmpegReader(video_file, outputdict={'-r': '1', '-q:v': '2', '-pix_fmt': 'rgb24'},
                                     verbosity=verbosity)

    # frames = [x for x in iter(reader.nextFrame())]
    frames = []
    for i, frame in enumerate(reader.nextFrame()):
        if (i % every_nth_frame) == 0:
            frames.append(frame)

    frames = np.stack(frames)
    y1, y2, x1, x2 = _detect_black_bars_from_video(frames, blackbar_threshold=blackbar_threshold,
                                                   max_perc_to_trim=max_perc_to_trim)
    frames = frames[:, y1:y2, x1:x2]
    return frames


def extract_single_frame_from_video(video_file, t, verbosity=0):
    """
    Reads the video, seeks to the given second option
    :param video_file: input video file
    :param t: where 2 seek to
    :param use_rgb: True if use RGB, else BGR
    :return: the frame at that timestep.
    """
    timecode = '{:.3f}'.format(t)
    input_dict ={'-ss': timecode, '-threads': '1'}
    reader = skvideo.io.FFmpegReader(video_file,
                                     inputdict=input_dict,
                                     outputdict={'-r': '1', '-q:v': '2', '-pix_fmt': 'rgb24', '-frames:v': '1'},
                                     verbosity=verbosity,
                                     )
    try:
        frame = next(iter(reader.nextFrame()))
    except StopIteration:
        frame = None
    return frame

def extract_frames_from_video(video_file, times, info, use_multithreading=False, use_rgb=True,
                              blackbar_threshold=32, max_perc_to_trim=.20, verbose=False):
    """
    Extracts multiple things from the video and even handles black bars

    :param video_file: what we are loading
    :param times: timestamps to use
    :param use_multithreading: Whether to use multithreading
    :param use_rgb whether to use RGB (default) or BGR
    :param blackbar_threshold: Pixels must be this intense for us to not trim
    :param max_perc_to_prim: Will trim 20% by default of the image at most in each dimension
    :return:
    """

    def _extract(i):
        return i, extract_single_frame_from_video(video_file, times[i], verbosity=10 if verbose else 0)

    time1 = time.time()

    if not use_multithreading:
        frames = [_extract(i)[1] for i in range(len(times))]
    else:
        frames = [None for t in times]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            submitted_threads = (executor.submit(_extract, i) for i in range(len(times)))
            for future in concurrent.futures.as_completed(submitted_threads):
                try:
                    i, img = future.result()
                    frames[i] = img
                except Exception as exc:
                    print("Oh no {}".format(str(exc)), flush=True)
    if verbose:
        print("Extracting frames from video, multithreading={} took {:.3f}".format(use_multithreading,
                                                                               time.time() - time1), flush=True)
    if any([x is None for x in frames]):
        print(f"Fail on {video_file}", flush=True)
        return None

    frames = np.stack(frames)
    y1, y2, x1, x2 = _detect_black_bars_from_video(frames, blackbar_threshold=blackbar_threshold,
                                                   max_perc_to_trim=max_perc_to_trim)
    frames = frames[:, y1:y2, x1:x2]

    #############
    return frames


def align_using_dtw(input_asr, grover_output, radius_perc=0.1, radius_abs=32):
    """
    :param input_asr: List of words
    :param grover_output: List of words also, could be different size
    :param radius_perc: Percent of input ASR
    :param radius_abs: Absolute ntokens
    :return:
    """
    max_radius = int(max(len(input_asr) * radius_perc, radius_abs))
    # sometimes grover just keeps going
    if len(grover_output) > len(input_asr):
        grover_output = grover_output[:len(input_asr) + max_radius]

    # DONT give the alignment freedom if it's at the end of a sequence to just "give up" by padding with zeros
    # Default value is high
    auto2other = np.zeros((len(input_asr), len(grover_output)), dtype=np.float32) + 9999.0

    def _preprocess_text(x):
        return x.translate(str.maketrans('', '', string.punctuation)).strip().lower()

    input_asr_pre = [_preprocess_text(x) for x in input_asr]
    input_gro_pre = [_preprocess_text(x) for x in grover_output]
    for a_idx, a in enumerate(input_asr_pre):
        start = max(a_idx - max_radius, 0)
        end = min(a_idx + max_radius, len(input_gro_pre))
        for o_idx in range(start, end):
            o = input_gro_pre[o_idx]
            auto2other[a_idx, o_idx] = editdistance.eval(a, o)

    idxs, score = tslearn.metrics.dtw_path_from_metric(auto2other, metric='precomputed')
    denoised_out = [[] for x in input_asr]
    has_seen = -1
    for idx1, idx2 in idxs:
        if (idx1 >= len(input_asr)) or (idx2 >= len(grover_output)):
            break
        if idx2 > has_seen:
            # Basically don't add if it's a duplicate -- a grover output that matches to 2 things
            # This often leads to slightly weird results because we really should match the next thing, but we instead matched the first thing
            # e.g.
            # input_asr_pre = ['much', 'of', 'a', 'pancake', 'waffle', 'person', 'so', 'i', 'love', 'a']
            # input_gro_pre = ['much', 'of', 'a', 'pancakewaffle', 'person', 'so', 'i', 'love', 'a', 'good']
            # but we align pancakewaffle-> pancake and person -> waffle AND person -> person
            denoised_out[idx1].append(grover_output[idx2])
        has_seen = idx2
    return [' '.join(x) for x in denoised_out]

def clean_subtitles(subtitle_dicts):
    """
    :param subtitle_dicts: {'word': X, 'time': Y}
    :return:
    """
    # Remove &gt;&gt; maybe using ftfy or something
    new_dicts = []
    for x in subtitle_dicts:
        if x['word'].startswith('&') or x['word'].endswith(';'):
            continue
        fixed_word = ftfy.ftfy(x['word'])
        if len(fixed_word) == 0:
            continue
        x['word'] = fixed_word
        new_dicts.append(x)
    return new_dicts

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

def clean_description(text):
    # Strip emojis first
    all_emojis = demoji.findall(text)
    for k, v in all_emojis.items():
        text = text.replace(k, f'[{v}]'.replace(' ', ''))
    text = text.strip()

    # Remove URLs
    # https://stackoverflow.com/questions/11331982/how-to-remove-any-url-within-a-string-in-python/11332580
    text = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', "%", text)

    text = re.sub(' +', ' ', text) # Probably should have done
    text = re.sub('\s*\n+', '\n', text)
    text = text.strip()
    return text

######
def make_spectrogram(waveform, params):
    """
    Makes a spectrogram using librosa
    :param waveform: wave file
    :param sample_rate: Sample rate
    :param params:
    :param playback_speed:
    :return:
    """
    librosa_params = {k: v for k, v in params.items() if k not in ('eps', 'magic_number')}
    eps = params['eps']
    mel = librosa.feature.melspectrogram(waveform, **librosa_params)
    log_mel = np.log(mel + eps) - np.log(eps)
    return log_mel

def make_jpg_spectrograms(waveform_list, params_list, use_multithreading=True, verbose=False, expected_size=None):
    """
    Converts a list of waveforms (at timestamps) to JPG spectrograms

    :param waveforms: List of wavefroms
    :param params_list: Parameters
    :param use_multithreading:
    :return:
    """
    def _extract(i):
        log_mel = make_spectrogram(waveform_list[i], params_list[i])
        # print("99.9% val {:.3f}".format(np.percentile(log_mel, 99)), flush=Truef
        perc99 = max(np.percentile(log_mel, 99), 1.0)
        magic_number = 255.0 / perc99

        compressed = np.minimum(log_mel * magic_number, 255.0).astype(np.uint8)
        if expected_size is not None:
            if log_mel.shape[1] != expected_size:
                print("SIZE IS NOT RIGHT", flush=True)
                return None, None, None

        img = Image.fromarray(compressed)
        jpgstr = pil_image_to_jpgstring(img)
        return i, jpgstr, magic_number

    time1 = time.time()
    if not use_multithreading:
        frames = [_extract(i)[1:] for i in range(len(waveform_list))]
    else:
        frames = [(None, None) for t in waveform_list]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            submitted_threads = (executor.submit(_extract, i) for i in range(len(waveform_list)))
            for future in concurrent.futures.as_completed(submitted_threads):
                try:
                    i, img, mn = future.result()
                    frames[i] = (img, mn)
                except Exception as exc:
                    print("Oh no {}".format(str(exc)), flush=True)
    if verbose:
        print("Extracting frames from video, multithreading={} took {:.3f}".format(use_multithreading,
                                                                               time.time() - time1), flush=True)
    if any([x[0] is None for x in frames]):
        raise ValueError("Fail on making some spectrograms")
    return frames

def _invert_jpg_spectrogram(jpgstr, params, magic_number):
    """
    For testing -- invert the spectrogram
    :param jpgstr:
    :param params:
    :return:
    """
    inv = Image.open(io.BytesIO(jpgstr))
    inv_np = np.asarray(inv)

    mel = np.exp(inv_np.astype(np.float32) / magic_number + np.log(params['eps'])) - params['eps']
    mel = np.maximum(mel, 1e-6)

    y2 = librosa.feature.inverse.mel_to_audio(mel, **{k: v for k, v in params.items() if k not in ('n_mels', 'eps')})
    return y2
