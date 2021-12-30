"""
Everything that you need to preprocess a video!
Or an image :)

pip install scikit-video wavfile librosa
"""
import sys
sys.path.append('../')
import scipy
import concurrent.futures
import skvideo.io
import numpy as np
import subprocess
import regex as re
import os
import tempfile
from scipy.io import wavfile
import librosa
import tensorflow as tf
from typing import Tuple, List, Dict
from pretrain.data_utils import resize_and_pad
from mreserve.lowercase_encoder import get_encoder, AUDIOSPAN, MASK, MASKAUDIO
import pandas as pd

import warnings
# This is so tensorflow doesn't hog GPUs
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    tf.config.experimental.set_visible_devices([], 'GPU')

encoder = get_encoder()


def _detect_black_bars_from_video(frames, blackbar_threshold=16, max_perc_to_trim=.2):
    """
    :param frames: [num_frames, height, width, 3]
    :param blackbar_threshold: Pixels must be this intense for us to not trim
    :param max_perc_to_prim: Will trim 20% by default of the image at most in each dimension
    :return:
    """
    # Detect black bars
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


def extract_single_frame_from_video(video_file, t):
    """
    Reads the video, seeks to the given second option
    :param video_file: input video file
    :param t: where 2 seek to
    :return: the frame at that timestep.
    """
    timecode = '{:.3f}'.format(t)
    input_dict = {'-ss': timecode, '-threads': '1'}
    reader = skvideo.io.FFmpegReader(video_file,
                                     inputdict=input_dict,
                                     outputdict={'-r': '1', '-q:v': '2', '-pix_fmt': 'rgb24', '-frames:v': '1'},
                                     verbosity=0,
                                     )
    try:
        frame = next(iter(reader.nextFrame()))
    except StopIteration:
        frame = None
    return frame


def extract_frames_from_video(video_file, times, use_multithreading=False, blackbar_threshold=32, max_perc_to_trim=.20):
    """
    Extracts multiple things from the video and even handles black bars

    :param video_file: what we are loading
    :param times: timestamps to use
    :param use_multithreading: Whether to use multithreading
    :param use_rgb whether to use RGB (default) or BGR
    :param blackbar_threshold: Pixels must be this intense for us to not trim
    :param max_perc_to_trim: Will trim 20% by default of the image at most in each dimension
    :return: Frames that are trimmed to not have any black bars
    """
    def _extract(i):
        return i, extract_single_frame_from_video(video_file, times[i])

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

    if any([x is None for x in frames]):
        print(f"Fail on {video_file}", flush=True)
        return None

    frames = np.stack(frames)
    y1, y2, x1, x2 = _detect_black_bars_from_video(frames, blackbar_threshold=blackbar_threshold,
                                                   max_perc_to_trim=max_perc_to_trim)
    return frames[:, y1:y2, x1:x2]


def make_spectrogram(waveform, playback_speed=1, sr=22050, pad_size=2):
    """
    Makes a spectrogram using librosa and the hard-coded parameters we used during pretraining.
    :param waveform: wave file
    :param playback_speed:
    :param sr: Sample rate
    :param pad_size: We will leave gaps (in the spectrogram) of this size, between things
    :return:
    """
    librosa_params = {
        'sr': sr,
        'n_mels': 64,
        'n_fft': 1536 * playback_speed,
        'hop_length': 588 * playback_speed,
        'window': scipy.signal.windows.hann,
        'fmin': 20.0,
        'fmax': 11025.0,  # Half the sample rate
    }
    eps = 1e-1
    mel = librosa.feature.melspectrogram(waveform, **librosa_params)
    log_mel = np.log(mel + eps) - np.log(eps)

    # Tack on playback speed as a (constant) feature
    log_mel = np.concatenate([log_mel, playback_speed * np.ones((1, log_mel.shape[1]), dtype=log_mel.dtype)], 0)
    log_mel = log_mel.T

    seq_size = 60
    if log_mel.shape != (seq_size * 3 + pad_size * 4, 65):
        raise ValueError("provided mel spectrogram {}. target size: {}".format(log_mel.shape, (seq_size * 3 + pad_size * 4, 65)))

    specs = np.stack([
        log_mel[pad_size:(pad_size + seq_size)],
        log_mel[(2 * pad_size + seq_size):(2 * pad_size + 2 * seq_size)],
        log_mel[(3 * pad_size + 2 * seq_size):(3 * pad_size + 3 * seq_size)],
    ])
    return specs

def invert_spectrogram(spectrogram, playback_speed=1, sr=22050):
    """
    Invert the spectrogram , this is just for debugging.
    :param spectrogram:
    :param playback_speed:
    :param sr:
    :return:
    """
    librosa_params = {
        'sr': sr,
        'n_mels': 64,
        'n_fft': 1536 * playback_speed,
        'hop_length': 588 * playback_speed,
        'window': scipy.signal.windows.hann,
        'fmin': 20.0,
        'fmax': 11025.0,  # Half the sample rate
    }
    assert spectrogram.shape == (60, 64)
    eps = 1e-1
    mel =  np.exp(spectrogram + np.log(eps)) - eps
    mel = np.maximum(mel, 1e-6)
    y2 = librosa.feature.inverse.mel_to_audio(mel.T, **{k: v for k, v in librosa_params.items() if
                                                      k not in ('n_mels', 'eps')})
    return y2



def video_to_segments(video_fn, time_interval=5.0, segment_start_time=0.0, num_segments_max=None):
    """
    Load and process the video into a list of segments, each one having
        * frame
        * spectrogram
        * start_time
        * end_time

    :param video_fn: Video filename to use. I only have tested this with .mp4
    :param time_interval: Interval in seconds
    :param segment_start_time: What time should we start extracting segments?
    :param num_segments_max: How many segments, at most should we extract?
    :return:
    """
    # Get info
    stream_txt = subprocess.run(f'ffprobe -i {video_fn} -show_streams -select_streams a -loglevel error',
                                capture_output=True, shell=True, text=True).stdout
    try:
        duration = float(re.findall(r'duration=(\d+?\.\d+)', stream_txt)[0])
    except IndexError:
        raise ValueError(f"could not parse stream for {video_fn}.\n{stream_txt}")

    duration -= 1.0  # just be safe to not try to get anything from the end of the video
    if duration < 5:
        raise ValueError(f"Video {video_fn} is too short")

    ##############################################
    # 0. Start the process for extracting audio
    ##############################################
    temp_folder = tempfile.TemporaryDirectory()
    audio_fn = os.path.join(temp_folder.name, 'audio.wav')
    ffmpeg_process = subprocess.Popen(['ffmpeg', '-y', '-i', video_fn, '-ac', '1', '-ar', '22050',
                                       audio_fn], stdout=-1, stderr=-1, text=True)

    ##############################################
    # 1. extract frames
    ##############################################
    times = []
    st = segment_start_time
    while (st + time_interval) < duration:
        et = min(duration, st + time_interval)
        times.append({'start_time': st, 'end_time': et, 'mid_time': (st + et) / 2.0})
        st = et
        if (num_segments_max is not None) and (len(times) >= num_segments_max):
            break

    frames = extract_frames_from_video(video_fn, times=[t['mid_time'] for t in times], use_multithreading=True)

    ##############################################
    # 2. Finish extracting audio
    ##############################################
    try:
        stdout, stderr = ffmpeg_process.communicate(None, timeout=10.0)
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
    desired_final_frame = int(sr * max([t['end_time'] for t in times]))
    if waveform.size < desired_final_frame:
        waveform = np.concatenate([waveform, np.zeros(desired_final_frame - waveform.size, dtype=np.float32)], 0)

    # Process each segment. here i'm always using a playback_speed of 1 (aka no fast forwarding).
    spectrograms = []
    for ts_group in times:
        start_idx = int(sr * ts_group['start_time'])
        end_idx = int(sr * ts_group['end_time'])

        wav_ts = waveform[start_idx:end_idx]
        spectrograms.append(make_spectrogram(wav_ts, playback_speed=1, sr=sr))
    temp_folder.cleanup()

    # Turn this into a unified list
    for i, (frame_i, spec_i, ts_i) in enumerate(zip(frames, spectrograms, times)):
        ts_i['frame'] = frame_i
        ts_i['spectrogram'] = spec_i
        ts_i['idx'] = i
    return times


def video_to_segments_zero_shot(video_fn, time_interval=1.0, times=None):
    """
    Load and process the video into a list of segments, each one having
        * frame
        * spectrogram
        * end_time

    :param video_fn: Video filename to use. I only have tested this with .mp4
    :param time_interval: Interval in seconds
    :return:
    """
    ##############################################
    # 0. Start the process for extracting audio
    ##############################################
    temp_folder = tempfile.TemporaryDirectory()
    audio_fn = os.path.join(temp_folder.name, 'audio.wav')
    ffmpeg_process = subprocess.Popen(['ffmpeg', '-y', '-i', video_fn, '-ac', '1', '-ar', '22050',
                                       audio_fn], stdout=-1, stderr=-1, text=True)

    ##############################################
    # 1. extract frames
    ##############################################
    frames = extract_frames_from_video(video_fn, times=[t['mid_time'] for t in times], use_multithreading=True)

    ##############################################
    # 2. Finish extracting audio
    ##############################################
    try:
        stdout, stderr = ffmpeg_process.communicate(None, timeout=500.0)
    except subprocess.TimeoutExpired:
        ffmpeg_process.kill()
        # stdout, stderr = subprocess.TimeoutExpired.communicate()
        raise ValueError("couldnt convert in time")
    except:  # Keyboardinterrupt
        ffmpeg_process.kill()
        raise
    ffmpeg_process.kill()

    sr, waveform = wavfile.read(audio_fn, mmap=True)
    waveform = waveform.astype('float32')
    waveform /= max(np.abs(waveform).max(), 1.0)

    # Pad to max time just in case
    desired_final_frame = int(sr * max([t['end_time'] for t in times]))
    if waveform.size < desired_final_frame:
        waveform = np.concatenate([waveform, np.zeros(desired_final_frame - waveform.size, dtype=np.float32)], 0)

    # Process each segment. here i'm always using a playback_speed of 1 (aka no fast forwarding).
    spectrograms = []
    total_audio_len = sr * 5.0
    for ts_group in times:
        rest_time = 5.0 - (ts_group['end_time'] - ts_group['start_time'])
        if rest_time > 0:
            start_idx = int(sr * ts_group['start_time'])
            end_idx = int(sr * ts_group['end_time'])
            wav_ts = waveform[start_idx:end_idx]
            left_pad = int((total_audio_len - len(wav_ts)) / 2)
            right_pad = int(total_audio_len - len(wav_ts) - left_pad)            
            wav_ts = np.concatenate([np.zeros(left_pad, dtype=np.float32), wav_ts, np.zeros(right_pad, dtype=np.float32)], 0)
        else:
            start_idx = int(sr * (ts_group['mid_time']-2.5))
            end_idx = int(sr * (ts_group['mid_time']+2.5))
            wav_ts = waveform[start_idx:end_idx]
        
        spectrograms.append(make_spectrogram(wav_ts, playback_speed=1, sr=sr))
    temp_folder.cleanup()

    # Turn this into a unified list
    for i, (frame_i, spec_i, ts_i) in enumerate(zip(frames, spectrograms, times)):
        ts_i['frame'] = frame_i
        ts_i['spectrogram'] = spec_i
        ts_i['idx'] = i
    return times


def video_to_segments_for_action_segmentation(video_fn, time_interval=1.0, segment_start_time=0.5):
    """
    For action segmentation we need to get dense predictions over the entire video
    :param video_fn:
    :param time_interval:
    :return:
    """
    # Get info
    stream_txt = subprocess.run(f'ffprobe -i {video_fn} -show_streams -select_streams a -loglevel error',
                                capture_output=True, shell=True, text=True).stdout
    try:
        duration = float(re.findall(r'duration=(\d+?\.\d+)', stream_txt)[0])
    except IndexError:
        if video_fn.endswith('_fixed.mp4'):
            raise ValueError(f"could not parse stream for {video_fn}.\n{stream_txt}")
        # convert it
        print("CONVERTING", flush=True)
        video_fn2 = video_fn.replace('.mp4', '_fixed.mp4')
        os.system(f"ffmpeg -y -f lavfi -i aevalsrc=0 -i {video_fn} -c:v copy -c:a aac -map 0 -map 1:v -shortest {video_fn2}")
        return video_to_segments_for_action_segmentation(video_fn2, time_interval, segment_start_time)
        # raise ValueError(f"could not parse stream for {video_fn}.\n{stream_txt}")

    duration -= 0.5  # just be safe to not try to get anything from the end of the video

    ##############################################
    # 0. Start the process for extracting audio
    ##############################################
    temp_folder = tempfile.TemporaryDirectory()
    audio_fn = os.path.join(temp_folder.name, 'audio.wav')
    ffmpeg_process = subprocess.Popen(['ffmpeg', '-y', '-i', video_fn, '-ac', '1', '-ar', '22050',
                                       audio_fn], stdout=-1, stderr=-1, text=True)

    ##############################################
    # 1. extract frames
    ##############################################
    times = []
    st = segment_start_time
    while (st + time_interval) < duration:
        et = min(duration, st + time_interval)
        times.append({'start_time': st, 'end_time': et, 'mid_time': (st + et) / 2.0})
        st = et

    frames = extract_frames_from_video(video_fn, times=[t['mid_time'] for t in times], use_multithreading=True)

    ##############################################
    # 2. Finish extracting audio
    ##############################################
    try:
        stdout, stderr = ffmpeg_process.communicate(None, timeout=10.0)
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
    desired_final_frame = int(sr * max([t['end_time'] for t in times]))
    if waveform.size < desired_final_frame:
        waveform = np.concatenate([waveform, np.zeros(desired_final_frame - waveform.size, dtype=np.float32)], 0)

    # Process 1 out of 2 audios -- equivalent of 2 seconds long versus 3.333
    spectrograms = []
    for i in range(len(times) // 2):
        start_idx = int(sr * times[i]['start_time'])
        end_idx = int(sr * times[i+1]['end_time'])

        wav_ts = waveform[start_idx:end_idx]
        total_audio_len = int(sr * 4.79)
        start_pad = int(sr * 0.05)
        end_pad = total_audio_len - wav_ts.size - start_pad

        wav_ts_pad = np.concatenate([
            np.zeros(start_pad, dtype=wav_ts.dtype),
            wav_ts,
            np.zeros(end_pad, dtype=wav_ts.dtype)], 0)

        new_spec = make_spectrogram(wav_ts_pad, playback_speed=1, sr=sr, pad_size=0)
        spectrograms.append(new_spec[0, None])
        spectrograms.append(new_spec[1, None])



    # one more if it's odd
    if len(spectrograms) != len(times):
        wav_ts = waveform[end_idx:]
        start_pad = int(sr * 0.05)
        total_audio_len = int(sr * 4.79)
        wav_ts = wav_ts[:(total_audio_len - start_pad)]
        end_pad = total_audio_len - wav_ts.size - start_pad

        wav_ts_pad = np.concatenate([
            np.zeros(start_pad, dtype=wav_ts.dtype),
            wav_ts,
            np.zeros(end_pad, dtype=wav_ts.dtype)], 0)
        new_spec = make_spectrogram(wav_ts_pad, playback_speed=1, sr=sr, pad_size=0)
        spectrograms.append(new_spec[0, None])

    temp_folder.cleanup()

    # Turn this into a unified list
    for i, (frame_i, spec_i, ts_i) in enumerate(zip(frames, spectrograms, times)):
        ts_i['frame'] = frame_i
        ts_i['spectrogram'] = spec_i
        ts_i['idx'] = i
    return times



def preprocess_image_to_patches(img, output_grid_size: Tuple[int, int]):
    """
    Turns an image into a list of patches (in tensorflow for now).
    :param img: image that is uint8
    :param output_grid_size: The resolution we use ( a tuple of length 2)
    :return:
    """
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    h1, w1 = output_grid_size
    P = 16
    assert h1 <= 24, "we didn't pretrain on anything bigger than 24x24 or 18x32"
    assert w1 <= 32, "we didn't pretrain on anything bigger than 24x24 or 18x32"
    img, this_image_info = resize_and_pad(img, (h1 * P, w1 * P), do_random_scale=False,
                                          do_flip_if_vertical=False, resize_method='bilinear')
    img = tf.nn.space_to_depth(img[None], P, data_format='NHWC')
    img = tf.reshape(img, [h1 * w1, P * P * 3])
    img = img._numpy()
    return img


def preprocess_video(video_segments: List[Dict], output_grid_size: Tuple[int, int], verbose=True):
    """
    Preprocess a list of video segments.
    :param video_segments: A list of at most 8 segments. Each segment is a dictionary that has to have:
        * `frame`: a [H, W, 3] image
        * `spectrogram` an array of size [3, 60, 65] -- for each subsegment it's a 60 (Time) x 65 (num_mels) audio spectrogram
        * `text`: Text that you want to provide as input. it can either be pre-tokenized or not
        * `use_text_as_input`: optional, set this to True (default) to use text as input, otherwise we use `spectrogram` as the input
    :param output_grid_size: The resolution we use ( a tuple of length 2)
    :param verbose: verbose

    :return: A dictionary of things. This isn't batched (that logic is a bit trickier, but it's also more efficient)
    """
    if len(video_segments) > 8:
        raise ValueError("We only support videos of at most 8 segments right now")

    images = np.stack([preprocess_image_to_patches(o_i['frame'],
                                                   output_grid_size=output_grid_size) for o_i in video_segments])
    subseg_idxs = [] # also known as 'audio_ptr'
    audio_clips = []
    tokens_out = []
    for i, segm_i in enumerate(video_segments):
        if segm_i.get('use_text_as_input', True):
            txt = segm_i.get('text', '')

            if isinstance(txt, str):
                txt_tok = encoder.encode(txt).ids
            else:
                txt_tok = txt
                txt = encoder.decode(txt,skip_special_tokens=False)

            if verbose:
                print(f"Segment {i}: using text not audio as input: {txt}", flush=True)

            # Append a dummy audio clip
            audio_clips.append(np.zeros([3, 60, 65], dtype=np.float32))

            # sub-segment index
            # Getting this exact isn't so critical since we always integer-divide by 3
            subseg_idxs.extend([i * 3] * len(txt_tok))
            tokens_out.extend(txt_tok)
        else:
            if verbose:
                print(f"Segment {i}: using audio as input (not text)", flush=True)
            audio_clips.append(segm_i['spectrogram'])

            # always 6 tokens of `audiospan' per subsegment (so 18 in total)
            tokens_out.extend([AUDIOSPAN] * 18)
            subseg_idxs.extend((i * 3 + np.arange(18) // 6).tolist())
    if len(tokens_out) >= 160:
        print(f"warning -- truncating tokens {len(tokens_out)} to be 160", flush=True)
        tokens_out = tokens_out[:160]
        subseg_idxs = subseg_idxs[:160]

    while len(tokens_out) < 160:
        tokens_out.append(0)
        subseg_idxs.append(-1)

    if verbose:
        out_df = pd.DataFrame([{'tok': encoder.decode([t_i], skip_special_tokens=False), 'idx': idx_i} for t_i, idx_i in zip(tokens_out, subseg_idxs) if t_i > 0])
        pd.set_option('display.max_rows', None)
        print(out_df)
    return {
        'images': images,
        'audio_clips': np.stack(audio_clips).reshape(-1, 60, 65),
        'tokens': np.array(tokens_out, dtype=np.int32),
        'subseg_idxs': np.array(subseg_idxs, dtype=np.int32),
    }
