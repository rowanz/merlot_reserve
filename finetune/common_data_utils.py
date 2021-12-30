import sys
sys.path.append('../')
import tensorflow as tf
from google.cloud import storage
from tempfile import TemporaryDirectory
import os
from mreserve.lowercase_encoder import get_encoder
import argparse
from PIL import Image
import numpy as np
from io import BytesIO
import random

encoder = get_encoder()

class GCSTFRecordWriter(object):
    def __init__(self, fn, auto_close=False, options=None):
        """
        Shuffle things in the shuffle buffer and write to tfrecords

        If buffer_size == 0 then no shuffling
        :param fn:
        :param buffer_size:
        """
        self.fn = fn
        if fn.startswith('gs://'):
            self.gclient = storage.Client()
            self.storage_dir = TemporaryDirectory()
            self.writer = tf.io.TFRecordWriter(os.path.join(self.storage_dir.name, 'temp.tfrecord'), options=options)
            self.bucket_name, self.file_name = self.fn.split('gs://', 1)[1].split('/', 1)

        else:
            self.gclient = None
            self.bucket_name = None
            self.file_name = None
            self.storage_dir = None
            self.writer = tf.io.TFRecordWriter(fn, options=options)
        self.auto_close=auto_close

    def write(self, x):
        self.writer.write(x)

    def close(self):
        self.writer.close()
        if self.gclient is not None:
            print("UPLOADING!!!!!", flush=True)
            bucket = self.gclient.get_bucket(self.bucket_name)
            blob = bucket.blob(self.file_name)
            blob.upload_from_filename(os.path.join(self.storage_dir.name, 'temp.tfrecord'))
            self.storage_dir.cleanup()

    def __enter__(self):
        # Called when entering "with" context.
        return self

    def __exit__(self, *_):
        # Called when exiting "with" context.
        # Upload shit
        if self.auto_close:
            print("CALLING CLOSE")
            self.close()


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def get_size_for_resize(image_size, shorter_size_trg=384, longer_size_max=512):
    """
    Gets a new size for the image. We will try to make it such that the bigger size is less than
    longer_size_max. However, we won't resize it if its shortest side is <= shorter_size_trg.
    :param image_size:
    :param shorter_size_trg:
    :param longer_size_max:
    :return:
    """

    w, h = image_size
    size = shorter_size_trg  # Try [size, size]

    if min(w, h) <= size:
        return w, h

    min_original_size = float(min((w, h)))
    max_original_size = float(max((w, h)))
    if max_original_size / min_original_size * size > longer_size_max:
        size = int(round(longer_size_max * min_original_size / max_original_size))

    if (w <= h and w == size) or (h <= w and h == size):
        return w, h
    if w < h:
        ow = size
        oh = int(size * h / w)
    else:
        oh = size
        ow = int(size * w / h)
    return ow, oh

def resize_image(image, shorter_size_trg=384, longer_size_max=512):
    """
    Resize image such that the longer size is <= longer_size_max.
    Gets a new size for the image. We will try to make it such that the bigger size is less than
    longer_size_max. However, we won't resize it if its shortest side is <= shorter_size_trg.
    :param image:
    :param shorter_size_trg:
    :param longer_size_max:
    """
    trg_size = get_size_for_resize(image.size, shorter_size_trg=shorter_size_trg,
                                       longer_size_max=longer_size_max)
    if trg_size != image.size:
        return image.resize(trg_size, resample=Image.BICUBIC)
    return image

def pil_image_to_jpgstring(image: Image, quality=95):
    """
    :param image: PIL image
    :return: it, as a jpg string
    """
    with BytesIO() as output:
        image.save(output, format='JPEG', quality=quality, optimize=True)
        return output.getvalue()


def create_base_parser():
    parser = argparse.ArgumentParser(description='SCRAPE!')
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
        default=1,
        type=int,
        help='Number of folds (corresponding to both the number of training files and the number of testing files)',
    )
    parser.add_argument(
        '-seed',
        dest='seed',
        default=1337,
        type=int,
        help='which seed to use'
    )
    parser.add_argument(
        '-split',
        dest='split',
        default='train',
        type=str,
        help='which split to use'
    )
    parser.add_argument(
        '-base_fn',
        dest='base_fn',
        default='gs://replace_with_your_path/',
        type=str,
        help='Base filename to use. You can start this with gs:// and we\'ll put it on google cloud.'
    )
    return parser
