import os
from io import BytesIO
from tempfile import TemporaryDirectory

import tensorflow as tf
from PIL import Image, ImageOps
from google.cloud import storage
import random

class GCSTFRecordWriter(object):
    def __init__(self, fn, buffer_size=1, auto_close=False, options=None):
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
        self.buffer_size = buffer_size
        self.buffer = []
        self.auto_close=auto_close

    def write(self, x):
        if self.buffer_size < 10:
            self.writer.write(x)
            return

        if len(self.buffer) < self.buffer_size:
            self.buffer.append(x)
        else:
            random.shuffle(self.buffer)
            for i in range(self.buffer_size // 5):  # Pop 20%
                self.writer.write(self.buffer.pop())

    def close(self):
        # Flush buffer
        if self.buffer_size > 1:
            random.shuffle(self.buffer)
        for x in self.buffer:
            self.writer.write(x)

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


def pil_image_to_jpgstring(image: Image, quality=95):
    """
    :param image: PIL image
    :return: it, as a jpg string
    """
    with BytesIO() as output:
        image.save(output, format='JPEG', quality=quality, optimize=True)
        return output.getvalue()

def get_size_for_resize(image_size, shorter_size_trg=384, longer_size_max=512):
    """
    Gets a new size for the image. We will try to make it such that the bigger size is less thanf
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

def pil_contain(image, size, method=Image.BICUBIC):
    """
    Returns a resized version of the image, set to the maximum width and height
    within the requested size, while maintaining the original aspect ratio.

    :param image: The image to resize and crop.
    :param size: The requested output size in pixels, given as a
                 (width, height) tuple.
    :param method: Resampling method to use. Default is
                   :py:attr:`PIL.Image.BICUBIC`. See :ref:`concept-filters`.
    :return: An image.
    """

    im_ratio = image.width / image.height
    dest_ratio = size[0] / size[1]

    if im_ratio != dest_ratio:
        if im_ratio > dest_ratio:
            new_height = int(image.height / image.width * size[0])
            if new_height != size[1]:
                size = (size[0], new_height)
        else:
            new_width = int(image.width / image.height * size[1])
            if new_width != size[0]:
                size = (new_width, size[1])
    return image.resize(size, resample=method)

def pad_and_scale(img, desired_width=640, desired_height=360):
    """
    :param img: Image to resize
    :param desired_size: What we will resize it to -- default 360p
    :return: new size
    """
    if img.size == (desired_width, desired_height):
        return img
    img = pil_contain(img, (desired_width, desired_height), method=Image.BICUBIC)
    if img.size == (desired_width, desired_height):
        return img

    pad_w = desired_width - img.width
    pad_w0 = pad_w // 2
    pad_w1 = pad_w - pad_w0

    pad_h = desired_height - img.height
    pad_h0 = pad_h // 2
    pad_h1 = pad_h - pad_h0
    img = ImageOps.expand(img, (pad_w0, pad_h0, pad_w1, pad_h1))
    return img






