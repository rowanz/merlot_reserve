"""
Tensorflow utilities for datalaoding
"""
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops


def apply_with_random_selector(x, func, num_cases):
    """Computes func(x, sel), with sel sampled from [0...num_cases-1].
    Args:
      x: input Tensor.
      func: Python function to apply.
      num_cases: Python int32, number of cases to sample sel from.
    Returns:
      The result of func(x, sel), where func receives the value of the
      selector as a python integer, but sel is sampled dynamically.
    """
    sel = tf.random.uniform([], maxval=num_cases, dtype=tf.int32)
    # Pass the real x only to one of the func calls.
    return control_flow_ops.merge([
        func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
        for case in range(num_cases)])[0]


def flip_if_vertical(image):
    """
    https://www.youtube.com/watch?v=f2picMQC-9E
    :param image:
    :return:
    """
    height = tf.cast(tf.shape(image)[0], tf.float32)
    width = tf.cast(tf.shape(image)[1], tf.float32)
     # Pad and then add some constants (if it's flipped) to tell the model that we messed with it
    image = tf.cond(
        height >= (4 * width / 3.0),
        lambda: tf.pad(tf.image.rot90(image), [[0,0], [4, 4], [0,0]], mode='CONSTANT', constant_values=0.5),
        lambda: image,
    )
    return image


def resize_and_pad(image, desired_output_size,
                   random_scale_min=0.1, random_scale_max=2.0, do_random_scale=False,
                   shrink_both_sides=True,
                   do_flip_if_vertical=True,
                   resize_method=tf.image.ResizeMethod.BILINEAR):
    """


    :param image:
    :param desired_output_size:
    :param boxes:
    :param random_scale_min:
    :param random_scale_max:
    :param do_random_scale:
    :param shrink_both_sides: whether both sides can be shrunk at the same time
    :return:
    """
    if do_flip_if_vertical:
        image = flip_if_vertical(image)

    desired_height, desired_width = desired_output_size
    desired_height_f = tf.cast(desired_height, dtype=tf.float32)
    desired_width_f = tf.cast(desired_width, dtype=tf.float32)

    height = tf.cast(tf.shape(image)[0], tf.float32)
    width = tf.cast(tf.shape(image)[1], tf.float32)

    if do_random_scale:
        random_scale_factor = tf.random.uniform([], random_scale_min, random_scale_max)
        if not shrink_both_sides:
            # Max random is where scale * W > W_desired
            #                     scale * H > H_desired
            rsf_max = tf.maximum(desired_width_f / width, desired_height_f / height)
            random_scale_factor = tf.minimum(rsf_max, random_scale_factor)

        scaled_y = tf.cast(random_scale_factor * desired_height_f, tf.int32)
        scaled_x = tf.cast(random_scale_factor * desired_width_f, tf.int32)

        # Recompute the accurate scale_factor using rounded scaled image size.
        image_scale_y = tf.cast(scaled_y, tf.float32) / height
        image_scale_x = tf.cast(scaled_x, tf.float32) / width
        image_scale = tf.minimum(image_scale_x, image_scale_y)

        # Conceptual captions has some REALLY WIDE images I believe
        # this ensures that we won't scale any side lower than to 64
        image_scale = tf.maximum(image_scale, 64.0 / tf.minimum(height, width))

        # Select non-zero random offset (x, y) if scaled image is larger than
        # self._output_size.
        scaled_height = tf.cast(height * image_scale, tf.int32)
        scaled_width = tf.cast(width * image_scale, tf.int32)
        offset_y = tf.cast(scaled_height - desired_height, tf.float32)
        offset_x = tf.cast(scaled_width - desired_width, tf.float32)
        offset_y = tf.maximum(0.0, offset_y) * tf.random.uniform([], 0, 1)
        offset_x = tf.maximum(0.0, offset_x) * tf.random.uniform([], 0, 1)
        offset_y = tf.cast(offset_y, tf.int32)
        offset_x = tf.cast(offset_x, tf.int32)
    else:
        image_scale_y = desired_height_f / height
        image_scale_x = desired_width_f / width
        image_scale = tf.minimum(image_scale_x, image_scale_y)
        scaled_height = tf.cast(height * image_scale, tf.int32)
        scaled_width = tf.cast(width * image_scale, tf.int32)
        offset_y = tf.constant(0)
        offset_x = tf.constant(0)

    # Now resize and crop
    if resize_method == 'random' and do_random_scale and (not tf.executing_eagerly()):
        resize_methods = sorted([k for k in tf.image.ResizeMethod.__dict__.keys() if k.isupper()])
        print("Random resize method:\n{}".format(','.join(resize_methods)))
        image = apply_with_random_selector(
            image,
            lambda x, method_idx: tf.image.resize(x, [scaled_height, scaled_width],
                                                  tf.image.ResizeMethod.__dict__[resize_methods[method_idx]],
                                                  antialias=True),
            num_cases=len(resize_methods))
    elif resize_method != 'random':
        image = tf.image.resize(image, [scaled_height, scaled_width], method=resize_method, antialias=True)
    else:
        print(f"you passed in {resize_method} but doing bilinear resize instead (possibly because eager is on)")
        image = tf.image.resize(image, [scaled_height, scaled_width],
                                method=tf.image.ResizeMethod.BILINEAR, antialias=True)

    image = tf.clip_by_value(image, 0.0, 1.0)

    image = image[offset_y:offset_y + desired_height, offset_x:offset_x + desired_width, :]
    image = tf.image.pad_to_bounding_box(image, 0, 0, desired_height, desired_width)

    if isinstance(desired_height, int) and isinstance(desired_width, int):
        image.set_shape([desired_height, desired_width, 3])
    else:
        print("Cant set shape bc desired height/width are dynamic")

    effective_height = tf.minimum(scaled_height, desired_height)
    effective_width = tf.minimum(scaled_width, desired_width)

    image_info = tf.stack([
        tf.cast(effective_height, dtype=tf.float32) / desired_height_f,
        tf.cast(effective_width, dtype=tf.float32) / desired_width_f,
        1.0 / image_scale,
        height,
        width,
        tf.cast(offset_y, dtype=tf.float32) / height,
        tf.cast(offset_x, dtype=tf.float32) / width,
    ])
    return image, image_info


def assert_rank(tensor, expected_rank, name=None):
    """Raises an exception if the tensor rank is not of the expected rank.

    Args:
      tensor: A tf.Tensor to check the rank of.
      expected_rank: Python integer or list of integers, expected rank.
      name: Optional name of the tensor for the error message.

    Raises:
      ValueError: If the expected shape doesn't match the actual shape.
    """
    if name is None and not tf.executing_eagerly():
        name = tensor.name

    expected_rank_dict = {}
    if isinstance(expected_rank, int):
        expected_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True

    actual_rank = tensor.shape.ndims
    if actual_rank not in expected_rank_dict:
        raise ValueError(
            "For the tensor `%s`, the actual rank "
            "`%d` (shape = %s) is not equal to the expected rank `%s`" %
            (name, actual_rank, str(tensor.shape), str(expected_rank)))


def get_shape_list(tensor, expected_rank=None, name=None):
    """Returns a list of the shape of tensor, preferring static dimensions.

    Args:
      tensor: A tf.Tensor object to find the shape of.
      expected_rank: (optional) int. The expected rank of `tensor`. If this is
        specified and the `tensor` has a different rank, and exception will be
        thrown.
      name: Optional name of the tensor for the error message.

    Returns:
      A list of dimensions of the shape of tensor. All static dimensions will
      be returned as python integers, and dynamic dimensions will be returned
      as tf.Tensor scalars.
    """
    if name is None and not tf.executing_eagerly():
        name = tensor.name

    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)

    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape


def pad_to_fixed_size(data, pad_value, output_shape, axis=0,
                      truncate=True):
    """
    Pads the data to be a fixed size in the dimensions specified by axis.

    :param data: n-dimensional input.
    :param pad_value: What we will pad with
    :param output_shape: The desired output shape. This has to cover everything, not just axis.
    :param truncate: If True (default), we will TRUNCATE in the dimensions specifed by axis if we're over.
    :param axis: The axes to pad in. Pass a list to pad multiple dims.
    :return:
    """
    axes = [axis] if isinstance(axis, int) else axis

    # Truncate if too long.
    pad_data = tf.identity(data)
    if truncate:
        slice_obj = [slice(0, os_i if i in axes else None, None) for i, os_i in enumerate(output_shape)]
        pad_data = pad_data[tuple(slice_obj)]

    # Anything not being padded, we assume is the output shape.
    current_shape = get_shape_list(pad_data, expected_rank=len(output_shape))
    for i, os_i in enumerate(output_shape):
        if i not in axes:
            current_shape[i] = os_i

    asserts = []
    for ax in axes:
        asserts.append(
            tf.Assert(tf.less_equal(current_shape[ax], output_shape[ax]), [current_shape[ax], output_shape[ax], ax])
        )

    with tf.control_dependencies(asserts):
        for ax in axes:
            pad_length = output_shape[ax] - current_shape[ax]
            pad_shape = [pad_length if i == ax else cs_i
                         for i, cs_i in enumerate(current_shape)]
            paddings = tf.fill(pad_shape, value=pad_value)
            pad_data = tf.concat([pad_data, paddings], axis=ax)

            # Update the dimension we padded in
            current_shape[ax] = output_shape[ax]

    pad_data = tf.reshape(pad_data, output_shape)
    return pad_data


def uniform_random_select(n, num_samples, sort_idx=True):
    """
    Randomly choose "num_samples" from N
    :param n:
    :param num_samples:
    :param sort_idx: Whether to sort the resulting index
    :return:
    """
    if isinstance(num_samples, int) and isinstance(n, int):
        assert num_samples <= n
    logits = tf.random.uniform([n])
    idx = tf.argsort(logits)[:num_samples]
    if sort_idx:
        idx = tf.sort(idx)
    idx = tf.cast(idx, dtype=tf.int32)
    return idx


def random_categorical_without_replacement(logits, num_samples):
    """
    Courtesy of https://github.com/tensorflow/tensorflow/issues/9260#issuecomment-437875125
    :param logits: [N] logits that are unscaled log probabilities
    :param num_samples:  <= N
    :return: num_samples inds that don't have repeatz
    """
    z = -tf.math.log(-tf.math.log(tf.random.uniform(tf.shape(logits), 0, 1)))
    _, indices = tf.nn.top_k(logits + z, num_samples)
    return tf.cast(indices, dtype=tf.int32)


def sample_bernoulli(p_a):
    if isinstance(p_a, float):
        if p_a == 0.0:
            print("sample_bernoulli p_a == 0.0: return False")
            return tf.constant(False)
        elif p_a == 1.0:
            print("sample_bernoulli p_a == 0.0: return True")
            return tf.constant(True)

    is_a = tf.random.categorical(tf.math.log([[1.0 - p_a, p_a]]), dtype=tf.int32, num_samples=1)
    is_a = tf.cast(tf.reshape(is_a, []), dtype=tf.bool)
    return is_a


def sample_bernoullis(p_a, N=1):
    if isinstance(p_a, float):
        if p_a == 0.0:
            print("sample_bernoulli p_a == 0.0: return False")
            return tf.constant([False for i in range(N)])
        elif p_a == 1.0:
            print("sample_bernoulli p_a == 0.0: return True")
            return tf.constant([True for i in range(N)])

    is_a = tf.random.categorical(tf.math.log([[1.0 - p_a, p_a]]), dtype=tf.int32, num_samples=N)
    is_a = tf.cast(tf.reshape(is_a, [N]), dtype=tf.bool)
    return is_a


def batch_index_iterator(len_l, batch_size, skip_end=True):
    """
    Provides indices that iterate over a list
    :param len_l: int representing size of thing that we will
        iterate over
    :param batch_size: size of each batch
    :param skip_end: if true, don't iterate over the last batch
    :return: A generator that returns (start, end) tuples
        as it goes through all batches
    """
    iterate_until = len_l
    if skip_end:
        iterate_until = (len_l // batch_size) * batch_size

    for b_start in range(0, iterate_until, batch_size):
        yield (b_start, min(b_start + batch_size, len_l))


def cumulative_maximum_int(x):
    """
    Returns the cumulative maximum of x over the last dimension
    :param x:
    :return:
    """
    assert x.dtype == tf.int32
    N = get_shape_list(x, 1)[0]
    x_tile = tf.tile(x[None], [N, 1])
    arange_x = tf.range(N)
    valid = tf.greater_equal(arange_x[:, None], arange_x[None])
    x_tile = tf.where(valid, x_tile, tf.fill([N, N], tf.int32.min))
    return tf.reduce_max(x_tile, -1)

def encode_string(tf_string, string_len):
    """
    Encodes the string into something TPU-able

    :param tf_string: string
    :param string_len: length
    :return: an encoded thing
    """
    out_raw = tf.cast(tf.io.decode_raw(tf_string, out_type=tf.uint8), dtype=tf.int32)[:string_len]
    return pad_to_fixed_size(out_raw, 0, [string_len])

def decode_string(x):
    import numpy as np
    return ''.join([chr(c) for c in x.astype(np.uint8) if c != 0])
