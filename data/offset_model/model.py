import regex as re
import numpy as np
import json
import os

def _count_vowels(word):
    return len(re.findall('a|e|i|o|u', word.lower()))
def _count_punctuation(word):
    return len(re.findall('\W', word.lower()))

def get_features(vtt):
    """
    Add features to a pandas table
    :param vtt:
    :return:
    """
    def add_feature(feature_name, x, default_value=0.0):
        x = x.astype(np.float32)
        assert x.shape[0] == vtt.shape[0]
        vtt[f'feat_{feature_name}'] = x
        vtt[f'feat_{feature_name}_left'] = np.concatenate([[default_value], x[:-1]], 0)
        vtt[f'feat_{feature_name}_right'] = np.concatenate([x[1:], [default_value]], 0)

    # Get features, just plug it into the DF I guess idk
    x_charlen = vtt['word'].apply(len).values.astype(np.float32)
    add_feature('charlen', x_charlen, default_value=1.0)
    add_feature('bpelen', vtt['encoded'].apply(len).values.astype(np.float32), default_value=0.0)
    add_feature('is_upper', vtt['word'].apply(lambda x: x[:1].isupper()).values.astype(np.float32), default_value=0.0)


    chunk_len = vtt['chunk_len'] = (vtt['end'] - vtt['start']).values.astype(np.float32)
    add_feature('chunklen', chunk_len, default_value=0.1)

    add_feature('num_vowels', vtt['word'].apply(_count_vowels).values.astype(np.float32), default_value=0.0)
    add_feature('num_punct', vtt['word'].apply(_count_punctuation).values.astype(np.float32), default_value=0.0)

cols = ['feat_charlen',
         'feat_charlen_left',
         'feat_charlen_right',
         'feat_bpelen',
         'feat_bpelen_left',
         'feat_bpelen_right',
         'feat_is_upper',
         'feat_is_upper_left',
         'feat_is_upper_right',
         'feat_chunklen',
         'feat_chunklen_left',
         'feat_chunklen_right',
         'feat_num_vowels',
         'feat_num_vowels_left',
         'feat_num_vowels_right',
         'feat_num_punct',
         'feat_num_punct_left',
         'feat_num_punct_right']

with open(os.path.join(os.path.dirname(os.path.join(__file__)), 'model_params.json'), 'r') as f:
    model_params = json.load(f)

hidden_size=32
mean = np.array(model_params.pop('mean'), dtype=np.float32)
std = np.array(model_params.pop('std'), dtype=np.float32)
w0 = np.array(model_params.pop('mapping.0.weight'), dtype=np.float32).reshape((hidden_size, len(cols)))
b0 = np.array(model_params.pop('mapping.0.bias'), dtype=np.float32)

w1 = np.array(model_params.pop('mapping.2.weight'), dtype=np.float32).reshape((hidden_size, hidden_size))
b1 = np.array(model_params.pop('mapping.2.bias'), dtype=np.float32)

w2 = np.array(model_params.pop('mapping.4.weight'), dtype=np.float32).reshape((2, hidden_size))
b2 = np.array(model_params.pop('mapping.4.bias'), dtype=np.float32)

temperature = np.exp(np.array(model_params.pop('temp'), dtype=np.float32))
bias = np.array(model_params.pop('bias'), dtype=np.float32)

def predict_offsets(vtt):
    feats_np = (vtt[cols].values - mean[None])/std[None]

    h0 = np.maximum(feats_np @ w0.T + b0[None], 0.0)
    h1 = np.maximum(h0 @ w1.T + b1[None], 0.0)
    preds = h1 @ w2.T + b2[None]
    preds = np.tanh(preds) * temperature + bias
    return preds
