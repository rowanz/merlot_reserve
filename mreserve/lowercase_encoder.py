import sys
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from tokenizers import Tokenizer, models, processors, trainers, pre_tokenizers, decoders, normalizers
from tokenizers.models import BPE
import itertools
# Place special tokens here
PADDING = 0
START = 1
END = 2
MASK = 3
MASKAUDIO = 4
AUDIOSPAN = 5
LTOVPOOL = 6
RESETCTX = 9

SPECIAL_TOKENS = ['<|PAD|>', '<|START|>', '<|END|>','<|MASK|>', '<|MASKAUDIO|>', '<|AUDIOSPAN|>', '<|LTOVPOOL|>'] + [f'<|unused{i}|>' for i in range(3)]

def get_encoder():
    directory_name = os.path.dirname(__file__)
    fn = os.path.join(directory_name, 'lowercase_encoder.json')
    tokenizer = Tokenizer.from_file(fn)
    return tokenizer