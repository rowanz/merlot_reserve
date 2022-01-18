"""
Aggressive sanitizing of text
"""

import os
import string
import re
import unidecode
from num2words import num2words
import random
import ftfy
import demoji


# Simple python script to move the files around

def ascii_map():
    data = {'…': '', '_': '', '`': '\''}
    for num in range(256):
        h = num
        filename = 'x{num:03x}'.format(num=num)
        try:
            mod = __import__('unidecode.' + filename, fromlist=True)
        except ImportError:
            pass
        else:
            for l, val in enumerate(mod.data):
                i = h << 8
                i += l
                if i >= 0x80:
                    data[i] = val
    return data

char_map = ascii_map()

# Brackets
char_map[ord('(')] = ' '
char_map[ord('<')] = ' '
char_map[ord(')')] = ' '
char_map[ord('>')] = ' '
char_map[ord('{')] = ' '
char_map[ord('}')] = ' '
char_map[ord('[')] = ' '
char_map[ord(']')] = ' '


char_map[ord('`')] = "'"
char_map[ord('^')] = " "
char_map[ord('_')] = " "
char_map[ord('|')] = " "
char_map[ord('~')] = " "
char_map[ord('-')] = " "
char_map[ord(' ')] = " "


FAST_UNIDECODE = str.maketrans(char_map)

spellout_map = {
    '&': ' and ',
    '/': ' slash ',
    '@': ' at ',
    '\\': ' backslash ',
    '+': ' plus ',
    '%': ' percent ',
    '=': ' equals ',
}
SPELLOUT_MAP = str.maketrans(spellout_map)


is_valid = re.compile(r"^[ A-Za-z0-9.,?!']*$")

def _fix_time(groups):
    hours = int(groups.group(1))
    minutes = int(groups.group(2))
    if minutes == 0:
        return num2words(hours)
    elif minutes < 10:
        return '{} oh {}'.format(num2words(hours), num2words(minutes))
    return '{} {}'.format(num2words(hours), num2words(minutes))

def clean_text(text):
    # Elipses and underscores are dumb. also gonna do my own version of unidecode here
    # text_orig = text
    text = ftfy.ftfy(text)
    all_emojis = demoji.findall(text)

    # kill emojis
    for k, v in all_emojis.items():
        text = text.replace(k, f'[{v}]'.replace(' ', ''))

    text = re.sub(r'(<p>|<strong>|><p>|<br>|<em>|<span>|\[unreadable\])', '', text)
    text = text.translate(FAST_UNIDECODE)
    text = text.translate(FAST_UNIDECODE)

    # Remove duplicate punctuation
    text = re.sub(r'([\-$%&\'+,./:;?!@\[\]\\_’\"\=])\1+', r'\1', text)

    # spell out text
    text = text.translate(SPELLOUT_MAP)

    # remove newlines and multiple spaces
    text = re.sub(r'\n', ' ', text.strip())
    text = re.sub(r'\s+', ' ', text)
    text = re.sub('[ \t\r\f\v]+', ' ', text)

    # do REALLY AGGRESSIVE stuff
    if random.random() > 0.5:
        text = re.sub(r'\$(\d+)(\.\d+)?', lambda x: num2words(int(x.group(1))) + ' dollars', text) # no money
        text = re.sub(r'(\d+(?:rd|nd|th|st))', lambda x: num2words(int(x.group(1)[:-2]), to='ordinal'), text)
        text = re.sub(r'\b(\d{1,2})\:(\d\d)\b', _fix_time, text)

        # spell out all numbers less than 100
        text = re.sub(r'(\d+)',
                       lambda x: num2words(int(x.group(1))) if int(x.group(1)) < 100 else x.group(1), text)
    return text
