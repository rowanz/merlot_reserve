from itertools import islice
import json

factor = 6

source = json.load(open(f'test_infill.json', 'r'))
chunk_size = len(source) // factor

split_size = [chunk_size] * factor + [len(source) % factor]
assert sum(split_size) == len(source)

source_iter = iter(source)
splits = [list(islice(source_iter, elem)) for elem in split_size]

for i, split in enumerate(splits):
    json.dump(split, open(f'test_infill.json.{i}', 'w'))

