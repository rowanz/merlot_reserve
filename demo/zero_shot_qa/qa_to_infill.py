prefix = '''Input: what is a car being driven through?
Output: a car is being driven through _.

Input: who are running across screen?
Output: _ are running across screen.

Input: when is a girl performing?
Output: a girl is performing at _.

Input: what is a cartoon doing?
Output: a cartoon is _.

Input: how many women talk in a bedroom?
Output: _ women talk in a bedroom.

Input: what a man playing while dancing with others?
Output: a man is playing _ while dancing with others.

Input: where is a flag hoisted?
Output: a flag is hoisted in _.

Input: who talks to another man on the couch?
Output: _ talks to another man on the couch.

Input: what does a teenage girl try to get at a public restroom?
Output: a teenage girl tries to get _ at a public restroom.

Input: when do the models walk as the audience watches?
Output: the models walk as the audience watches at _.

Input: what shows a person killing animals in a green forest?
Output: _ shows a person killing animals in a green forest.

Input: who does a man ask to go on a date?
Output: a man asks _ to go on a date.

Input: what are three people sitting on?
Output: three people are sitting on _.

Input: %s
Output:'''

import json
from tqdm import tqdm

def request(prompt, temperature):
    return "insert your GPT3 code / API key here"

split = 'test'
print(f'MSRVTT-QA/{split}_qa.json')

with open(f'MSRVTT-QA/{split}_qa.json') as f:
    ds = json.load(f)

# ds = ds[:10]

bad_templates = []
for item in tqdm(ds):
    prompt = prefix % item['question']
    transformed_query = request(prompt, temperature=0.1)[0]
    if transformed_query.count('_') == 1:
        item['question'] = transformed_query.replace('_', '<|MASK|>')
    else:
        item['bad_template'] = transformed_query.replace('_', '<|MASK|>')
        bad_templates.append(item)

with open(f'MSRVTT-QA/{split}_infill.json', 'w') as f:
    json.dump(ds, f, indent=4)

with open(f'MSRVTT-QA/{split}_bad_infill.json', 'w') as f:
    json.dump(bad_templates, f, indent=4)
