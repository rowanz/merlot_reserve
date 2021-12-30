import json

with open(f'MSRVTT-QA/train_qa.json') as f:
    ds = json.load(f)

freq_dict = {}
for item in ds:
    answer = item['answer']
    if answer not in freq_dict:
        freq_dict[answer] = 0
    freq_dict[answer] += 1

ans_count = sorted([(k, v) for k, v in freq_dict.items()], key=lambda x: x[1], reverse=True)

dic_size = 2000
min_count = min([x[1] for x in ans_count][:dic_size])
cand_set = [x[0] for x in ans_count if x[1] >= min_count]
print(len(cand_set))

with open(f'MSRVTT-QA/vocab_2k.json', 'w') as f:
    json.dump(cand_set, f)

