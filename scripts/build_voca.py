import json
from collections import Counter
import re

def tokenize(captions):
    text = captions.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)  # 특수문자 제거
    tokens = text.split()
    
    return tokens

def build_voca(json_path, min_freq=5, max_size=None):
    w2i = dict()
    i2w = dict()

    with open(json_path, 'r') as f:
        data = json.load(f)

    counter = Counter()

    for item in data:
        captions = item["captions"]
        for caption in captions:
            tokens = tokenize(caption)
            counter.update(tokens)
        
    words = [w for w, freq in counter.most_common() if freq >= min_freq]
    
    voca = ["<pad>", "<sos>", "<eos>", "<unk>"]
    voca.extend(words[:max_size])
    voca_size = len(voca)

    for i, w in enumerate(voca):
        w2i[w] = i
        i2w[i] = w

    return w2i, i2w, voca_size


json_path = '/workspace/data/annotations/captions_flo.json'

w2i, i2w, voca_size = build_voca(json_path, 5, 10000)

print(voca_size)
print(i2w[1])