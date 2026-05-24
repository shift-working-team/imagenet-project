import json
from collections import Counter
import re

def tokenizer(captions):
    text = captions.lower()
    text = re.sub(r"([.,!?])", r" \1 ", text)  # 특수문자 제거
    tokens = text.split()
    
    return tokens

def build_vocab(json_path, min_freq=3, max_size=10000):
    w2i = dict()
    i2w = dict()

    with open(json_path, 'r') as f:
        data = json.load(f)

    counter = Counter()

    for item in data:
        captions = item["captions"]
        for caption in captions:
            tokens = tokenizer(caption)
            counter.update(tokens)
        
    words = [w for w, freq in counter.most_common() if freq >= min_freq]

    voca = ["<pad>", "<sos>", "<eos>", "<unk>"]
    voca.extend(words[:max_size-4])
    voca_size = len(voca)

    for i, w in enumerate(voca):
        w2i[w] = i
        i2w[i] = w

    print(voca_size)

    return w2i, i2w, voca_size