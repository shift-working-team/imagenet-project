import json
from collections import Counter
import re
import sentencepiece as spm

def tokenizer(captions):
    text = captions.lower()
    text = re.sub(r"([.,!?])", r" \1 ", text)  # 특수문자 제거
    tokens = text.split()
    
    return tokens

def sub_tokenizer(caption, sp):
    tokens = sp.encode(caption, out_type=str)

    return tokens


def build_vocab(json_path, min_freq=3, max_size=10000, use_subword=False, sp_model_path="/workspace/src/dataset/sub_tokenizer.model"):
    w2i = dict()
    i2w = dict()

    # ==================================================
    # SentencePiece tokenizer 사용
    # ==================================================
    if use_subword:

        sp = spm.SentencePieceProcessor()
        sp.load(sp_model_path)

        voca_size = sp.get_piece_size()

        for i in range(voca_size):
            token = sp.id_to_piece(i)

            w2i[token] = i
            i2w[i] = token
    else:
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