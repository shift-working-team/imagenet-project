import json
import sentencepiece as spm
import yaml

# params
with open("/workspace/params.yaml", "r", encoding="utf-8") as f:
    params = yaml.safe_load(f)


def train_sentencepiece(
        json_path,
        model_prefix="sub_tokenizer",
        vocab_size=500,
        model_type="unigram"
    ):

    with open(json_path, 'r') as f:
        data = json.load(f)

    txt_path = "/workspace/src/dataset/sub_tokenizing_captions.txt"

    with open(txt_path, "w", encoding="utf-8") as f:
        for item in data:
            captions = item["captions"]

            for caption in captions:
                f.write(caption.lower() + "\n")

    spm.SentencePieceTrainer.train(
        input=txt_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,

        pad_piece="<pad>",
        bos_piece="<sos>",
        eos_piece="<eos>",
        unk_piece="<unk>",
        
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3
    )

    print("tokenizer training done")


if __name__ == "__main__":
    train_sentencepiece(
        json_path="/workspace/data/captioning/annotations/train.json",
        model_prefix="/workspace/src/dataset/sub_tokenizer",
        vocab_size=params["captioning"]["tokenizer"]["sp_vocab_size"],
        model_type="unigram"
    )