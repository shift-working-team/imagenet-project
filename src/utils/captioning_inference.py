from pathlib import Path

import sentencepiece as spm
import torch
import yaml

from dataset.build_vocab import build_vocab, tokenizer
from models.resnet18 import EncoderResnet18
from models.swin import EncoderSwinTiny
from models.transformer import DecoderTransformer
from models.vit import EncoderViTB16
from transforms.image_transform import get_caption_transform


def resolve_path(workspace_root, path_value):
    path = Path(path_value)
    if path.is_absolute():
        return path
    return workspace_root / path


def load_params(params_path):
    with open(params_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_device(params):
    device_name = params.get("train", {}).get("device", "cuda")
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_caption_vocab(params, workspace_root):
    cap_params = params["captioning"]
    tokenizer_params = cap_params["tokenizer"]

    train_caption_path = resolve_path(
        workspace_root,
        cap_params["data"]["train_caption"],
    )
    sp_model_path = resolve_path(
        workspace_root,
        tokenizer_params["sp_model_path"],
    )

    return build_vocab(
        str(train_caption_path),
        min_freq=tokenizer_params["min_freq"],
        max_size=tokenizer_params["max_vocab_size"],
        use_subword=tokenizer_params["use_subword"],
        sp_model_path=str(sp_model_path),
    )


def build_caption_models(params, voca_size, device):
    cap_params = params["captioning"]
    encoder_name = cap_params["encoder"]
    decoder_name = cap_params["decoder"]
    d_model = cap_params["transformer"]["d_model"]

    if decoder_name != "transformer":
        raise ValueError(
            "This captioning inference script supports transformer decoder only."
        )

    if encoder_name == "resnet18":
        encoder = EncoderResnet18(embed_size=d_model)
    elif encoder_name == "swin":
        encoder = EncoderSwinTiny(embed_size=d_model)
    elif encoder_name == "vit":
        encoder = EncoderViTB16(embed_size=d_model)
    else:
        raise ValueError(f"Unsupported caption encoder: {encoder_name}")

    decoder = DecoderTransformer(
        n_layers=cap_params["transformer"]["n_layers"],
        nhead=cap_params["transformer"]["nhead"],
        d_model=d_model,
        d_ff=d_model * 4,
        voca_size=voca_size,
        max_len=cap_params["max_caption_length"],
        drop_p=cap_params["transformer"]["drop_p"],
    )

    encoder = encoder.to(device)
    decoder = decoder.to(device)
    encoder.eval()
    decoder.eval()

    return encoder, decoder


def get_default_checkpoint_path(params, workspace_root):
    cap_params = params["captioning"]
    encoder_name = cap_params["encoder"]
    decoder_name = cap_params["decoder"]
    version = cap_params["version"]
    save_dir = resolve_path(workspace_root, cap_params["checkpoint"]["save_dir"])
    return save_dir / f"{encoder_name}-{decoder_name}_{version}_best.pt"


def load_caption_checkpoint(encoder, decoder, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    decoder.load_state_dict(checkpoint["decoder_state_dict"])

    return checkpoint


def decode_tokens(tokens, w2i, i2w, use_subword, sp_model_path=None):
    special_ids = {
        w2i[token]
        for token in ("<pad>", "<sos>", "<eos>")
        if token in w2i
    }

    if w2i.get("<eos>") in tokens:
        tokens = tokens[:tokens.index(w2i["<eos>"])]

    tokens = [token for token in tokens if token not in special_ids]

    if use_subword:
        sp = spm.SentencePieceProcessor()
        sp.load(str(sp_model_path))
        return sp.decode(tokens)

    words = [i2w.get(token, "<unk>") for token in tokens]
    return " ".join(words)


@torch.no_grad()
def generate_caption_from_tensor(
    image_tensor,
    encoder,
    decoder,
    w2i,
    i2w,
    params,
    device,
    sp_model_path=None,
    use_beam_search=None,
    beam_size=None,
):
    cap_params = params["captioning"]
    tokenizer_params = cap_params["tokenizer"]

    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)
    features = encoder(image_tensor, return_features=True)

    start_token = torch.full(
        (features.size(0),),
        w2i["<sos>"],
        dtype=torch.long,
        device=device,
    )

    if use_beam_search is None:
        use_beam_search = cap_params["beam_search"]["use_beam_search"]
    if beam_size is None:
        beam_size = cap_params["beam_search"]["beam_size"]

    if use_beam_search:
        generated_tokens, _, _ = decoder.generate_beam(
            features,
            start_token,
            w2i["<eos>"],
            beam_size,
        )
    else:
        generated_tokens, _, _ = decoder.generate(
            features,
            start_token,
            w2i["<eos>"],
        )

    return [
        decode_tokens(
            tokens,
            w2i,
            i2w,
            tokenizer_params["use_subword"],
            sp_model_path=sp_model_path,
        )
        for tokens in generated_tokens
    ]


def build_caption_runtime(workspace_root, checkpoint_path=None):
    workspace_root = Path(workspace_root)
    params_path = workspace_root / "params.yaml"
    params = load_params(params_path)
    device = get_device(params)
    w2i, i2w, voca_size = build_caption_vocab(params, workspace_root)
    encoder, decoder = build_caption_models(params, voca_size, device)

    if checkpoint_path is None:
        checkpoint_path = get_default_checkpoint_path(params, workspace_root)
    else:
        checkpoint_path = Path(checkpoint_path)

    checkpoint_path = resolve_path(workspace_root, checkpoint_path)
    checkpoint = load_caption_checkpoint(
        encoder,
        decoder,
        checkpoint_path,
        device,
    )

    sp_model_path = resolve_path(
        workspace_root,
        params["captioning"]["tokenizer"]["sp_model_path"],
    )

    return {
        "params": params,
        "device": device,
        "w2i": w2i,
        "i2w": i2w,
        "encoder": encoder,
        "decoder": decoder,
        "transform": get_caption_transform(),
        "checkpoint": checkpoint,
        "checkpoint_path": checkpoint_path,
        "sp_model_path": sp_model_path,
    }
