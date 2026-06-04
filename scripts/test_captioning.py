import argparse
import os
import sys
from pathlib import Path

from PIL import Image
import torch


WORKSPACE_ROOT = Path(
    os.environ.get("WORKSPACE_ROOT", Path(__file__).resolve().parents[1])
)
SRC_DIR = WORKSPACE_ROOT / "src"
sys.path.append(str(SRC_DIR))

from utils.captioning_inference import (
    build_caption_runtime,
    decode_tokens,
    resolve_path,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a caption and attention heatmaps for one image."
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Input image path.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint path. Defaults to captioning.checkpoint.save_dir/<encoder>-<decoder>_<version>_best.pt.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/captioning/test_attention",
        help="Directory to save attention heatmaps.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Transformer layer index for heatmaps. Defaults to captioning.heatmap.layer.",
    )
    parser.add_argument(
        "--beam-search",
        action="store_true",
        help="Use beam search regardless of params.yaml.",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=None,
    )
    return parser.parse_args()


@torch.no_grad()
def generate_caption_with_attention(runtime, image_tensor, use_beam_search, beam_size):
    params = runtime["params"]
    w2i = runtime["w2i"]
    device = runtime["device"]
    encoder = runtime["encoder"]
    decoder = runtime["decoder"]

    image_tensor = image_tensor.unsqueeze(0).to(device)
    features = encoder(image_tensor, return_features=True)
    start_token = torch.full(
        (features.size(0),),
        w2i["<sos>"],
        dtype=torch.long,
        device=device,
    )

    if use_beam_search:
        generated_tokens, dec_atten, enc_dec_atten = decoder.generate_beam(
            features,
            start_token,
            w2i["<eos>"],
            beam_size,
        )
    else:
        generated_tokens, dec_atten, enc_dec_atten = decoder.generate(
            features,
            start_token,
            w2i["<eos>"],
        )

    caption = decode_tokens(
        generated_tokens[0],
        w2i,
        runtime["i2w"],
        params["captioning"]["tokenizer"]["use_subword"],
        sp_model_path=runtime["sp_model_path"],
    )

    return caption, generated_tokens[0], dec_atten[0], enc_dec_atten[0]


def main():
    args = parse_args()
    runtime = build_caption_runtime(
        WORKSPACE_ROOT,
        checkpoint_path=args.checkpoint,
    )
    params = runtime["params"]

    image_path = resolve_path(WORKSPACE_ROOT, args.image)
    output_dir = resolve_path(WORKSPACE_ROOT, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path).convert("RGB")
    image_tensor = runtime["transform"](image)

    use_beam_search = (
        True
        if args.beam_search
        else params["captioning"]["beam_search"]["use_beam_search"]
    )
    beam_size = args.beam_size or params["captioning"]["beam_search"]["beam_size"]
    layer = args.layer or params["captioning"]["heatmap"]["layer"]

    caption, _, dec_atten, enc_dec_atten = generate_caption_with_attention(
        runtime,
        image_tensor,
        use_beam_search=use_beam_search,
        beam_size=beam_size,
    )

    stem = image_path.stem
    dec_atten_path = output_dir / f"{stem}_decoder_attention.jpg"
    cross_atten_path = output_dir / f"{stem}_cross_attention.jpg"
    caption_tokens = caption.split()

    runtime["decoder"].show_dec_atten(
        dec_atten,
        caption_tokens,
        layer,
        str(dec_atten_path),
    )
    runtime["decoder"].show_cross_atten(
        enc_dec_atten,
        caption_tokens,
        layer,
        image_tensor,
        str(cross_atten_path),
    )

    print(f"Image: {image_path}")
    print(f"Caption: {caption}")
    print(f"Decoder attention: {dec_atten_path}")
    print(f"Cross attention: {cross_atten_path}")


if __name__ == "__main__":
    main()
