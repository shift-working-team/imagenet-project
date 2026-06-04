import argparse
import os
import sys
import tempfile
from pathlib import Path

import gradio as gr
import yaml


WORKSPACE_ROOT = Path(
    os.environ.get("WORKSPACE_ROOT", Path(__file__).resolve().parents[1])
)
SRC_DIR = WORKSPACE_ROOT / "src"
sys.path.append(str(SRC_DIR))

from utils.captioning_inference import (
    build_caption_runtime,
)


APP_STATE = None


def get_runtime(checkpoint_path=None):
    global APP_STATE
    if APP_STATE is None:
        APP_STATE = build_caption_runtime(
            WORKSPACE_ROOT,
            checkpoint_path=checkpoint_path,
        )
    return APP_STATE


def predict(image, use_beam_search, beam_size):
    if image is None:
        return "Please upload an image.", None, None

    runtime = get_runtime()
    params = runtime["params"]
    image = image.convert("RGB")
    image_tensor = runtime["transform"](image)

    tmp_dir = tempfile.mkdtemp(prefix="captioning_gradio_")
    try:
        image_tensor = image_tensor.unsqueeze(0).to(runtime["device"])
        features = runtime["encoder"](image_tensor, return_features=True)
        start_token = runtime["w2i"]["<sos>"]

        import torch

        start_token = torch.full(
            (features.size(0),),
            start_token,
            dtype=torch.long,
            device=runtime["device"],
        )

        if use_beam_search:
            generated_tokens, dec_atten, enc_dec_atten = runtime["decoder"].generate_beam(
                features,
                start_token,
                runtime["w2i"]["<eos>"],
                int(beam_size),
            )
        else:
            generated_tokens, dec_atten, enc_dec_atten = runtime["decoder"].generate(
                features,
                start_token,
                runtime["w2i"]["<eos>"],
            )

        from utils.captioning_inference import decode_tokens

        caption = decode_tokens(
            generated_tokens[0],
            runtime["w2i"],
            runtime["i2w"],
            params["captioning"]["tokenizer"]["use_subword"],
            sp_model_path=runtime["sp_model_path"],
        )

        caption_tokens = caption.split()
        layer = params["captioning"]["heatmap"]["layer"]
        dec_atten_path = Path(tmp_dir) / "decoder_attention.jpg"
        cross_atten_path = Path(tmp_dir) / "cross_attention.jpg"

        runtime["decoder"].show_dec_atten(
            dec_atten[0],
            caption_tokens,
            layer,
            str(dec_atten_path),
        )
        runtime["decoder"].show_cross_atten(
            enc_dec_atten[0],
            caption_tokens,
            layer,
            image_tensor.squeeze(0).detach().cpu(),
            str(cross_atten_path),
        )

        dec_image = str(dec_atten_path)
        cross_image = str(cross_atten_path)

        return caption, dec_image, cross_image
    except Exception:
        raise


def create_demo(checkpoint_path=None):
    runtime = get_runtime(checkpoint_path)
    params = runtime["params"]
    beam_config = params["captioning"]["beam_search"]

    with gr.Blocks(title="Image Captioning Demo") as demo:
        gr.Markdown("# Image Captioning Demo")
        gr.Markdown(f"checkpoint: {runtime['checkpoint_path']}")

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(
                    type="pil",
                    label="Input Image",
                )
                use_beam_search = gr.Checkbox(
                    value=bool(beam_config["use_beam_search"]),
                    label="Use Beam Search",
                )
                beam_size = gr.Slider(
                    minimum=1,
                    maximum=10,
                    step=1,
                    value=int(beam_config["beam_size"]),
                    label="Beam Size",
                )
                caption_button = gr.Button(
                    "Generate Caption",
                    variant="primary",
                )

            with gr.Column():
                caption_output = gr.Textbox(
                    label="Generated Caption",
                    lines=4,
                )
                dec_atten_output = gr.Image(
                    type="filepath",
                    label="Decoder Attention",
                )
                cross_atten_output = gr.Image(
                    type="filepath",
                    label="Cross Attention",
                )

        caption_button.click(
            fn=predict,
            inputs=[
                image_input,
                use_beam_search,
                beam_size,
            ],
            outputs=[
                caption_output,
                dec_atten_output,
                cross_atten_output,
            ],
        )

    return demo


def parse_args():
    with open(WORKSPACE_ROOT / "params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    demo_config = params.get("demo", {})

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=None,
    )
    parser.add_argument(
        "--host",
        default=os.environ.get(
            "GRADIO_SERVER_NAME",
            demo_config.get("host", "0.0.0.0"),
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(
            os.environ.get(
                "GRADIO_SERVER_PORT",
                demo_config.get("port", 7860),
            )
        ),
    )
    parser.add_argument(
        "--share",
        action="store_true",
        default=(
            os.environ.get(
                "GRADIO_SHARE",
                str(demo_config.get("share", False)),
            ).lower()
            == "true"
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    demo = create_demo(checkpoint_path=args.checkpoint)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )
