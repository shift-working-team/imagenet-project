import argparse
import os
import sys
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import yaml
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


WORKSPACE_ROOT = Path(
    os.environ.get("WORKSPACE_ROOT", "/workspace")
)

SRC_DIR = WORKSPACE_ROOT / "src"
PARAMS_PATH = WORKSPACE_ROOT / "params.yaml"

sys.path.append(str(SRC_DIR))

from models.swin import EncoderSwinTiny
from transforms.image_transform import get_classification_valid_transform
from visualization.generate_gradcam import (
    SwinClassifierWrapper,
    reshape_transform,
)


def load_params():
    with open(PARAMS_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_workspace_path(path_value):
    path = Path(path_value)

    if path.is_absolute():
        return path

    return WORKSPACE_ROOT / path


def load_class_names(raw_dir):
    class_names = sorted(
        [
            path.name
            for path in raw_dir.iterdir()
            if path.is_dir()
        ]
    )

    if not class_names:
        raise ValueError(
            f"No class directories found in {raw_dir}"
        )

    return class_names


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    model.load_state_dict(checkpoint)


def build_model():
    params = load_params()

    raw_dir = resolve_workspace_path(params["data"]["raw_dir"])
    class_names = load_class_names(raw_dir)

    device_name = params["train"]["device"]
    device = torch.device(
        device_name
        if device_name == "cuda" and torch.cuda.is_available()
        else "cpu"
    )

    model = EncoderSwinTiny(
        num_classes=len(class_names)
    ).to(device)

    checkpoint_path = resolve_workspace_path(
        params["classification"]["final_checkpoint"]
    )

    load_checkpoint(
        model,
        checkpoint_path,
        device,
    )

    model.eval()

    return {
        "params": params,
        "model": model,
        "device": device,
        "class_names": class_names,
        "transform": get_classification_valid_transform(),
        "checkpoint_path": checkpoint_path,
    }


APP_STATE = build_model()


def get_demo_config():
    return APP_STATE["params"].get("demo", {})


def make_gradcam_overlay(model, image, tensor, device):
    for param in model.backbone.parameters():
        param.requires_grad = True

    for param in model.classifier.parameters():
        param.requires_grad = True

    gradcam_model = SwinClassifierWrapper(model).to(device)
    gradcam_model.eval()

    resized_image = image.resize((224, 224))
    image_np = np.array(resized_image).astype(np.float32) / 255.0

    target_layer = model.backbone.features[-1][-1].norm2

    with GradCAM(
        model=gradcam_model,
        target_layers=[target_layer],
        reshape_transform=reshape_transform,
    ) as cam:
        grayscale_cam = cam(
            input_tensor=tensor,
        )[0]

    overlay = show_cam_on_image(
        image_np,
        grayscale_cam,
        use_rgb=True,
    )

    return Image.fromarray(overlay)


def predict(image, show_gradcam):
    if image is None:
        return None, "Please upload an image.", {}, []

    model = APP_STATE["model"]
    device = APP_STATE["device"]
    class_names = APP_STATE["class_names"]
    transform = APP_STATE["transform"]

    image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    demo_config = get_demo_config()
    top_k = min(
        int(demo_config.get("top_k", 5)),
        len(class_names),
    )
    top_probs, top_indices = torch.topk(
        probs,
        k=top_k,
    )

    top_probs = top_probs.detach().cpu().tolist()
    top_indices = top_indices.detach().cpu().tolist()

    confidences = {
        class_names[idx]: float(prob)
        for idx, prob in zip(top_indices, top_probs)
    }

    predicted_idx = top_indices[0]
    predicted_label = class_names[predicted_idx]
    predicted_confidence = top_probs[0]

    summary = (
        f"Prediction: {predicted_label} "
        f"({predicted_confidence * 100:.2f}%)"
    )

    table = [
        [
            rank,
            class_names[idx],
            f"{prob * 100:.2f}%",
        ]
        for rank, (idx, prob) in enumerate(
            zip(top_indices, top_probs),
            start=1,
        )
    ]

    gradcam_image = None
    if show_gradcam:
        gradcam_image = make_gradcam_overlay(
            model,
            image,
            tensor,
            device,
        )

    return gradcam_image, summary, confidences, table


def create_demo():
    demo_config = get_demo_config()
    title = "Image Classification Demo"
    description = (
        "Upload an image and classify it with the final checkpoint. "
        f"checkpoint: {APP_STATE['checkpoint_path']}"
    )

    with gr.Blocks(title=title) as demo:
        gr.Markdown(f"# {title}")
        gr.Markdown(description)

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(
                    type="pil",
                    label="Input Image",
                )
                gradcam_checkbox = gr.Checkbox(
                    value=bool(demo_config.get("show_gradcam", True)),
                    label="Show Grad-CAM",
                )
                predict_button = gr.Button(
                    "Predict",
                    variant="primary",
                )

            with gr.Column():
                gradcam_output = gr.Image(
                    type="pil",
                    label="Grad-CAM",
                )
                label_output = gr.Textbox(
                    label="Prediction",
                )
                confidence_output = gr.Label(
                    label="Top Prediction Scores",
                    num_top_classes=5,
                )
                table_output = gr.Dataframe(
                    headers=["Rank", "Class", "Confidence"],
                    datatype=["number", "str", "str"],
                    label="Top-5",
                    interactive=False,
                )

        predict_button.click(
            fn=predict,
            inputs=[
                image_input,
                gradcam_checkbox,
            ],
            outputs=[
                gradcam_output,
                label_output,
                confidence_output,
                table_output,
            ],
        )

    return demo


def parse_args():
    params = load_params()
    demo_config = params.get("demo", {})

    parser = argparse.ArgumentParser()
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

    demo = create_demo()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )
