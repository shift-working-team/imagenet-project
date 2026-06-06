import os
import sys
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import yaml
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


WORKSPACE_ROOT = Path(
    os.environ.get("WORKSPACE_ROOT", Path(__file__).resolve().parents[1])
)
SRC_DIR = WORKSPACE_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from src.models.swin import EncoderSwinTiny
from src.transforms.image_transform import get_classification_valid_transform
from src.utils.captioning_inference import build_caption_runtime, decode_tokens
from src.visualization.generate_gradcam import (
    SwinClassifierWrapper,
    reshape_transform,
)


CLASSIFICATION_STATE = None
CAPTIONING_STATE = None


def load_params():
    """params.yaml을 읽어서 데모, 모델, 체크포인트 설정을 가져온다."""
    with open(WORKSPACE_ROOT / "params.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# params.yaml의 demo.class_names에서 학습 당시 클래스 목록을 가져온다.
def load_class_names(params):
    class_names = params.get("demo", {}).get("class_names", [])

    if not isinstance(class_names, list) or not all(
        isinstance(class_name, str)
        for class_name in class_names
    ):
        raise ValueError("demo.class_names must be a list of class name strings.")

    if not class_names:
        raise ValueError("No class names found in params.yaml demo.class_names.")

    return class_names

# CUDA 사용 가능 여부를 기준으로 장치를 선택
def get_device(params):
    device_name = params.get("train", {}).get("device", "cuda")

    # 설정이 cuda이고 실제 CUDA가 있으면 GPU를 사용한다.
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def load_classification_checkpoint(model, checkpoint_path, device):
    """분류 모델 체크포인트를 로드하고 model_state_dict 형식이면 내부 state_dict만 꺼낸다."""
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    # 저장 포맷이 {"model_state_dict": ...} 형태인 경우 실제 가중치만 사용한다.
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    model.load_state_dict(checkpoint)


def build_classification_runtime():
    """분류 모델, transform, 클래스명, 체크포인트 경로를 묶은 런타임 상태를 만든다."""
    params = load_params()
    model_name = params["classification"]["model_name"]

    # 현재 Grad-CAM wrapper와 모델 생성 로직은 Swin-T 전용이므로 다른 모델은 명시적으로 막는다.
    if model_name != "swin_t":
        raise ValueError(
            "The combined Gradio demo currently supports only swin_t "
            f"for classification, got: {model_name}"
        )

    class_names = load_class_names(params)
    device = get_device(params)

    model = EncoderSwinTiny(
        num_classes=len(class_names)
    ).to(device)

    checkpoint_path = WORKSPACE_ROOT / params["classification"]["final_checkpoint"]
    load_classification_checkpoint(
        model,
        checkpoint_path,
        device,
    )
    model.eval()

    return {
        "params": params,
        "model": model,
        "model_name": model_name,
        "device": device,
        "class_names": class_names,
        "transform": get_classification_valid_transform(),
        "checkpoint_path": checkpoint_path,
    }


def get_classification_runtime():
    """분류 런타임을 최초 요청 시 한 번만 만들고 이후에는 캐시된 상태를 재사용한다."""
    global CLASSIFICATION_STATE

    # 버튼 클릭 전에는 모델을 로드하지 않고, 첫 예측 시점에만 로드한다.
    if CLASSIFICATION_STATE is None:
        CLASSIFICATION_STATE = build_classification_runtime()

    return CLASSIFICATION_STATE


def get_caption_checkpoint_path(params):
    """캡셔닝 체크포인트 경로를 params.yaml에서 우선 찾고, 없으면 기본 파일명 규칙으로 만든다."""
    checkpoint_config = params["captioning"]["checkpoint"]
    final_checkpoint = checkpoint_config.get("final_checkpoint")

    # final_checkpoint가 명시되어 있으면 그 파일을 우선 사용한다.
    if final_checkpoint:
        return WORKSPACE_ROOT / checkpoint_config["save_dir"] / final_checkpoint

    # 명시 경로가 없으면 학습 코드의 encoder-decoder_version_best.pt 규칙으로 fallback한다.
    encoder_name = params["captioning"]["encoder"]
    decoder_name = params["captioning"]["decoder"]
    version = params["captioning"]["version"]
    return (
        WORKSPACE_ROOT
        / checkpoint_config["save_dir"]
        / f"{encoder_name}-{decoder_name}_{version}_best.pt"
    )


def get_captioning_runtime():
    """캡셔닝 런타임을 최초 요청 시 한 번만 만들고 이후에는 캐시된 상태를 재사용한다."""
    global CAPTIONING_STATE

    # 캡셔닝 탭을 실제로 실행하기 전까지 encoder/decoder 로딩을 미룬다.
    if CAPTIONING_STATE is None:
        params = load_params()
        CAPTIONING_STATE = build_caption_runtime(
            WORKSPACE_ROOT,
            checkpoint_path=get_caption_checkpoint_path(params),
        )
    return CAPTIONING_STATE


def make_gradcam_overlay(model, image, tensor, device):
    """분류 모델의 마지막 Swin block을 대상으로 Grad-CAM overlay 이미지를 생성한다."""
    # Grad-CAM은 gradient가 필요하므로 frozen backbone/classifier도 일시적으로 gradient를 켠다.
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
        grayscale_cam = cam(input_tensor=tensor)[0]

    overlay = show_cam_on_image(
        image_np,
        grayscale_cam,
        use_rgb=True,
    )

    return Image.fromarray(overlay)


def predict_classification(image, show_gradcam):
    """업로드된 이미지를 분류하고, 선택 시 Grad-CAM 결과까지 함께 반환한다."""
    # 이미지가 없으면 Gradio 출력 개수에 맞춰 빈 결과를 반환한다.
    if image is None:
        return None, "Please upload an image.", {}, []

    runtime = get_classification_runtime()
    params = runtime["params"]
    model = runtime["model"]
    device = runtime["device"]
    class_names = runtime["class_names"]
    transform = runtime["transform"]

    image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    top_k = max(
        1,
        min(
            int(params["demo"].get("top_k", 5)),
            len(class_names),
        ),
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

    # 사용자가 체크박스를 켠 경우에만 비용이 큰 Grad-CAM을 생성한다.
    if show_gradcam:
        gradcam_image = make_gradcam_overlay(
            model,
            image,
            tensor,
            device,
        )

    return gradcam_image, summary, confidences, table


def caption_token_labels(generated_tokens, runtime, caption):
    """attention heatmap 제목으로 사용할 생성 토큰 라벨을 만든다."""
    special_ids = {
        runtime["w2i"].get("<pad>"),
        runtime["w2i"].get("<sos>"),
        runtime["w2i"].get("<eos>"),
    }
    labels = [
        runtime["i2w"].get(token, "<unk>")
        for token in generated_tokens
        if token not in special_ids
    ]

    # 토큰 id 기반 라벨이 있으면 attention 길이와 맞기 쉬운 이 라벨을 사용한다.
    if labels:
        return labels

    # 예외적으로 라벨이 비어 있으면 문장 문자열을 단어 단위로 나눠 fallback한다.
    return caption.split()


@torch.no_grad()
def predict_captioning(image):
    """업로드된 이미지에 대해 캡션을 생성하고 cross-attention heatmap들을 반환한다."""
    # 이미지가 없으면 Gradio 출력 개수에 맞춰 빈 결과를 반환한다.
    if image is None:
        return "Please upload an image.", []

    runtime = get_captioning_runtime()
    params = runtime["params"]
    image = image.convert("RGB")
    image_tensor = runtime["transform"](image)
    image_tensor = image_tensor.unsqueeze(0).to(runtime["device"])

    features = runtime["encoder"](
        image_tensor,
        return_features=True,
    )
    start_token = torch.full(
        (features.size(0),),
        runtime["w2i"]["<sos>"],
        dtype=torch.long,
        device=runtime["device"],
    )

    beam_config = params["captioning"]["beam_search"]
    use_beam_search = beam_config.get("use_beam_search", True)
    beam_size = beam_config.get("beam_size", 3)

    # params.yaml에서 beam search를 켠 경우 여러 후보를 탐색해 캡션을 생성한다.
    if use_beam_search:
        generated_tokens, _, enc_dec_atten = runtime["decoder"].generate_beam(
            features,
            start_token,
            runtime["w2i"]["<eos>"],
            beam_size,
        )
    else:
        # beam search를 끈 경우 매 step에서 가장 확률 높은 토큰을 선택하는 greedy 생성을 사용한다.
        generated_tokens, _, enc_dec_atten = runtime["decoder"].generate(
            features,
            start_token,
            runtime["w2i"]["<eos>"],
        )

    caption = decode_tokens(
        generated_tokens[0],
        runtime["w2i"],
        runtime["i2w"],
        params["captioning"]["tokenizer"]["use_subword"],
        sp_model_path=runtime["sp_model_path"],
    )

    caption_tokens = caption_token_labels(
        generated_tokens[0],
        runtime,
        caption,
    )

    tmp_dir = tempfile.mkdtemp(prefix="combined_captioning_gradio_")
    heatmap_images = []
    n_layers = len(runtime["decoder"].layers)

    # 각 decoder layer별 cross-attention heatmap 이미지를 만들어 Gallery에 표시한다.
    for layer in range(1, n_layers + 1):
        cross_atten_path = Path(tmp_dir) / f"cross_attention_layer_{layer}.jpg"
        runtime["decoder"].show_cross_atten(
            enc_dec_atten[0],
            caption_tokens,
            layer,
            image_tensor.squeeze(0).detach().cpu(),
            str(cross_atten_path),
        )
        heatmap_images.append(
            (
                str(cross_atten_path),
                f"Layer {layer}",
            )
        )

    return caption, heatmap_images


def create_demo():
    """분류 탭과 캡셔닝 탭을 가진 하나의 Gradio Blocks 앱을 만든다."""
    params = load_params()
    top_k = max(1, int(params["demo"].get("top_k", 5)))
    caption_checkpoint = get_caption_checkpoint_path(params)

    with gr.Blocks(title="ImageNet Classification and Captioning Demo") as demo:
        gr.Markdown("# ImageNet Classification and Captioning Demo")

        with gr.Tabs():
            with gr.Tab("Classification"):
                gr.Markdown(
                    "Upload an image and classify it with the final checkpoint."
                )
                gr.Markdown(
                    f"checkpoint: {WORKSPACE_ROOT / params['classification']['final_checkpoint']}"
                )

                with gr.Row():
                    with gr.Column():
                        classification_image_input = gr.Image(
                            type="pil",
                            label="Input Image",
                        )
                        gradcam_checkbox = gr.Checkbox(
                            value=bool(params["demo"].get("show_gradcam", True)),
                            label="Show Grad-CAM",
                        )
                        classification_button = gr.Button(
                            "Predict",
                            variant="primary",
                        )

                    with gr.Column():
                        gradcam_output = gr.Image(
                            type="pil",
                            label="Grad-CAM",
                        )
                        classification_summary_output = gr.Textbox(
                            label="Prediction",
                        )
                        confidence_output = gr.Label(
                            label="Top Prediction Scores",
                            num_top_classes=top_k,
                        )
                        table_output = gr.Dataframe(
                            headers=["Rank", "Class", "Confidence"],
                            datatype=["number", "str", "str"],
                            label=f"Top-{top_k}",
                            interactive=False,
                        )

                classification_button.click(
                    fn=predict_classification,
                    inputs=[
                        classification_image_input,
                        gradcam_checkbox,
                    ],
                    outputs=[
                        gradcam_output,
                        classification_summary_output,
                        confidence_output,
                        table_output,
                    ],
                )

            with gr.Tab("Captioning"):
                gr.Markdown(
                    "Upload an image and generate a caption with cross-attention heatmaps."
                )
                gr.Markdown(f"checkpoint: {caption_checkpoint}")

                with gr.Row():
                    with gr.Column():
                        captioning_image_input = gr.Image(
                            type="pil",
                            label="Input Image",
                        )
                        captioning_button = gr.Button(
                            "Generate Caption",
                            variant="primary",
                        )

                    with gr.Column():
                        caption_output = gr.Textbox(
                            label="Generated Caption",
                            lines=4,
                        )
                        cross_atten_output = gr.Gallery(
                            label="Cross Attention Heatmaps",
                            columns=2,
                            object_fit="contain",
                            height="auto",
                        )

                captioning_button.click(
                    fn=predict_captioning,
                    inputs=[captioning_image_input],
                    outputs=[
                        caption_output,
                        cross_atten_output,
                    ],
                )

    return demo


if __name__ == "__main__":
    params = load_params()

    demo = create_demo()
    demo.launch(
        server_name=params["demo"]["host"],
        server_port=params["demo"]["port"],
        share=params["demo"]["share"],
    )
