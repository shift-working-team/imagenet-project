import sys
sys.path.append("/workspace/src")

import os
import json
import yaml
import wandb
import pandas as pd
import shutil
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt

from dataset.classification_dataset import (
    ClassificationDataset
)

from transforms.image_transform import (
    get_classification_valid_transform
)

from models.swin import EncoderSwinTiny
from visualization.generate_gradcam import save_gradcam




# params
with open(
    "/workspace/params.yaml",
    "r",
    encoding="utf-8"
) as f:
    params = yaml.safe_load(f)




# device
device = torch.device(
    params["train"]["device"]
    if torch.cuda.is_available()
    else "cpu"
)




# class mapping
classes = sorted(
    [
        cls
        for cls in os.listdir(
            params["data"]["raw_dir"]
        )
        if os.path.isdir(
            os.path.join(
                params["data"]["raw_dir"],
                cls
            )
        )
    ]
)

class_to_idx = {
    cls: idx
    for idx, cls in enumerate(classes)
}

idx_to_class = {
    idx: cls
    for cls, idx in class_to_idx.items()
}

num_classes = len(classes)




# dataset
test_dataset = ClassificationDataset(
    root_dir=params["data"]["raw_dir"],
    class_to_idx=class_to_idx,
    split="test",
    transform=get_classification_valid_transform()
)

test_loader = DataLoader(
    test_dataset,
    batch_size=params["train"]["batch_size"],
    shuffle=False,
    num_workers=params["train"]["num_workers"]
)




# model
model = EncoderSwinTiny(
    num_classes=num_classes
).to(device)




# checkpoint
checkpoint_path = (
    params["classification"]["final_checkpoint"]
)

model.load_state_dict(
    torch.load(
        checkpoint_path,
        map_location=device
    )
)

model.eval()


# scheduler tag
scheduler_tag = (
    params["classification"]["scheduler"]["name"]
    if params["classification"]["scheduler"]["use"]
    else "none"
)

my_config = {
    "model_name": (
        params["classification"]["model_name"]
    ),
    "batch_size": (
        params["train"]["batch_size"]
    ),
    "image_size": (
        params["preprocess"]["image_size"]
    ),
    "seed": (
        params["train"]["seed"]
    ),
    "dataset_version": (
        params["data"]["dataset_version"]
    ),
    "optimizer": (
        params["classification"]["optimizer"]
    ),
    "scheduler": scheduler_tag,
    "weight_decay": (
        params["classification"]["weight_decay"]
    ),
    "label_smoothing": (
        params["classification"]["label_smoothing"]
    ),
    "checkpoint": (
        params["classification"]["final_checkpoint"]
    )
}

wandb.init(
    project=params["logging"]["project_name"],
    entity="super-shift-working",
    name="cls_final_test",

    config=my_config,

    tags=[
        "classification",
        "final_test",

        f"model:{params['classification']['model_name']}",
        f"dataset:{params['data']['dataset_version']}",

        f"optimizer:{params['classification']['optimizer']}",
        f"scheduler:{scheduler_tag}",

        f"weight_decay:{params['classification']['weight_decay']}",
        f"label_smoothing:{params['classification']['label_smoothing']}",
    ]
)



# output dir
output_dir = (
    "/workspace/outputs/classification"
)

os.makedirs(
    output_dir,
    exist_ok=True
)


# example dirs
correct_dir = os.path.join(
    output_dir,
    "correct_examples"
)

incorrect_dir = os.path.join(
    output_dir,
    "incorrect_examples"
)

correct_gradcam_dir = os.path.join(
    correct_dir,
    "gradcam"
)

incorrect_gradcam_dir = os.path.join(
    incorrect_dir,
    "gradcam"
)

os.makedirs(
    correct_dir,
    exist_ok=True
)

os.makedirs(
    incorrect_dir,
    exist_ok=True
)

os.makedirs(
    correct_gradcam_dir,
    exist_ok=True
)

os.makedirs(
    incorrect_gradcam_dir,
    exist_ok=True
)


# save examples
max_correct = 10
max_incorrect = 50
max_correct_gradcam = 5
max_incorrect_gradcam = 5

correct_count = 0
incorrect_count = 0

correct_gradcam_paths = []
incorrect_gradcam_paths = []
correct_gradcam_candidates = defaultdict(list)
incorrect_gradcam_candidates = defaultdict(list)


# prediction
all_preds = []
all_labels = []
all_paths = []

with torch.no_grad():

    for images, labels, image_paths in test_loader:

        images = images.to(device)

        outputs = model(images)

        preds = outputs.argmax(dim=1)

        all_preds.extend(
            preds.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )

        all_paths.extend(
            image_paths
        )




# metrics
metrics_to_use = (
    params["classification"]
    ["metrics"]["final_test"]
)

metrics = {}

if "accuracy" in metrics_to_use:

    metrics["accuracy"] = float(
        accuracy_score(
            all_labels,
            all_preds
        )
    )

if "precision" in metrics_to_use:

    metrics["precision"] = float(
        precision_score(
            all_labels,
            all_preds,
            average="macro",
            zero_division=0
        )
    )

if "recall" in metrics_to_use:

    metrics["recall"] = float(
        recall_score(
            all_labels,
            all_preds,
            average="macro",
            zero_division=0
        )
    )

if "macro_f1" in metrics_to_use:

    metrics["macro_f1"] = float(
        f1_score(
            all_labels,
            all_preds,
            average="macro"
        )
    )

wandb.run.summary[
    "checkpoint_path"
] = checkpoint_path

wandb.run.summary[
    "num_test_images"
] = len(test_dataset)



# metrics.json
with open(
    os.path.join(
        output_dir,
        "metrics.json"
    ),
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metrics,
        f,
        indent=4,
        ensure_ascii=False
    )

wandb.save(
    os.path.join(
        output_dir,
        "metrics.json"
    )
)

for k, v in metrics.items():

    wandb.run.summary[
        f"test_{k}"
    ] = v

wandb.log(
    {
        f"test/{k}": v
        for k, v in metrics.items()
    }
)


# classification report
report = classification_report(
    all_labels,
    all_preds,
    target_names=classes,
    zero_division=0
)

with open(
    os.path.join(
        output_dir,
        "classification_report.txt"
    ),
    "w",
    encoding="utf-8"
) as f:
    f.write(report)

wandb.save(
    os.path.join(
        output_dir,
        "classification_report.txt"
    )
)




# predictions.csv
rows = []

for path, label, pred in zip(
    all_paths,
    all_labels,
    all_preds
):

    rows.append(
        {
            "image_path": path,
            "true_label": idx_to_class[label],
            "pred_label": idx_to_class[pred]
        }
    )

    # correct examples
    if label == pred:
        if correct_count < max_correct:
            save_name = (
                f"true_{idx_to_class[label]}"
                f"_pred_{idx_to_class[pred]}"
                f"_{os.path.basename(path)}"
            )

            shutil.copy(
                path,
                os.path.join(
                    correct_dir,
                    save_name
                )
            )
            correct_count += 1
        
        correct_gradcam_candidates[label].append(path)

    # incorrect examples
    else:
        if incorrect_count < max_incorrect:
            save_name = (
                f"true_{idx_to_class[label]}"
                f"_pred_{idx_to_class[pred]}"
                f"_{os.path.basename(path)}"
            )

            shutil.copy(
                path,
                os.path.join(
                    incorrect_dir,
                    save_name
                )
            )
            incorrect_count += 1
        
        incorrect_gradcam_candidates[label].append(path)


def select_diverse_gradcam_paths(
    candidates_by_class,
    max_examples
):

    selected_paths = []
    class_indices = sorted(candidates_by_class)
    sample_idx = 0

    while len(selected_paths) < max_examples:

        added = False

        for class_idx in class_indices:

            class_candidates = candidates_by_class[class_idx]

            if sample_idx >= len(class_candidates):
                continue

            selected_paths.append(
                class_candidates[sample_idx]
            )
            added = True

            if len(selected_paths) >= max_examples:
                break

        if not added:
            break

        sample_idx += 1

    return selected_paths


correct_gradcam_paths = select_diverse_gradcam_paths(
    correct_gradcam_candidates,
    max_correct_gradcam
)

incorrect_gradcam_paths = select_diverse_gradcam_paths(
    incorrect_gradcam_candidates,
    max_incorrect_gradcam
)


pd.DataFrame(rows).to_csv(
    os.path.join(
        output_dir,
        "predictions.csv"
    ),
    index=False
)

# Generate Grad-CAM for correct examples
print("Generating Grad-CAM for correct examples...")
for idx, img_path in enumerate(correct_gradcam_paths):
    if idx >= max_correct_gradcam:
        break
    
    try:
        save_name = (
            f"gradcam_{idx}_"
            f"{os.path.basename(img_path)}.png"
        )
        
        save_gradcam(
            model,
            img_path,
            os.path.join(
                correct_gradcam_dir,
                save_name
            ),
            device
        )
    except Exception as e:
        print(f"Error generating Grad-CAM for {img_path}")
        print(type(e))
        print(e)
        raise

# Generate Grad-CAM for incorrect examples
print("Generating Grad-CAM for incorrect examples...")
for idx, img_path in enumerate(incorrect_gradcam_paths):
    if idx >= max_incorrect_gradcam:
        break
    
    try:
        save_name = (
            f"gradcam_{idx}_"
            f"{os.path.basename(img_path)}.png"
        )
        
        save_gradcam(
            model,
            img_path,
            os.path.join(
                incorrect_gradcam_dir,
                save_name
            ),
            device
        )
    except Exception as e:
        print(f"Error generating Grad-CAM for {img_path}: {e}")

wandb.run.summary[
    "num_correct_examples_saved"
] = correct_count

wandb.run.summary[
    "num_incorrect_examples_saved"
] = incorrect_count

wandb.run.summary[
    "num_correct_gradcam_generated"
] = len(correct_gradcam_paths)

wandb.run.summary[
    "num_incorrect_gradcam_generated"
] = len(incorrect_gradcam_paths)

wandb.save(
    os.path.join(
        output_dir,
        "predictions.csv"
    )
)



# confusion matrix
if "confusion_matrix" in metrics_to_use:

    cm = confusion_matrix(
        all_labels,
        all_preds
    )

    plt.figure(figsize=(20,20))

    plt.imshow(
        cm,
        cmap="Blues"
    )

    plt.colorbar()

    plt.xticks(
        range(len(classes)),
        classes,
        rotation=90,
        fontsize=6
    )

    plt.yticks(
        range(len(classes)),
        classes,
        fontsize=6
    )

    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.title(
        "Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            "confusion_matrix.png"
        )
    )

    plt.close()

    wandb.save(
        os.path.join(
            output_dir,
            "confusion_matrix.png"
        )
    )

    wandb.log(
        {
            "test/confusion_matrix":
            wandb.Image(
                os.path.join(
                    output_dir,
                    "confusion_matrix.png"
                )
            )
        }
    )


print("Test Evaluation Finished")

wandb.finish()
