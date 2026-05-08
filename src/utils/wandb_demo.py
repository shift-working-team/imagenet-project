import wandb
import torch

# 1. 설정값 정의 (yaml 파일에서 읽어오는 것을 추천)
my_config = {
    "model_name": "basic-cnn",
    "learning_rate": 0.001,
    "batch_size": 16,
    "image_size": 224,
    "seed": 42,
    "dataset_version": "dvc-v1"
}

# 2. W&B 초기화
wandb.init(
    project="imagenet-project",
    entity="super-shift-working", # 팀 계정이 있다면 작성
    config=my_config,
    name="baseline-experiment"
)

# 3. 학습 루프
for epoch in range(10):
    # (학습 과정 생략)
    train_loss = 0.5 / (epoch + 1)
    val_acc = 0.7 + (epoch * 0.02)
    
    # 4. 지표 기록
    wandb.log({
        "train/loss": train_loss,
        "val/accuracy": val_acc,
        "learning_rate": my_config["learning_rate"]
    })
    
    # 5. 이미지 캡셔닝 결과물 예시 (에폭마다 기록)
    test_img = torch.randn(3, 224, 224) # 가상의 이미지
    wandb.log({
        "predictions": wandb.Image(test_img, caption=f"Epoch {epoch} prediction results")
    })

# 6. 마무리
wandb.finish()