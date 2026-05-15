import sys
sys.path.append("/workspace/src/models")
import torch


# model imports
from lstm import DecoderLSTM
from gru import DecoderGRU
from transformer import DecoderTransformer
from resnet18 import EncoderResnet18
from efficientnet import get_efficientnet_b0
from convnext import get_convnext_tiny
from mobilenet import get_mobilenet_v3_small



# device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"device: {device}")



# caption model dummy input
feature = torch.randn(1, 512).to(device)

caption = torch.tensor(
    [[0, 1, 2, 3, 4]]
).to(device)



### LSTM Forward ###
lstm_model = DecoderLSTM().to(device)

lstm_out = lstm_model(
    feature,
    caption
)

print(f"LSTM: {lstm_out.shape}")



### GRU Forward ###
gru_model = DecoderGRU().to(device)

gru_out = gru_model(
    feature,
    caption
)

print(f"GRU: {gru_out.shape}")



### Transformer Forward ###
transformer_model = DecoderTransformer().to(device)

transformer_out = transformer_model(
    feature,
    caption,
    0
)

print(f"Transformer: {transformer_out.shape}")



### ResNet18 Forward ###
NUM_CLASSES = 50

resnet18_model = EncoderResnet18(
    num_classes=NUM_CLASSES
).to(device)

dummy_images = torch.randn(
    8, 3, 224, 224
).to(device)

logits, features = resnet18_model(
    dummy_images
)

print(f"ResNet18 logits: {logits.shape}")
print(f"ResNet18 features: {features.shape}")



### EfficientNet-B0 Forward ###
efficientnet_model = get_efficientnet_b0(
    num_classes=NUM_CLASSES
).to(device)

efficientnet_out = efficientnet_model(
    dummy_images
)

print(
    f"EfficientNet-B0: "
    f"{efficientnet_out.shape}"
)

# expected:
# torch.Size([8, 50])



### ConvNeXt-Tiny Forward ###
convnext_model = get_convnext_tiny(
    num_classes=NUM_CLASSES
).to(device)

convnext_out = convnext_model(
    dummy_images
)

print(
    f"ConvNeXt-Tiny: "
    f"{convnext_out.shape}"
)

# expected:
# torch.Size([8, 50])



### MobileNetV3 Small Forward ###
mobilenet_model = get_mobilenet_v3_small(
    num_classes=NUM_CLASSES
).to(device)

mobilenet_out = mobilenet_model(
    dummy_images
)

print(
    f"MobileNetV3 Small: "
    f"{mobilenet_out.shape}"
)

# expected:
# torch.Size([8, 50])