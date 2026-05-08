from dataset.build_voca import build_voca, tokenizer
from dataset.dataset import CaptionDataset
from transforms.image_transform import get_train_transform

import sys
sys.path.append("/workspace/src")

from torch.utils.data import DataLoader


json_path = '/workspace/data/annotations/annotation.json'
image_dir = '/workspace/data/raw/'

w2i, i2w, voca_size = build_voca(json_path, min_freq=1, max_size=10000)
transform = get_train_transform()
train_dataset = CaptionDataset(json_path, image_dir, w2i, split='train', tokenizer=tokenizer, transform=transform, max_len=30)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=1)

for image, caption in train_loader:
    print(image)
    print(caption)
    break