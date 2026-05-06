import json
import random
from PIL import Image
from torch.utils.data import Dataset
import torch

class CaptionDataset(Dataset):
    def __init__(self, json_path, image_dir, w2i, transform=None, max_len=20):
        with open(json_path, 'r') as f:
            self.data = json.load(json_path)

        self.image_dir = image_dir
        self.w2i = w2i
        self.transform = transform
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]

        image_path = self.image_dir + data["image"]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        
        caption = random.choice(data["captions"])

        words = caption.lower().split()

        tokens = self.w2i["<sos>"] + [self.w2i[w] for w in words] + self.w2i["<eos>"]

        if len(tokens) < self.max_len:
            tokens += ["<pad>"] * (self.max_len - len(tokens))
        else:
            tokens = tokens[:self.max_len]

        tokens = torch.tensor(tokens)

        return image, tokens
    