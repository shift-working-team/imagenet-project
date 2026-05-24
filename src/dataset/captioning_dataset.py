import os
import json
import random
from PIL import Image
from torch.utils.data import Dataset
import torch


class CaptionDataset(Dataset):

    def __init__(
        self,
        json_path,
        image_dir,
        w2i,
        tokenizer: callable,
        split='train',
        transform=None,
        max_len=30,
        train_num_caption=1,
        debug=False
    ):

        with open(json_path, 'r') as f:
            self.data = json.load(f)

        # 디버깅용
        if debug:
            self.data= self.data[:10]

        if split == "val":
            self.is_val = True
        else:
            self.is_val = False
        
        self.image_dir = image_dir
        self.w2i = w2i
        self.transform = transform
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.train_num_caption = train_num_caption

    def __len__(self):
        return len(self.data)
    
    def encode_caption(self, caption):

        words = self.tokenizer(caption)

        tokens = (
            [self.w2i["<sos>"]] +
            [self.w2i.get(w, self.w2i["<unk>"]) for w in words]
            )
        
        # truncation
        if len(tokens) > self.max_len:
            tokens = (tokens[:self.max_len - 1])
            tokens.append(self.w2i["<eos>"])
        else:
            tokens += ([self.w2i["<pad>"]] * (self.max_len - len(tokens)))

        return torch.tensor(tokens, dtype=torch.long)

    def __getitem__(self, index):

        data = self.data[index]

        image_path = os.path.join(self.image_dir, data["file_name"])

        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        captions = (data["captions"])

        # validation
        if self.is_val:
            caption = random.choice(captions)

            tokens = (self.encode_caption(caption))

            return image, tokens, captions

        # train
        selected_captions = (random.sample(captions, k=self.train_num_caption))

        images = []
        token_list = []
        for caption in selected_captions:
            images.append(image)
            token_list.append(self.encode_caption(caption))

        images = torch.stack(images)
        tokens = torch.stack(token_list)

        return images, tokens

