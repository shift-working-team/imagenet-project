import os
from PIL import Image
from torch.utils.data import Dataset


class ClassificationDataset(Dataset):

    def __init__(
        self,
        root_dir,
        class_to_idx,
        split="train",
        transform=None,
        split_ratio=(0.7, 0.15, 0.15)
    ):

        self.transform = transform
        self.samples = []

        for class_name in sorted(os.listdir(root_dir)):
            class_path = os.path.join(
                root_dir,
                class_name
            )

            if not os.path.isdir(class_path):
                continue

            images = sorted(os.listdir(class_path))

            total = len(images)

            train_end = int(total * split_ratio[0])
            val_end = train_end + int(total * split_ratio[1])

            if split == "train":
                split_images = images[:train_end]

            elif split == "val":
                split_images = images[train_end:val_end]

            else:
                split_images = images[val_end:]

            label = class_to_idx[class_name]

            for image_name in split_images:
                image_path = os.path.join(
                    class_path,
                    image_name
                )

                self.samples.append(
                    (image_path, label)
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = Image.open(
            image_path
        ).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label