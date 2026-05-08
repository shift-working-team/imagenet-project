from torchvision import datasets

def get_dataset(data_dir, transform):

    dataset = datasets.ImageFolder(
        root=data_dir,
        transform=transform
    )

    return dataset