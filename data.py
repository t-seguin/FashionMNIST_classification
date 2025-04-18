import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data.dataset
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random


import os.path

NUM_THREADS = 4  # Loading the dataset is using 4 CPU threads
BATCH_SIZE = 128  # Using minibatches of 128 samples

# ---------------------------------------------------------------- Datasets


def _load_dataset(
    dataset_dir: str = None, valid_ratio=0.2
) -> tuple[torch.utils.data.dataset.Dataset]:

    if not dataset_dir:
        dataset_dir = os.path.join(os.path.expanduser("~"), "Datasets", "FashionMNIST")
    # Going to use 80%/20% split for train/valid

    # Load the dataset for the training/validation sets
    train_valid_dataset = torchvision.datasets.FashionMNIST(
        root=dataset_dir,
        train=True,
        transform=None,  # transforms.ToTensor(),
        download=True,
    )

    # Split it into training and validation sets
    nb_train = int((1.0 - valid_ratio) * len(train_valid_dataset))
    nb_valid = int(valid_ratio * len(train_valid_dataset))
    train_dataset, valid_dataset = torch.utils.data.dataset.random_split(
        train_valid_dataset, [nb_train, nb_valid]
    )

    # Load the test set
    test_dataset = torchvision.datasets.FashionMNIST(
        root=dataset_dir, transform=None, train=False  # transforms.ToTensor(),
    )

    return train_dataset, valid_dataset, test_dataset


class DatasetTransformer(torch.utils.data.Dataset):

    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.base_dataset[index]
        return self.transform(img), target

    def __len__(self):
        return len(self.base_dataset)


def preview_data(data_loader: torch.utils.data.DataLoader, nsamples: int = 10):
    classes_names = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]
    imgs, labels = next(iter(data_loader))

    plt.figure(figsize=(20, 10), facecolor="w")
    max_horizontal_subplot = 10
    horizontal_subplots = min(10, nsamples)
    vertical_subplots = nsamples // max_horizontal_subplot + min(
        1, nsamples % max_horizontal_subplot
    )
    for i in range(nsamples):
        ax = plt.subplot(vertical_subplots, horizontal_subplots, i + 1)
        plt.imshow(imgs[i, 0, :, :], vmin=0, vmax=1.0, cmap=cm.gray)
        ax.set_title("{}".format(classes_names[labels[i]]), fontsize=15)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.savefig("fashionMNIST_samples.png", bbox_inches="tight")
    plt.show()


def compute_mean_std(loader):
    # Compute the mean over minibatches
    mean_img = None
    for imgs, _ in loader:
        if mean_img is None:
            mean_img = torch.zeros_like(imgs[0])
        mean_img += imgs.sum(dim=0)
    mean_img /= len(loader.dataset)

    # Compute the std over minibatches
    std_img = torch.zeros_like(mean_img)
    for imgs, _ in loader:
        std_img += ((imgs - mean_img) ** 2).sum(dim=0)
    std_img /= len(loader.dataset)
    std_img = torch.sqrt(std_img)

    # Set the variance of pixels with no variance to 1
    # Because there is no variance
    # these pixels will anyway have no impact on the final decision
    std_img[std_img == 0] = 1

    return mean_img, std_img


def get_normalize_transform(
    train_dataset: torch.utils.data.Dataset, batch_size: int = BATCH_SIZE
):
    normalizing_dataset = DatasetTransformer(train_dataset, transforms.ToTensor())
    normalizing_loader = torch.utils.data.DataLoader(
        dataset=normalizing_dataset, batch_size=batch_size, num_workers=NUM_THREADS
    )

    # Compute mean and variance from the training set
    mean_train_tensor, std_train_tensor = compute_mean_std(normalizing_loader)

    data_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x - mean_train_tensor) / std_train_tensor),
        ]
    )

    return data_transforms


def get_aumented_dataset(
    train_dataset, augment_rate: float = 0.3
) -> torch.utils.data.Dataset:

    num_augmented = int(augment_rate * len(train_dataset))
    augmented_indices = random.sample(range(len(train_dataset)), num_augmented)
    subset_to_augment = torch.utils.data.Subset(train_dataset, augmented_indices)
    augmentation_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
            transforms.ToTensor(),
        ]
    )
    augmented_train_dataset = DatasetTransformer(
        subset_to_augment, augmentation_transform
    )

    return augmented_train_dataset


def get_data_loader(batch_size: int = BATCH_SIZE) -> tuple[torch.utils.data.DataLoader]:

    train_dataset, valid_dataset, test_dataset = _load_dataset()

    data_transforms = get_normalize_transform(train_dataset, batch_size)
    augmented_dataset = get_aumented_dataset(train_dataset)
    train_dataset = DatasetTransformer(train_dataset, data_transforms)
    valid_dataset = DatasetTransformer(valid_dataset, data_transforms)
    test_dataset = DatasetTransformer(test_dataset, data_transforms)

    train_loader = torch.utils.data.DataLoader(
        dataset=torch.utils.data.dataset.ConcatDataset(
            [
                train_dataset,
                augmented_dataset,
            ]
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,  # <-- this reshuffles the data at every epoch
        num_workers=NUM_THREADS,
    )

    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_THREADS,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_THREADS,
    )

    return train_loader, valid_loader, test_loader


if __name__ == "__main__":

    train_loader, valid_loader, test_loader = get_data_loader()

    print(
        f"The train set contains {len(train_loader.dataset)} images, in {len(train_loader)} batches"
    )
    print(
        f"The validation set contains {len(valid_loader.dataset)} images"
        f", in {len(valid_loader)} batches"
    )
    print(
        f"The test set contains {len(test_loader.dataset)} images, in {len(test_loader)} batches"
    )

    preview_data(train_loader, 30)
