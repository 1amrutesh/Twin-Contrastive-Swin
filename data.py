import torchvision
from typing import Any, Callable, Optional
from PIL import Image
from torchvision.datasets.folder import default_loader
from transforms import build_transform
from torch.utils import data


class CIFAR10Dataset(torchvision.datasets.CIFAR10):
    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.train:
            return img, target, index
        else:
            return img, target, index + 50000


class CIFAR100Dataset(CIFAR10Dataset):
    base_folder = "cifar-100-python"
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
    train_list = [["train", "16019d7e3df5f24257cddd939b257f8d"]]
    test_list = [["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"]]
    meta = {
        "filename": "meta",
        "key": "fine_label_names",
        "md5": "7973b15100ade9c7d40fb424638fde48",
    }


IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)


class CustomImageFolder(torchvision.datasets.DatasetFolder):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super().__init__(
            root,
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )
        self.imgs = self.samples

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, index


def build_dataset(type, args):
    is_train = type == "train"
    transform = build_transform(is_train, args)
    root = args.data_path

    if args.dataset == "CIFAR-10":
        dataset = data.ConcatDataset(
            [
                CIFAR10Dataset(
                    root=root + "CIFAR-10",
                    train=True,
                    download=True,
                    transform=transform,
                ),
                CIFAR10Dataset(
                    root=root + "CIFAR-10",
                    train=False,
                    download=True,
                    transform=transform,
                ),
            ]
        )
    elif args.dataset == "CIFAR-100":
        dataset = data.ConcatDataset(
            [
                CIFAR100Dataset(
                    root=root + "CIFAR-100",
                    train=True,
                    download=True,
                    transform=transform,
                ),
            ]
        )
    elif args.dataset == "MIAD":
        if is_train:
            path = root 
        else:
            path = root 
        dataset = CustomImageFolder(root=path, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    return dataset
