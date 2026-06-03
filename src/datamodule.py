from pathlib import Path
import random

import torch
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T
from torchvision.datasets import ImageFolder

import lightning.pytorch as pl

class RealFakeDataModule(pl.LightningDataModule):

    def __init__(self, dataset_path, batch_size=64, num_workers=0, subset_fraction=None, subset_seed=22, image_size=224):
        super().__init__()

        self.dataset_path = Path(dataset_path)

        self.train_path = self.dataset_path / "train"
        self.valid_path = self.dataset_path / "valid"
        self.test_path = self.dataset_path / "test"

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.subset_fraction = subset_fraction
        self.subset_seed = subset_seed
        self.image_size = image_size

        self.train_transform = T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.RandomHorizontalFlip(p=0.5), # augmentation
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.eval_transform = T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


    def _balanced_subset(self, dataset, apply_subset=True):

        if self.subset_fraction is None or not apply_subset:
            return dataset

        rng = random.Random(self.subset_seed)

        indices_by_class = {}

        for class_idx in range(len(dataset.classes)):
            indices_by_class[class_idx] = []

        for idx, (_, label) in enumerate(dataset.samples):
            indices_by_class[label].append(idx)

        selected = []

        for label in indices_by_class:
            n_samples = int(len(indices_by_class[label]) * self.subset_fraction)
            sampled_indices = rng.sample(indices_by_class[label], n_samples)
            selected.extend(sampled_indices)

        rng.shuffle(selected)

        return Subset(dataset, selected)

    def setup(self, stage=None):

        self.train_dataset = ImageFolder(self.train_path, transform=self.train_transform)
        self.val_dataset = ImageFolder(self.valid_path, transform=self.eval_transform)
        self.test_dataset = ImageFolder(self.test_path, transform=self.eval_transform)

        self.train_dataset = self._balanced_subset(self.train_dataset, apply_subset=True)
        self.val_dataset = self._balanced_subset(self.val_dataset, apply_subset=True)
        self.test_dataset = self._balanced_subset(self.test_dataset, apply_subset=False)

        dataset_ref = self.train_dataset

        if isinstance(self.train_dataset, Subset):
            dataset_ref = self.train_dataset.dataset

        print("Class mapping:", dataset_ref.class_to_idx)
        print("Train:", len(self.train_dataset))
        print("Validation:", len(self.val_dataset))
        print("Test:", len(self.test_dataset))

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )