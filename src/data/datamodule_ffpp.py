from pathlib import Path
import random
from collections import Counter

from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms as T
from torchvision.datasets import ImageFolder

import lightning.pytorch as pl


class FixedClassImageFolder(ImageFolder):

    def __init__(self, root, class_names, transform=None):
        self.class_names = class_names
        super().__init__(root=root, transform=transform)

    def find_classes(self, directory):
        classes = []
        class_to_idx = {}

        for class_index, class_name in enumerate(self.class_names):
            class_path = Path(directory) / class_name

            if not class_path.exists():
                raise FileNotFoundError(f"Missing class folder: {class_path}")

            classes.append(class_name)
            class_to_idx[class_name] = class_index

        return classes, class_to_idx


# Deepfakes_000_003_sample_004_frame_000123.png -> 000_003
def extract_video_key_from_frame_path(frame_path, class_name):
    stem = Path(frame_path).stem
    prefix = class_name + "_"

    if stem.startswith(prefix):
        stem = stem[len(prefix):]

    video_key = stem.split("_sample_")[0]

    return video_key


def get_raw_label(dataset, index):
    if isinstance(dataset, Subset):
        base_index = dataset.indices[index]
        return get_raw_label(dataset.dataset, base_index)

    return dataset.samples[index][1]


class BinaryLabelWrapper(Dataset):

    def __init__(self, dataset, real_class_index):
        self.dataset = dataset
        self.real_class_index = real_class_index

    def __len__(self):
        return len(self.dataset)

    def get_mapped_label(self, index):
        raw_label = get_raw_label(self.dataset, index)

        if raw_label == self.real_class_index:
            return 0

        return 1

    def __getitem__(self, index):
        image, raw_label = self.dataset[index]

        if raw_label == self.real_class_index:
            label = 0
        else:
            label = 1

        return image, label


class FaceForensicsDataModule(pl.LightningDataModule):

    def __init__(self, train_path, val_path, test_path, class_names, task="multiclass", batch_size=64, num_workers=0, train_n_frames_per_video=None, val_n_frames_per_video=None, subset_seed=22, image_size=224):
        
        super().__init__()

        if task not in ["binary", "multiclass"]:
            raise ValueError("task must be either 'binary' or 'multiclass'.")

        self.train_path = Path(train_path)
        self.val_path = Path(val_path)
        self.test_path = Path(test_path)

        self.class_names = class_names
        self.task = task

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_n_frames_per_video = train_n_frames_per_video
        self.val_n_frames_per_video = val_n_frames_per_video

        self.subset_seed = subset_seed
        self.image_size = image_size

        self.real_class_name = "Original"
        self.real_class_index = self.class_names.index(self.real_class_name)

        if self.task == "binary":
            self.output_class_names = ["Real", "Fake"]
        else:
            self.output_class_names = self.class_names

        self.num_classes = len(self.output_class_names)

        self.train_transform = T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.RandomHorizontalFlip(p=0.5),
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

    def _subset_frames_per_video(self, dataset, n_frames_per_video=None):

        if n_frames_per_video is None:
            return dataset

        if n_frames_per_video < 1:
            raise ValueError("n_frames_per_video must be at least 1 or None.")

        rng = random.Random(self.subset_seed)

        videos_by_class = {}

        for class_index in range(len(self.class_names)):
            videos_by_class[class_index] = {}

        for index, sample in enumerate(dataset.samples):
            frame_path, label = sample
            class_name = self.class_names[label]

            video_key = extract_video_key_from_frame_path(frame_path, class_name)

            if video_key not in videos_by_class[label]:
                videos_by_class[label][video_key] = []

            videos_by_class[label][video_key].append(index)

        selected_indices = []

        for class_index in range(len(self.class_names)):
            for video_key in videos_by_class[class_index]:
                frame_indices = videos_by_class[class_index][video_key]

                n_frames = n_frames_per_video

                if n_frames > len(frame_indices):
                    n_frames = len(frame_indices)

                sampled_indices = rng.sample(frame_indices, n_frames)

                for index in sampled_indices:
                    selected_indices.append(index)

        rng.shuffle(selected_indices)

        return Subset(dataset, selected_indices)
    

    def _count_labels(self, dataset):
        counts = Counter()
        if isinstance(dataset, BinaryLabelWrapper):
            for index in range(len(dataset)):
                label = dataset.get_mapped_label(index)
                counts[label] += 1

            return counts

        if isinstance(dataset, Subset):
            base_dataset = dataset.dataset

            for index in dataset.indices:
                _, label = base_dataset.samples[index]
                counts[label] += 1

            return counts

        for _, label in dataset.samples:
            counts[label] += 1

        return counts

    def _print_counts(self, dataset):
        counts = self._count_labels(dataset)

        for class_index in range(len(self.output_class_names)):
            class_name = self.output_class_names[class_index]
            print(f"{class_name}: {counts[class_index]}")


    def setup(self, stage=None):

        train_dataset = FixedClassImageFolder(root=self.train_path, class_names=self.class_names, transform=self.train_transform)
        val_dataset = FixedClassImageFolder(root=self.val_path, class_names=self.class_names, transform=self.eval_transform)
        test_dataset = FixedClassImageFolder(root=self.test_path, class_names=self.class_names, transform=self.eval_transform)

        train_dataset = self._subset_frames_per_video(train_dataset, n_frames_per_video=self.train_n_frames_per_video)
        val_dataset = self._subset_frames_per_video(val_dataset, n_frames_per_video=self.val_n_frames_per_video)

        if self.task == "binary":
            self.train_dataset = BinaryLabelWrapper(train_dataset, self.real_class_index)
            self.val_dataset = BinaryLabelWrapper(val_dataset, self.real_class_index)
            self.test_dataset = BinaryLabelWrapper(test_dataset, self.real_class_index)
        else:
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset
            self.test_dataset = test_dataset

        print("Task:", self.task)

        if self.task == "binary":
            print("Class mapping:", {"Real": 0, "Fake": 1})
        else:
            dataset_ref = train_dataset.dataset if isinstance(train_dataset, Subset) else train_dataset
            print("Class mapping:", dataset_ref.class_to_idx)

        print(f"Train: {len(self.train_dataset)}")
        print(f"Validation: {len(self.val_dataset)}")
        print(f"Test: {len(self.test_dataset)}")

        print("\nTrain counts:")
        self._print_counts(self.train_dataset)

        print("\nValidation counts:")
        self._print_counts(self.val_dataset)

        print("\nTest counts:")
        self._print_counts(self.test_dataset)


    def train_dataloader(self):

        persistent_workers = False

        if self.num_workers > 0:
            persistent_workers = True

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=persistent_workers
        )

    def val_dataloader(self):

        persistent_workers = False

        if self.num_workers > 0:
            persistent_workers = True

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=persistent_workers
        )

    def test_dataloader(self):

        persistent_workers = False

        if self.num_workers > 0:
            persistent_workers = True

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=persistent_workers
        )