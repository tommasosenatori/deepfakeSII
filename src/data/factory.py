from src.data.datamodule_140k import RealFakeDataModule
from src.data.datamodule_ffpp import FaceForensicsDataModule


FFPP_CLASSES = ["Original", "Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]


def build_datamodule(dataset_name, task, batch_size, num_workers, subset_fraction=None, subset_seed=22, image_size=224, train_n_frames_per_video=None, val_n_frames_per_video=None):
    if dataset_name == "140k":

        if task != "binary":
            raise ValueError("140k supports only binary classification.")
        return RealFakeDataModule(dataset_path="datasets/140k_real_fake_faces", batch_size=batch_size, num_workers=num_workers, subset_fraction=subset_fraction, subset_seed=subset_seed, image_size=image_size)

    if dataset_name == "ffpp":
        return FaceForensicsDataModule(train_path="datasets/faceforensics++_6classes/train",val_path="datasets/faceforensics++_6classes/val", test_path="datasets/faceforensics++_6classes/test", class_names=FFPP_CLASSES, task=task, batch_size=batch_size, num_workers=num_workers, train_n_frames_per_video=train_n_frames_per_video, val_n_frames_per_video=val_n_frames_per_video, subset_seed=subset_seed, image_size=image_size)

    if dataset_name == "ffpp_small":
        return FaceForensicsDataModule( train_path="datasets/fff++_6classes_small/train", val_path="datasets/fff++_6classes_small/val", test_path="datasets/fff++_6classes_small/test", class_names=FFPP_CLASSES, task=task, batch_size=batch_size, num_workers=num_workers, train_n_frames_per_video=train_n_frames_per_video, val_n_frames_per_video=val_n_frames_per_video, subset_seed=subset_seed, image_size=image_size)

    raise ValueError(f"Unknown dataset: {dataset_name}")