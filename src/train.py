from pathlib import Path
import argparse
import time

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from datamodule import RealFakeDataModule
from models import BinaryImageClassifier


def main(args):

    experiment_name = f"{args.dataset_name}__{args.model_name}__{args.subset_fraction}"

    data_module = RealFakeDataModule(
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        subset_fraction=args.subset_fraction,
        subset_seed=args.seed,
        image_size=args.image_size
    )

    model = BinaryImageClassifier(
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(args.output_dir) / "checkpoints" / experiment_name,
        filename="best",
        monitor="val_loss",
        mode="min",
        save_top_k=1
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=args.patience
    )

    logger = TensorBoardLogger(
        save_dir=Path(args.output_dir) / "logs",
        name=experiment_name
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices="auto",
        callbacks=[
            checkpoint_callback,
            early_stopping
        ],
        logger=logger,
        enable_progress_bar=True
    )

    

    start_time = time.time()
    trainer.fit(model, datamodule=data_module)
    training_time = time.time() - start_time

    epochs_completed = trainer.current_epoch
    print(f"\nTraining time: {training_time:.2f} s")
    print(f"Epochs completed: {epochs_completed}")
    print(f"Average epoch time: {training_time / epochs_completed:.2f} s\n")

    print("Best checkpoint:")
    print(checkpoint_callback.best_model_path)

    best_model = BinaryImageClassifier.load_from_checkpoint(
        checkpoint_callback.best_model_path
    )

    test_results = trainer.test(best_model, datamodule=data_module)

    print("Test results:")
    print(test_results)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path", type=str, default="datasets/140k_real_fake_faces")
    parser.add_argument("--dataset_name", type=str, default="140kfaces")

    parser.add_argument("--model_name", type=str, default="resnet18")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--subset_fraction", type=float, default=None)
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--image_size", type=int, default=224)

    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=7)

    parser.add_argument("--output_dir", type=str, default="outputs")

    args = parser.parse_args()

    main(args)