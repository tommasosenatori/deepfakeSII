from pathlib import Path
import argparse
import time
import os

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from src.data.factory import build_datamodule
from src.models import ImageClassifier


def get_num_classes(dataset_name, task):

    if task == "binary":
        return 2

    if task == "multiclass":
        if dataset_name != "ffpp":
            raise ValueError("Multiclass is currently supported only for FF++.")

        return 6

    raise ValueError(f"Unknown task: {task}")


def find_max_batch_size(model_name, task, num_classes, image_size, learning_rate, weight_decay):

    if not torch.cuda.is_available():
        raise RuntimeError("Auto batch size requires CUDA.")

    print("\nSearching maximum batch size...\n")

    best_batch_size = 32
    candidate = 32

    while True:

        model = None
        x = None
        logits = None
        loss = None

        try:
            torch.cuda.empty_cache()

            model = ImageClassifier(
                model_name=model_name,
                task=task,
                num_classes=num_classes,
                learning_rate=learning_rate,
                weight_decay=weight_decay
            ).cuda()

            model.train()

            x = torch.randn(
                candidate,
                3,
                image_size,
                image_size,
                device="cuda"
            )

            logits = model(x)

            loss = logits.mean()
            loss.backward()

            best_batch_size = candidate

            print(f"Batch size {candidate}: OK")

            del model
            del x
            del logits
            del loss

            torch.cuda.empty_cache()

            candidate *= 2

        except RuntimeError as e:

            error_message = str(e).lower()

            is_memory_error = False

            if "out of memory" in error_message:
                is_memory_error = True

            if "cuda" in error_message and "memory" in error_message:
                is_memory_error = True

            if "cudnn_status_internal_error" in error_message:
                is_memory_error = True

            if "allocation_failed" in error_message:
                is_memory_error = True

            if "host_allocation_failed" in error_message:
                is_memory_error = True

            if is_memory_error:
                print(f"Batch size {candidate}: memory error")
                print("Stopping batch size search.")

                try:
                    del model
                    del x
                    del logits
                    del loss
                except Exception:
                    pass

                torch.cuda.empty_cache()

                break

            raise

    safe_batch_size = int(best_batch_size * 0.70)

    if safe_batch_size < 1:
        safe_batch_size = 1

    print()
    print(f"Maximum working batch size: {best_batch_size}")
    print(f"Selected batch size: {safe_batch_size}")
    print()

    return safe_batch_size


def main(args):

    torch.set_float32_matmul_precision("high")

    num_classes = get_num_classes(
        dataset_name=args.dataset_name,
        task=args.task
    )
    
    if args.auto_batch_size:
        args.batch_size = find_max_batch_size(
            model_name=args.model_name,
            task=args.task,
            num_classes=num_classes,
            image_size=args.image_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay
        )

    print("\n========== SYSTEM INFO ==========")
    print("Dataset:", args.dataset_name)
    print("Task:", args.task)
    print("Number of classes:", num_classes)
    print("Model:", args.model_name)
    print("Batch size:", args.batch_size)
    print("Workers:", args.num_workers)
    print("Train frames per video:", args.train_n_frames_per_video)
    print("Val frames per video:", args.val_n_frames_per_video)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    try:
        print("CPU affinity:", len(os.sched_getaffinity(0)))
    except Exception:
        pass

    print("=================================\n")

    experiment_name = (
        f"{args.dataset_name}"
        f"__{args.task}"        
        f"__{args.model_name}"
        f"__subset_{args.subset_fraction}"
        f"__frames_train_{args.train_n_frames_per_video}"
        f"__frames_val_{args.val_n_frames_per_video}"
    )

    data_module = build_datamodule(
        dataset_name=args.dataset_name,
        task=args.task,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        subset_fraction=args.subset_fraction,
        subset_seed=args.seed,
        image_size=args.image_size,
        train_n_frames_per_video=args.train_n_frames_per_video,
        val_n_frames_per_video=args.val_n_frames_per_video
    )

    model = ImageClassifier(
        model_name=args.model_name,
        task=args.task,
        num_classes=num_classes,
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

    precision = "32-true"

    if torch.cuda.is_available():
        precision = "16-mixed"

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices="auto",
        precision=precision,
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

    best_model = ImageClassifier.load_from_checkpoint(
        checkpoint_callback.best_model_path
    )

    test_results = trainer.test(best_model, datamodule=data_module)

    print("Test results:")
    print(test_results)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", type=str, default="ffpp", choices=["140k", "ffpp"])
    parser.add_argument("--task", type=str, default="multiclass", choices=["binary", "multiclass"])

    parser.add_argument("--model_name", type=str, default="resnet18")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--auto_batch_size", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--subset_fraction", type=float, default=None)
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--image_size", type=int, default=224)

    parser.add_argument("--train_n_frames_per_video", type=int, default=None)
    parser.add_argument("--val_n_frames_per_video", type=int, default=None)

    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=7)

    parser.add_argument("--output_dir", type=str, default="outputs")

    args = parser.parse_args()

    main(args)