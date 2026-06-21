from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Subset
from sklearn.metrics import classification_report

from src.data.datamodule_ffpp import extract_video_key_from_frame_path


def get_target_names(data_module):

    if hasattr(data_module, "output_class_names"):
        return data_module.output_class_names

    if hasattr(data_module, "class_names"):
        return data_module.class_names

    dataset = data_module.test_dataset

    if isinstance(dataset, Subset):
        dataset = dataset.dataset

    if hasattr(dataset, "classes"):
        return dataset.classes

    raise ValueError("Could not infer target names from the data module.")


def collect_predictions(model, data_module, task, device=None):

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    dataloader = data_module.test_dataloader()

    model.eval()
    model.to(device)

    y_true = []
    y_pred = []
    y_prob = []

    with torch.no_grad():

        for images, labels in tqdm(dataloader, desc="Testing frames"):

            images = images.to(device)

            logits = model(images)

            if task == "binary":
                fake_probs = torch.sigmoid(logits)
                real_probs = 1.0 - fake_probs

                probs = torch.stack(
                    [real_probs, fake_probs],
                    dim=1
                )

                preds = (fake_probs >= 0.5).long()

            elif task == "multiclass":
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)

            else:
                raise ValueError(f"Unknown task: {task}")

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    return np.array(y_true), np.array(y_pred), np.array(y_prob)


def print_classification_report(title, y_true, y_pred, target_names):

    labels = []

    for class_index in range(len(target_names)):
        labels.append(class_index)

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        digits=4,
        zero_division=0
    )

    print(report)

    return report


def get_sample_path_and_raw_label(dataset, index):

    # Handles BinaryLabelWrapper
    if hasattr(dataset, "get_mapped_label") and hasattr(dataset, "dataset"):
        return get_sample_path_and_raw_label(dataset.dataset, index)

    # Handles Subset
    if isinstance(dataset, Subset):
        base_index = dataset.indices[index]
        return get_sample_path_and_raw_label(dataset.dataset, base_index)

    # Handles ImageFolder / FixedClassImageFolder
    frame_path, raw_label = dataset.samples[index]

    return frame_path, raw_label


def build_ffpp_frame_predictions(data_module, y_true, y_pred, y_prob, target_names):

    dataset = data_module.test_dataset

    if not hasattr(data_module, "class_names"):
        raise ValueError("Video-level evaluation requires raw FF++ class names.")

    raw_class_names = data_module.class_names

    rows = []

    for dataset_index in tqdm(range(len(dataset)), desc="Building frame dataframe"):

        frame_path, raw_label = get_sample_path_and_raw_label(dataset, dataset_index)

        raw_class_name = raw_class_names[raw_label]
        video_key = extract_video_key_from_frame_path(frame_path, raw_class_name)

        true_label = int(y_true[dataset_index])
        pred_label = int(y_pred[dataset_index])

        row = {
            "dataset_index": dataset_index,
            "frame_path": str(frame_path),
            "raw_class": raw_class_name,
            "video_key": video_key,
            "y_true": true_label,
            "y_pred": pred_label,
            "true_class": target_names[true_label],
            "pred_class": target_names[pred_label]
        }

        for class_index in range(len(target_names)):
            class_name = target_names[class_index]
            row["prob_" + class_name] = y_prob[dataset_index, class_index]

        rows.append(row)

    frame_df = pd.DataFrame(rows)

    if len(frame_df) != len(y_true):
        raise ValueError(
            f"Metadata length ({len(frame_df)}) and prediction length ({len(y_true)}) do not match."
        )

    return frame_df


def aggregate_frame_predictions_to_video(frame_df, target_names):

    prob_columns = []

    for class_name in target_names:
        prob_columns.append("prob_" + class_name)

    video_rows = []

    grouped = frame_df.groupby(["raw_class", "video_key"], sort=False)

    for (raw_class, video_key), group in tqdm(grouped, total=grouped.ngroups, desc="Aggregating videos"):

        mean_probs = group[prob_columns].mean().values

        y_true_video = int(group["y_true"].iloc[0])
        y_pred_video = int(np.argmax(mean_probs))

        row = {
            "raw_class": raw_class,
            "video_key": video_key,
            "n_frames": len(group),
            "y_true": y_true_video,
            "y_pred": y_pred_video,
            "true_class": target_names[y_true_video],
            "pred_class": target_names[y_pred_video]
        }

        for class_index in range(len(target_names)):
            class_name = target_names[class_index]
            row["prob_" + class_name] = mean_probs[class_index]

        video_rows.append(row)

    video_df = pd.DataFrame(video_rows)

    return video_df


def print_test_reports(model, data_module, task, compute_video_report=False):

    target_names = get_target_names(data_module)

    y_true, y_pred, y_prob = collect_predictions(
        model=model,
        data_module=data_module,
        task=task
    )

    frame_report = print_classification_report(
        title="Frame-level classification report",
        y_true=y_true,
        y_pred=y_pred,
        target_names=target_names
    )

    frame_df = None
    video_df = None
    video_report = None

    if compute_video_report:

        frame_df = build_ffpp_frame_predictions(
            data_module=data_module,
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            target_names=target_names
        )

        video_df = aggregate_frame_predictions_to_video(
            frame_df=frame_df,
            target_names=target_names
        )

        video_report = print_classification_report(
            title="Video-level classification report",
            y_true=video_df["y_true"].values,
            y_pred=video_df["y_pred"].values,
            target_names=target_names
        )

    return {
        "frame_report": frame_report,
        "video_report": video_report,
        "frame_df": frame_df,
        "video_df": video_df
    }
