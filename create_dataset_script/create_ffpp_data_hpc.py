from pathlib import Path
import os
import cv2
import json
import time
import argparse
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "datasets" / "faceforensics++"
SPLITS_PATH = DATASET_PATH / "splits"

TRAIN_JSON = SPLITS_PATH / "train.json"
VAL_JSON = SPLITS_PATH / "val.json"
TEST_JSON = SPLITS_PATH / "test.json"

IMAGE_SIZE = 224
CROP_SCALE = 1.3
SKIP_EXISTING = True

CLASS_TO_FOLDER = {}

CLASS_TO_FOLDER["Original"] = "original"
CLASS_TO_FOLDER["Deepfakes"] = "Deepfakes"
CLASS_TO_FOLDER["Face2Face"] = "Face2Face"
CLASS_TO_FOLDER["FaceShifter"] = "FaceShifter"
CLASS_TO_FOLDER["FaceSwap"] = "FaceSwap"
CLASS_TO_FOLDER["NeuralTextures"] = "NeuralTextures"

SPLIT_TO_JSON = {}

SPLIT_TO_JSON["train"] = TRAIN_JSON
SPLIT_TO_JSON["val"] = VAL_JSON
SPLIT_TO_JSON["test"] = TEST_JSON

DETECTOR = None


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--output_name", type=str, default="faceforensics++_6classes_parallel")
    parser.add_argument("--train_samples", type=int, default=270)
    parser.add_argument("--val_samples", type=int, default=100)
    parser.add_argument("--test_samples", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max_pairs_per_split", type=int, default=None)
    parser.add_argument("--split", type=str, default="all", choices=["all", "train", "val", "test"])

    args = parser.parse_args()

    return args


def create_output_folders(output_dataset_path):

    for split_name in SPLIT_TO_JSON.keys():
        split_path = output_dataset_path / split_name

        for class_name in CLASS_TO_FOLDER.keys():
            class_path = split_path / class_name
            class_path.mkdir(parents=True, exist_ok=True)


def load_split_pairs(json_path):

    with open(json_path, "r") as file:
        pairs = json.load(file)

    return pairs


def get_uniform_frame_numbers(total_frames, number_of_samples):

    frame_numbers = []

    if total_frames <= 0:
        return frame_numbers

    if number_of_samples <= 1:
        frame_numbers.append(0)
        return frame_numbers

    for i in range(number_of_samples):
        frame_number = round(i * (total_frames - 1) / (number_of_samples - 1))
        frame_number = int(frame_number)
        frame_numbers.append(frame_number)

    return frame_numbers


def get_haar_detector():

    possible_paths = []

    if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
        possible_paths.append(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")

    conda_prefix = os.environ.get("CONDA_PREFIX")

    if conda_prefix is not None:
        possible_paths.append(Path(conda_prefix) / "share" / "opencv4" / "haarcascades" / "haarcascade_frontalface_default.xml")
        possible_paths.append(Path(conda_prefix) / "share" / "opencv" / "haarcascades" / "haarcascade_frontalface_default.xml")
        possible_paths.append(Path(conda_prefix) / "lib" / "python3.10" / "site-packages" / "cv2" / "data" / "haarcascade_frontalface_default.xml")

    for haar_path in possible_paths:
        if haar_path.exists():
            detector = cv2.CascadeClassifier(str(haar_path))

            if not detector.empty():
                return detector

    raise RuntimeError("Haar cascade not found. Run: find $CONDA_PREFIX -name 'haarcascade_frontalface_default.xml'")


def init_worker():

    global DETECTOR

    cv2.setNumThreads(1)
    DETECTOR = get_haar_detector()


def detect_largest_face(frame, detector):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(detections) == 0:
        return None

    best_face = None
    best_area = -1

    for detection in detections:
        x = int(detection[0])
        y = int(detection[1])
        w = int(detection[2])
        h = int(detection[3])
        area = w * h

        if area > best_area:
            best_area = area
            best_face = (x, y, w, h)

    return best_face


def crop_face_with_padding(frame, face, crop_scale):

    x = face[0]
    y = face[1]
    w = face[2]
    h = face[3]

    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    crop_size = int(round(max(w, h) * crop_scale))

    center_x = x + w // 2
    center_y = y + h // 2

    left = center_x - crop_size // 2
    top = center_y - crop_size // 2

    right = left + crop_size
    bottom = top + crop_size

    pad_left = max(0, -left)
    pad_top = max(0, -top)
    pad_right = max(0, right - frame_width)
    pad_bottom = max(0, bottom - frame_height)

    safe_left = max(0, left)
    safe_top = max(0, top)
    safe_right = min(frame_width, right)
    safe_bottom = min(frame_height, bottom)

    crop = frame[safe_top:safe_bottom, safe_left:safe_right]

    if crop.size == 0:
        return None

    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        crop = cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    return crop


def find_video_path(class_name, video_stem):

    input_folder = CLASS_TO_FOLDER[class_name]
    video_path = DATASET_PATH / input_folder / (video_stem + ".mp4")

    if video_path.exists():
        return video_path

    return None


def make_video_task(video_path, output_class_path, split_name, class_name, video_key, source_id, target_id, number_of_samples):

    task = {}

    task["video_path"] = str(video_path)
    task["output_class_path"] = str(output_class_path)
    task["split"] = split_name
    task["class"] = class_name
    task["video_key"] = video_key
    task["source_id"] = source_id
    task["target_id"] = target_id
    task["requested_samples"] = number_of_samples

    return task


def make_missing_row(split_name, class_name, video_stem, source_id, target_id):

    missing_row = {}

    missing_row["split"] = split_name
    missing_row["class"] = class_name
    missing_row["video_stem"] = video_stem
    missing_row["source_id"] = source_id
    missing_row["target_id"] = target_id

    return missing_row


def build_video_tasks(output_dataset_path, split_to_samples, selected_split, max_pairs_per_split):

    tasks = []
    missing_videos = []

    for split_name in SPLIT_TO_JSON.keys():

        if selected_split != "all" and split_name != selected_split:
            continue

        json_path = SPLIT_TO_JSON[split_name]
        pairs = load_split_pairs(json_path)
        number_of_samples = split_to_samples[split_name]
        processed_originals = set()

        if max_pairs_per_split is not None:
            pairs = pairs[:max_pairs_per_split]

        print("Building tasks for split:", split_name, flush=True)
        print("Pairs:", len(pairs), flush=True)
        print("Samples per video:", number_of_samples, flush=True)

        for pair in pairs:
            id_a = pair[0]
            id_b = pair[1]

            directions = []
            directions.append((id_a, id_b))
            directions.append((id_b, id_a))

            for source_id, target_id in directions:
                pair_name = source_id + "_" + target_id

                if target_id not in processed_originals:
                    class_name = "Original"
                    video_stem = target_id
                    video_key = target_id
                    video_path = find_video_path(class_name, video_stem)
                    output_class_path = output_dataset_path / split_name / class_name

                    if video_path is None:
                        missing_videos.append(make_missing_row(split_name, class_name, video_stem, source_id, target_id))
                    else:
                        task = make_video_task(video_path, output_class_path, split_name, class_name, video_key, source_id, target_id, number_of_samples)
                        tasks.append(task)

                    processed_originals.add(target_id)

                for class_name in CLASS_TO_FOLDER.keys():
                    if class_name == "Original":
                        continue

                    video_stem = pair_name
                    video_key = pair_name
                    video_path = find_video_path(class_name, video_stem)
                    output_class_path = output_dataset_path / split_name / class_name

                    if video_path is None:
                        missing_videos.append(make_missing_row(split_name, class_name, video_stem, source_id, target_id))
                        continue

                    task = make_video_task(video_path, output_class_path, split_name, class_name, video_key, source_id, target_id, number_of_samples)
                    tasks.append(task)

    return tasks, missing_videos


def extract_faces_from_video_task(task):

    global DETECTOR

    if DETECTOR is None:
        DETECTOR = get_haar_detector()

    video_path = Path(task["video_path"])
    output_class_path = Path(task["output_class_path"])
    split_name = task["split"]
    class_name = task["class"]
    video_key = task["video_key"]
    source_id = task["source_id"]
    target_id = task["target_id"]
    number_of_samples = int(task["requested_samples"])

    summary = {}

    summary["split"] = split_name
    summary["class"] = class_name
    summary["video_path"] = str(video_path)
    summary["video_key"] = video_key
    summary["source_id"] = source_id
    summary["target_id"] = target_id
    summary["requested_samples"] = number_of_samples
    summary["total_frames"] = 0
    summary["saved_images"] = 0
    summary["written_images"] = 0
    summary["skipped_existing"] = 0
    summary["missed_faces"] = 0
    summary["read_errors"] = 0
    summary["write_errors"] = 0
    summary["status"] = "ok"
    summary["error"] = ""

    try:
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_numbers = get_uniform_frame_numbers(total_frames, number_of_samples)
        summary["total_frames"] = total_frames

        for sample_index in range(len(frame_numbers)):
            frame_number = frame_numbers[sample_index]
            output_name = class_name + "_" + video_key + "_sample_" + str(sample_index).zfill(3) + "_frame_" + str(frame_number).zfill(6) + ".jpg"
            output_path = output_class_path / output_name

            if SKIP_EXISTING and output_path.exists():
                summary["saved_images"] = summary["saved_images"] + 1
                summary["skipped_existing"] = summary["skipped_existing"] + 1
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()

            if not ret:
                summary["read_errors"] = summary["read_errors"] + 1
                continue

            face = detect_largest_face(frame, DETECTOR)

            if face is None:
                summary["missed_faces"] = summary["missed_faces"] + 1
                continue

            crop = crop_face_with_padding(frame, face, CROP_SCALE)

            if crop is None:
                summary["missed_faces"] = summary["missed_faces"] + 1
                continue

            crop = cv2.resize(crop, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
            ok = cv2.imwrite(str(output_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

            if ok:
                summary["saved_images"] = summary["saved_images"] + 1
                summary["written_images"] = summary["written_images"] + 1
            else:
                summary["write_errors"] = summary["write_errors"] + 1

        cap.release()

    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = repr(exc)

    return summary


def run_parallel_extraction(tasks, workers):

    summaries = []
    start_time = time.time()

    print("Total video tasks:", len(tasks), flush=True)
    print("Workers:", workers, flush=True)

    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker) as executor:
        futures = []

        for task in tasks:
            future = executor.submit(extract_faces_from_video_task, task)
            futures.append(future)

        for future in tqdm(as_completed(futures), total=len(futures), desc="Videos completed"):
            summary = future.result()
            summaries.append(summary)

    end_time = time.time()
    elapsed_seconds = end_time - start_time

    return summaries, elapsed_seconds


def save_outputs(output_dataset_path, summaries, missing_videos, elapsed_seconds):

    summaries_df = pd.DataFrame(summaries)
    missing_df = pd.DataFrame(missing_videos)

    summaries_path = output_dataset_path / "preprocessing_summary.csv"
    missing_path = output_dataset_path / "missing_videos.csv"

    summaries_df.to_csv(summaries_path, index=False)
    missing_df.to_csv(missing_path, index=False)

    total_saved = int(summaries_df["saved_images"].sum()) if len(summaries_df) > 0 else 0
    total_written = int(summaries_df["written_images"].sum()) if len(summaries_df) > 0 else 0
    total_skipped = int(summaries_df["skipped_existing"].sum()) if len(summaries_df) > 0 else 0
    total_missed = int(summaries_df["missed_faces"].sum()) if len(summaries_df) > 0 else 0
    total_read_errors = int(summaries_df["read_errors"].sum()) if len(summaries_df) > 0 else 0
    total_write_errors = int(summaries_df["write_errors"].sum()) if len(summaries_df) > 0 else 0

    elapsed_minutes = elapsed_seconds / 60.0
    written_per_minute = total_written / elapsed_minutes if elapsed_minutes > 0 else 0
    saved_per_minute = total_saved / elapsed_minutes if elapsed_minutes > 0 else 0

    benchmark = {}

    benchmark["elapsed_seconds"] = elapsed_seconds
    benchmark["elapsed_minutes"] = elapsed_minutes
    benchmark["total_saved_images"] = total_saved
    benchmark["total_written_images"] = total_written
    benchmark["total_skipped_existing"] = total_skipped
    benchmark["total_missed_faces"] = total_missed
    benchmark["total_read_errors"] = total_read_errors
    benchmark["total_write_errors"] = total_write_errors
    benchmark["written_images_per_minute"] = written_per_minute
    benchmark["saved_images_per_minute"] = saved_per_minute

    benchmark_df = pd.DataFrame([benchmark])
    benchmark_path = output_dataset_path / "benchmark.csv"
    benchmark_df.to_csv(benchmark_path, index=False)

    print("Done.", flush=True)
    print("Summary saved to:", summaries_path, flush=True)
    print("Missing videos saved to:", missing_path, flush=True)
    print("Benchmark saved to:", benchmark_path, flush=True)
    print("Elapsed minutes:", round(elapsed_minutes, 2), flush=True)
    print("Written images:", total_written, flush=True)
    print("Saved images including skipped existing:", total_saved, flush=True)
    print("Written images per minute:", round(written_per_minute, 2), flush=True)
    print("Saved images per minute:", round(saved_per_minute, 2), flush=True)


def count_created_images(output_dataset_path):

    for split_name in SPLIT_TO_JSON.keys():
        for class_name in CLASS_TO_FOLDER.keys():
            class_path = output_dataset_path / split_name / class_name
            image_count = len(list(class_path.glob("*.jpg")))
            print(split_name, class_name, image_count, flush=True)


def print_split_info():

    train_pairs = load_split_pairs(TRAIN_JSON)
    val_pairs = load_split_pairs(VAL_JSON)
    test_pairs = load_split_pairs(TEST_JSON)

    print("Train pairs:", len(train_pairs), flush=True)
    print("Val pairs:", len(val_pairs), flush=True)
    print("Test pairs:", len(test_pairs), flush=True)


def main():

    args = parse_args()

    output_dataset_path = PROJECT_ROOT / "datasets" / args.output_name

    split_to_samples = {}

    split_to_samples["train"] = args.train_samples
    split_to_samples["val"] = args.val_samples
    split_to_samples["test"] = args.test_samples

    print("Output dataset path:", output_dataset_path, flush=True)
    print("Train samples:", args.train_samples, flush=True)
    print("Val samples:", args.val_samples, flush=True)
    print("Test samples:", args.test_samples, flush=True)
    print("Selected split:", args.split, flush=True)
    print("Max pairs per split:", args.max_pairs_per_split, flush=True)

    create_output_folders(output_dataset_path)
    print_split_info()

    tasks, missing_videos = build_video_tasks(output_dataset_path, split_to_samples, args.split, args.max_pairs_per_split)
    summaries, elapsed_seconds = run_parallel_extraction(tasks, args.workers)

    save_outputs(output_dataset_path, summaries, missing_videos, elapsed_seconds)

    print("Created images:", flush=True)
    count_created_images(output_dataset_path)


if __name__ == "__main__":
    main()
