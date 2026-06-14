from pathlib import Path
import cv2
import json
import csv
import pandas as pd
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "datasets" / "faceforensics++"
SPLITS_PATH = DATASET_PATH / "splits"
OUTPUT_DATASET_PATH = PROJECT_ROOT / "datasets" / "faceforensics++_6classes"

TRAIN_JSON = SPLITS_PATH / "train.json"
VAL_JSON = SPLITS_PATH / "val.json"
TEST_JSON = SPLITS_PATH / "test.json"

IMAGE_SIZE = 224 # for CNN models
CROP_SCALE = 1.3 # padding after cropping the faces
SKIP_EXISTING = True
MAX_PAIRS_PER_SPLIT = None


# map class names to folder names in the dataset
CLASS_TO_FOLDER = {}

CLASS_TO_FOLDER["Original"] = "original"
CLASS_TO_FOLDER["Deepfakes"] = "Deepfakes"
CLASS_TO_FOLDER["Face2Face"] = "Face2Face"
CLASS_TO_FOLDER["FaceShifter"] = "FaceShifter"
CLASS_TO_FOLDER["FaceSwap"] = "FaceSwap"
CLASS_TO_FOLDER["NeuralTextures"] = "NeuralTextures"


# json files to use
SPLIT_TO_JSON = {}

SPLIT_TO_JSON["train"] = TRAIN_JSON
SPLIT_TO_JSON["val"] = VAL_JSON
SPLIT_TO_JSON["test"] = TEST_JSON


# how many samples to use per image, for eahc split (taken from original ff++ paper)
SPLIT_TO_SAMPLES = {}

SPLIT_TO_SAMPLES["train"] = 27
SPLIT_TO_SAMPLES["val"] = 10
SPLIT_TO_SAMPLES["test"] = 10


def create_output_folders():

    for split_name in SPLIT_TO_JSON.keys():
        split_path = OUTPUT_DATASET_PATH / split_name
        
        for class_name in CLASS_TO_FOLDER.keys():
            class_path = split_path / class_name
            class_path.mkdir(parents=True, exist_ok=True)


# reads json file and returns list of pairs
def load_split_pairs(json_path):
    with open(json_path, "r") as file:
        pairs = json.load(file)
    return pairs


# ensure that the same pair doesn't appear in multiple splits (es. 000_003 should be in train, but not in test set)
def check_pair_leakage():

    split_to_pair_names = {}

    for split_name in SPLIT_TO_JSON.keys():
        json_path = SPLIT_TO_JSON[split_name]
        pairs = load_split_pairs(json_path)
        pair_names = set() # set, to check duplicates efficiently

        for pair in pairs: # ["071", "054"]
            source_id = pair[0] # "071"
            target_id = pair[1] # "054"
            pair_name = source_id + "_" + target_id # 071_054
            pair_names.add(pair_name)
            
        split_to_pair_names[split_name] = pair_names

    split_names = list(split_to_pair_names.keys())

    # train vs val, train vs test, val vs test
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            split_a = split_names[i]
            split_b = split_names[j]
            overlap = split_to_pair_names[split_a].intersection(split_to_pair_names[split_b]) # checks for pair overlaps
            print(split_a, "vs", split_b, "pair overlap:", len(overlap))


# ensure the same video versions belong to the same split (es. 000_003.mp4 should be in the same split as original/003.mp4)
def check_target_leakage():

    split_to_targets = {}

    for split_name in SPLIT_TO_JSON.keys():
        json_path = SPLIT_TO_JSON[split_name]
        pairs = load_split_pairs(json_path)
        targets = set()

        for pair in pairs: # ["000", "003"]
            target_id = pair[1] # "003"
            targets.add(target_id)

        split_to_targets[split_name] = targets
        print(split_name, "target videos:", len(targets))

    split_names = list(split_to_targets.keys())

    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            split_a = split_names[i]
            split_b = split_names[j]
            overlap = split_to_targets[split_a].intersection(split_to_targets[split_b])
            print(split_a, "vs", split_b, "target overlap:", len(overlap))


# returns the list of frame numbers we want to extract from a particular video
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
    haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(haar_path)
    if detector.empty():
        raise RuntimeError("Haar cascade not loaded correctly.")
    return detector


# face region detection with haar detector
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
        area = w * h # bb area of the face

        # take the largest face
        if area > best_area:
            best_area = area
            best_face = (x, y, w, h)

    return best_face


# crop face from image, do 1.3x, and 0-pad if it goes out of frame
def crop_face_with_padding(frame, face, crop_scale):

    x = face[0]
    y = face[1]
    w = face[2]
    h = face[3]

    frame_height = frame.shape[0]
    frame_width = frame.shape[1]
    crop_size = int(round(max(w, h) * crop_scale)) # 1.3x

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

    # 0-pad if the cropped frame goes out of bound in at least one direction
    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        crop = cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    return crop


def extract_faces_from_video(video_path, output_class_path, split_name, class_name, video_key, source_id, target_id, number_of_samples, detector):

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # number of frames of the video
    frame_numbers = get_uniform_frame_numbers(total_frames, number_of_samples) # which frames to sample

    saved_count = 0
    missed_faces = 0
    read_errors = 0
    write_errors = 0

    for sample_index in range(len(frame_numbers)): # for each frame
        frame_number = frame_numbers[sample_index]
        output_name = class_name + "_" + video_key + "_sample_" + str(sample_index).zfill(3) + "_frame_" + str(frame_number).zfill(6) + ".png" # es. Deepfakes_000_003_sample_004_frame_000123.png
        output_path = output_class_path / output_name

        if SKIP_EXISTING and output_path.exists():
            saved_count = saved_count + 1
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()

        if not ret:
            read_errors = read_errors + 1 # read image fails
            continue

        face = detect_largest_face(frame, detector)

        if face is None:
            missed_faces = missed_faces + 1 # detect largest face fails
            continue

        crop = crop_face_with_padding(frame, face, CROP_SCALE)

        if crop is None:
            missed_faces = missed_faces + 1 # crop face fails
            continue

        crop = cv2.resize(crop, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA) # resize crop to 224x224
        ok = cv2.imwrite(str(output_path), crop)

        if ok:
            saved_count = saved_count + 1
        else:
            write_errors = write_errors + 1 # save image fails

    cap.release() # close video
    
    summary = {}
    summary["split"] = split_name
    summary["class"] = class_name
    summary["video_path"] = str(video_path)
    summary["video_key"] = video_key
    summary["source_id"] = source_id
    summary["target_id"] = target_id
    summary["total_frames"] = total_frames
    summary["requested_samples"] = number_of_samples
    summary["saved_images"] = saved_count
    summary["missed_faces"] = missed_faces
    summary["read_errors"] = read_errors
    summary["write_errors"] = write_errors

    return summary


# find_video_path("Deepfakes", "000_003") -> datasets/faceforensics++/Deepfakes/000_003.mp4
def find_video_path(class_name, video_stem):
    input_folder = CLASS_TO_FOLDER[class_name]
    video_path = DATASET_PATH / input_folder / (video_stem + ".mp4")
    if video_path.exists():
        return video_path
    return None


# main function
def create_cropped_dataset():
    create_output_folders()
    detector = get_haar_detector()
    summaries = [] # stats
    missing_videos = []

    for split_name in SPLIT_TO_JSON.keys(): # for each split
        json_path = SPLIT_TO_JSON[split_name]
        pairs = load_split_pairs(json_path) # load pairs
        number_of_samples = SPLIT_TO_SAMPLES[split_name] # how many frames to extract (270 for train, 100 else)
        processed_originals = set()

        if MAX_PAIRS_PER_SPLIT is not None: # to do some tests to see if function works, we can selectr only a fraction of the pairs
            pairs = pairs[:MAX_PAIRS_PER_SPLIT]

        print("Processing split:", split_name)
        print("Pairs:", len(pairs))

        for pair in tqdm(pairs): # ["000","003"]
            id_a = pair[0] # "000"
            id_b = pair[1] # "003"
            directions = [] # we need to consider both directions!
            directions.append((id_a, id_b)) # 000_003
            directions.append((id_b, id_a)) # 003_000

            for source_id, target_id in directions: # for both directions
                pair_name = source_id + "_" + target_id

                # check if target video (original) still has not been processed
                if target_id not in processed_originals:
                    class_name = "Original"
                    video_stem = target_id # "003"
                    video_key = target_id
                    video_path = find_video_path(class_name, video_stem) # original/003.mp4
                    output_class_path = OUTPUT_DATASET_PATH / split_name / class_name

                    # if it couldn't find video, save info
                    if video_path is None:
                        missing_row = {}
                        missing_row["split"] = split_name
                        missing_row["class"] = class_name
                        missing_row["video_stem"] = video_stem
                        missing_row["source_id"] = source_id
                        missing_row["target_id"] = target_id
                        missing_videos.append(missing_row)
                    else:
                        summary = extract_faces_from_video(video_path, output_class_path, split_name, class_name, video_key, source_id, target_id, number_of_samples, detector)
                        summaries.append(summary)

                    processed_originals.add(target_id) # flag this target id as already processed

                # for every fake class
                for class_name in CLASS_TO_FOLDER.keys():
                    if class_name == "Original":
                        continue

                    video_stem = pair_name # "000_003" or "003_000"
                    video_key = pair_name
                    video_path = find_video_path(class_name, video_stem)
                    output_class_path = OUTPUT_DATASET_PATH / split_name / class_name

                    # if it couldn't find video, save info
                    if video_path is None:
                        missing_row = {}
                        missing_row["split"] = split_name
                        missing_row["class"] = class_name
                        missing_row["video_stem"] = video_stem
                        missing_row["source_id"] = source_id
                        missing_row["target_id"] = target_id
                        missing_videos.append(missing_row)
                        continue

                    summary = extract_faces_from_video(video_path, output_class_path, split_name, class_name, video_key, source_id, target_id, number_of_samples, detector)
                    summaries.append(summary)

    summaries_df = pd.DataFrame(summaries)
    missing_df = pd.DataFrame(missing_videos)
    summaries_df.to_csv(OUTPUT_DATASET_PATH / "preprocessing_summary.csv", index=False)
    missing_df.to_csv(OUTPUT_DATASET_PATH / "missing_videos.csv", index=False)

    print("Done.")
    print("Summary saved to:", OUTPUT_DATASET_PATH / "preprocessing_summary.csv")
    print("Missing videos saved to:", OUTPUT_DATASET_PATH / "missing_videos.csv")

    return summaries_df, missing_df


# count how many images have been created for each split and class
def count_created_images():
    for split_name in SPLIT_TO_JSON.keys():
        for class_name in CLASS_TO_FOLDER.keys():
            class_path = OUTPUT_DATASET_PATH / split_name / class_name
            image_count = len(list(class_path.glob("*.png")))
            print(split_name, class_name, image_count)


def main():
    # check json files are loaded correctly
    train_pairs = load_split_pairs(TRAIN_JSON)
    val_pairs = load_split_pairs(VAL_JSON)
    test_pairs = load_split_pairs(TEST_JSON)

    print("Train pairs:", len(train_pairs))
    print("Val pairs:", len(val_pairs))
    print("Test pairs:", len(test_pairs))

    check_pair_leakage()
    check_target_leakage()

    summaries_df, missing_df = create_cropped_dataset()

    print("Created images:")
    count_created_images()


if __name__ == "__main__":
    main()