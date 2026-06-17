import os
import json
import glob


import json


def load_pose_file_multi(filepath):
    """
    Load JSON file structured as:
    {
        "persons": [
            { "person_id": int, "poses": [ {frame_idx, keypoints, keypoints_visible}, ... ] },
            ...
        ]
    }
    Returns:
        A dict: person_id -> (all_keypoints, all_confidence)
    """
    with open(filepath, "r") as f:
        data = json.load(f)

    result = {}

    for person in data["persons"]:
        pid = person["person_id"]
        poses = person["poses"]

        all_kpts = []
        all_conf = []

        for frame in poses:
            kpts = frame["keypoints"][0]  # (num_joints, 2)
            conf = frame["keypoints_visible"][0]  # (num_joints,)
            all_kpts.append(kpts)
            all_conf.append(conf)

        result[pid] = (all_kpts, all_conf)

    return result


def get_all_json_files(folder):
    """Return a dict mapping filenames -> full paths, searching all subfolders."""
    file_dict = {}
    pattern = os.path.join(folder, "**", "*.json")

    for path in glob.glob(pattern, recursive=True):
        fname = os.path.basename(path)
        file_dict[fname] = path

    return file_dict


def get_matching_files(orig_folder, zero_folder, log_folder):
    """Return matched (orig, zero, log) triplets by filename."""
    orig_files = get_all_json_files(orig_folder)
    zero_files = get_all_json_files(zero_folder)
    log_files = get_all_json_files(log_folder)

    matches = []

    for fname, orig_path in orig_files.items():
        if fname in zero_files and fname in log_files:
            matches.append((orig_path, zero_files[fname], log_files[fname]))
        else:
            print(f"Missing match for {fname}")

    print(f"\nMatched {len(matches)} files.\n")
    return matches


# -------------------------------
# 🔥 Example usage
# -------------------------------
original_folder = r"C:\Users\BhavyaSehgal\Downloads\Humaneva_spatial_results\detect_RTMW\detect\2025-09-18_00-46-59\S3"
zero_folder = r"C:\Users\BhavyaSehgal\Downloads\Humaneva_spatial_results\zero_dce\detect\2025-12-02_07-09-46\S3"
log_folder = r"C:\Users\BhavyaSehgal\Downloads\Humaneva_spatial_results\LogTransform\detect\2025-12-02_07-09-46\S3"

matched_files = get_matching_files(original_folder, zero_folder, log_folder)

for orig_path, zero_path, log_path in matched_files:
    print("\nProcessing:", os.path.basename(orig_path))

    # Load all persons for each version
    orig_data = load_pose_file_multi(orig_path)
    zero_data = load_pose_file_multi(zero_path)
    log_data = load_pose_file_multi(log_path)

    # Always take FIRST person
    orig_pid = list(orig_data.keys())[0]
    zero_pid = list(zero_data.keys())[0]
    log_pid = list(log_data.keys())[0]

    print("Original person_id:", orig_pid)
    print("Zero-DCE person_id:", zero_pid)
    print("Log person_id:", log_pid)

    # Extract their (keypoints, confidence)
    orig_kpts, orig_conf = orig_data[orig_pid]
    zero_kpts, zero_conf = zero_data[zero_pid]
    log_kpts, log_conf = log_data[log_pid]

    # Example: Frame 0
    f = 0
    print("Frames available:", len(orig_conf))

    print("Original conf[f] shape:", len(orig_conf[f]))
    print("Zero-DCE conf[f] shape:", len(zero_conf[f]))
    print("Log conf[f] shape:", len(log_conf[f]))

    # Example: Left ankle index
    left_ankle_idx = 15

    print("Original LA conf =", orig_conf[f][left_ankle_idx])
    print("Zero-DCE LA conf =", zero_conf[f][left_ankle_idx])
    print("Log LA conf =", log_conf[f][left_ankle_idx])
