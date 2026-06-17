import os
import cv2
import numpy as np

# ----------------- USER PARAMETERS -----------------
input_root = (
    r"/storage/Projects/Gaitly/bsehgal/lower_body_pose_est/HumanEva/S3/Image_Data/"
)
output_root = r"/storage/Projects/Gaitly/bsehgal/Spatial_testing/MCLAHE/Output_Videos"

video_extensions = (".mp4", ".avi", ".mov", ".mkv")

# MCLAHE parameters
CLAHE_TILE = (8, 8)
BASE_CLIP = 2.0
EPS = 1e-6

os.makedirs(output_root, exist_ok=True)


# ----------------- MCLAHE Implementation -----------------
def mclahe_color_restore_frame(bgr, base_clip=BASE_CLIP, tile_size=CLAHE_TILE):
    img = bgr.astype(np.float32) / 255.0
    R, G, B = img[:, :, 2], img[:, :, 1], img[:, :, 0]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Y_8u = np.clip((Y * 255.0).astype(np.uint8), 0, 255)

    # Entropy
    hist = cv2.calcHist([Y_8u], [0], None, [256], [0, 256]).flatten()
    hist_prob = hist / (hist.sum() + EPS)
    entropy = -np.sum(hist_prob * np.log2(hist_prob + EPS))

    # Local complexity (tile variance)
    h, w = Y_8u.shape
    tile_h, tile_w = tile_size
    variances = [
        np.var(Y_8u[i : i + tile_h, j : j + tile_w])
        for i in range(0, h, tile_h)
        for j in range(0, w, tile_w)
    ]
    local_complexity = float(np.mean(variances))

    # Dynamic clip limit
    clip_limit = base_clip * (1 + 0.5 * entropy / 8.0 + 0.5 * local_complexity / 255.0)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    Y_clahe = clahe.apply(Y_8u).astype(np.float32) / 255.0

    # Brightness adaptation
    Tm = float(np.mean(Y_clahe))
    s = 1.0 - np.tanh(Tm)
    Ynorm = np.clip(Y + EPS, EPS, None)
    ratioR = np.clip(R / Ynorm, 0.0, 10.0)
    ratioG = np.clip(G / Ynorm, 0.0, 10.0)
    ratioB = np.clip(B / Ynorm, 0.0, 10.0)

    R_out = Y_clahe * (ratioR**s)
    G_out = Y_clahe * (ratioG**s)
    B_out = Y_clahe * (ratioB**s)

    out = np.stack([B_out, G_out, R_out], axis=2)
    out = np.clip(out, 0.0, 1.0)
    out = (out * 255.0).astype(np.uint8)
    return out


# ----------------- Video processing pipeline -----------------
def process_video_file(in_path, out_base_dir):
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        print("Failed to open:", in_path)
        return

    width, height = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_name = os.path.splitext(os.path.basename(in_path))[0]

    video_folder = os.path.join(out_base_dir, video_name)
    os.makedirs(video_folder, exist_ok=True)

    out_mclahe = os.path.join(video_folder, f"{video_name}_MCLAHE.mp4")
    vw_mclahe = cv2.VideoWriter(out_mclahe, fourcc, fps, (width, height))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    processed = 0
    print(f"Processing {in_path} -> {frame_count} frames")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_mclahe = mclahe_color_restore_frame(frame)
        vw_mclahe.write(frame_mclahe)

        processed += 1
        if processed % 100 == 0:
            print(f"  processed {processed}/{frame_count} frames...")

    cap.release()
    vw_mclahe.release()
    print("Done:", in_path)


def process_folder(input_folder, output_folder):
    for root, dirs, files in os.walk(input_folder):
        for fn in files:
            if fn.lower().endswith(video_extensions):
                in_path = os.path.join(root, fn)
                rel = os.path.relpath(root, input_folder)
                out_dir = os.path.join(output_folder, rel)
                process_video_file(in_path, out_dir)


# ----------------- Run -----------------
if __name__ == "__main__":
    process_folder(input_root, output_root)
    print("All done.")
