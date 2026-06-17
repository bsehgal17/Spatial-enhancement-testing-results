import os
import cv2
import numpy as np

# ----------------- USER PARAMETERS -----------------
input_root = r"C:\Users\BhavyaSehgal\Downloads\HumanEva\S3\Image_Data"
output_root = r"/storage/Projects/Gaitly/bsehgal/Spatial_testing/ESIHE/Output_Videos"

video_extensions = (".mp4", ".avi", ".mov", ".mkv")

os.makedirs(output_root, exist_ok=True)


# ----------------- ESIHE Enhancement -----------------
def esihe_frame(bgr):
    # Convert to grayscale for histogram processing
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    L = 255  # max intensity
    # Compute mean intensity (exposure)
    mean_intensity = np.mean(gray)

    # Split into underexposed and overexposed regions
    lower_mask = gray <= mean_intensity
    upper_mask = gray > mean_intensity

    out = np.zeros_like(gray)

    # Lower sub-image histogram equalization
    if np.any(lower_mask):
        lower_vals = gray[lower_mask]
        hist, bins = np.histogram(
            lower_vals, bins=int(mean_intensity) + 1, range=(0, mean_intensity)
        )
        cdf = np.cumsum(hist) / (hist.sum() + 1e-6)
        equalized_lower = np.interp(lower_vals, bins[:-1], cdf * mean_intensity)
        out[lower_mask] = equalized_lower

    # Upper sub-image histogram equalization
    if np.any(upper_mask):
        upper_vals = gray[upper_mask]
        hist, bins = np.histogram(
            upper_vals, bins=256 - int(mean_intensity), range=(mean_intensity, 255)
        )
        cdf = np.cumsum(hist) / (hist.sum() + 1e-6)
        equalized_upper = np.interp(
            upper_vals, bins[:-1], mean_intensity + cdf * (255 - mean_intensity)
        )
        out[upper_mask] = equalized_upper

    out = np.clip(out, 0, 255).astype(np.uint8)

    # Apply color restoration
    img = bgr.astype(np.float32)
    gray_orig = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) + 1e-6
    out_color = np.zeros_like(img)
    for i in range(3):
        out_color[:, :, i] = img[:, :, i] * (out / gray_orig)
    out_color = np.clip(out_color, 0, 255).astype(np.uint8)
    return out_color


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

    out_esihe = os.path.join(video_folder, f"{video_name}_ESIHE.mp4")
    vw_esihe = cv2.VideoWriter(out_esihe, fourcc, fps, (width, height))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    processed = 0
    print(f"Processing {in_path} -> {frame_count} frames")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_esihe = esihe_frame(frame)
        vw_esihe.write(frame_esihe)

        processed += 1
        if processed % 100 == 0:
            print(f"  processed {processed}/{frame_count} frames...")

    cap.release()
    vw_esihe.release()
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
