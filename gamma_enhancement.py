import os
import cv2
import numpy as np

# ----------------- USER PARAMETERS -----------------
input_root = (
    r"C/storage/Projects/Gaitly/bsehgal/lower_body_pose_est/HumanEva/S3/Image_Data/"
)
output_root = r"/storage/Projects/Gaitly/bsehgal/Spatial_testing/Gamma/Output_Videos"

video_extensions = (".mp4", ".avi", ".mov", ".mkv")

# Gamma parameter
GAMMA = 0.5  # <1 brightens, >1 darkens

os.makedirs(output_root, exist_ok=True)


# ----------------- Gamma Enhancement -----------------
def gamma_enhance_frame(bgr, gamma=GAMMA):
    # Convert to float in [0,1]
    img = bgr.astype(np.float32) / 255.0
    img_gamma = np.power(img, gamma)
    # Back to uint8
    out = np.clip(img_gamma * 255.0, 0, 255).astype(np.uint8)
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

    out_gamma = os.path.join(video_folder, f"{video_name}_Gamma_{GAMMA:.2f}.mp4")
    vw_gamma = cv2.VideoWriter(out_gamma, fourcc, fps, (width, height))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    processed = 0
    print(f"Processing {in_path} -> {frame_count} frames")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gamma = gamma_enhance_frame(frame)
        vw_gamma.write(frame_gamma)

        processed += 1
        if processed % 100 == 0:
            print(f"  processed {processed}/{frame_count} frames...")

    cap.release()
    vw_gamma.release()
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
