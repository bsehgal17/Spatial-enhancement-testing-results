import os
import cv2
import numpy as np
from skimage.morphology import disk, white_tophat

# ----------------- USER PARAMETERS -----------------
input_root = (
    r"/storage/Projects/Gaitly/bsehgal/lower_body_pose_est/HumanEva/S3/Image_Data/"
)
output_root = (
    r"/storage/Projects/Gaitly/bsehgal/Spatial_testing/DarkEnhancement/Output_Videos"
)

video_extensions = (".mp4", ".avi", ".mov", ".mkv")
CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)
TOPHAT_RADIUS = 3
GAMMAS = [1, 5]  # gamma values

os.makedirs(output_root, exist_ok=True)


# ----------------- Dark Image Enhancement Pipeline -----------------
def dark_image_enhancement(bgr):
    # 1. Convert RGB to HSV
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    H, S, V = cv2.split(hsv)

    # 2. Normalize intensity component to [0,1]
    V_norm = V / 255.0

    # 3. Inverse of normalized intensity
    V_inv = 1.0 - V_norm

    # 4. CLAHE on inverse image
    V_inv_8u = np.clip(V_inv * 255.0, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
    V_clahe = clahe.apply(V_inv_8u).astype(np.float32) / 255.0

    # 5. Gamma enhancement with multiple gammas
    gamma_images = []
    for gamma in GAMMAS:
        V_gamma = np.clip(V_clahe**gamma, 0, 1)
        gamma_images.append(V_gamma)

    # 6. Morphological Top-hat transform for weights
    V_top_hat = white_tophat(
        (V_gamma * 255).astype(np.uint8), selem=disk(TOPHAT_RADIUS)
    )
    weights = V_top_hat.astype(np.float32) / 255.0 + 1e-6  # avoid zero

    # 7. PCA-based fusion of gamma-enhanced images with weights
    stacked = np.stack([g * weights for g in gamma_images], axis=-1)
    flat = stacked.reshape(-1, len(GAMMAS))
    cov = np.cov(flat.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    principal_vector = eigvecs[:, np.argmax(eigvals)]
    fused_flat = flat @ principal_vector
    fused = fused_flat.reshape(V.shape)
    fused = np.clip(fused, 0, 1)

    # 8. Inverse of fused image
    fused_inv = 1.0 - fused

    # 9. Replace V component with fused result
    V_final = fused_inv * 255.0
    hsv_final = cv2.merge([H, S, V_final.astype(np.uint8)])

    # 10. Convert back to RGB
    bgr_final = cv2.cvtColor(hsv_final.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return bgr_final


# ----------------- Video Processing -----------------
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
    out_path = os.path.join(video_folder, f"{video_name}_DarkEnhanced.mp4")
    vw = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    processed = 0
    print(f"Processing {in_path} -> {frame_count} frames")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        enhanced_frame = dark_image_enhancement(frame)
        vw.write(enhanced_frame)

        processed += 1
        if processed % 50 == 0:
            print(f"  processed {processed}/{frame_count} frames...")

    cap.release()
    vw.release()
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
