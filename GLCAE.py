import os
import cv2
import numpy as np
from scipy.ndimage import laplace

# ----------------- USER PARAMETERS -----------------
input_root = (
    r"/storage/Projects/Gaitly/bsehgal/lower_body_pose_est/HumanEva/S3/Image_Data/"
)
output_root = r"/storage/Projects/Gaitly/bsehgal/Spatial_testing/GLCAE/Output_Videos"

video_extensions = (".mp4", ".avi", ".mov", ".mkv")

# CLAHE parameters for local enhancement
CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)

# Brightness weighting sigma
BRIGHTNESS_SIGMA = 0.2

os.makedirs(output_root, exist_ok=True)


# ----------------- Hue-preserving framework -----------------
def hue_preserve(original_rgb, original_intensity, enhanced_intensity):
    out = np.zeros_like(original_rgb, dtype=np.float32)
    mask = enhanced_intensity <= original_intensity
    for c in range(3):
        # formula from paper
        out[..., c] = np.where(
            mask,
            enhanced_intensity / (original_intensity + 1e-6) * original_rgb[..., c],
            ((255 - enhanced_intensity) / (255 - original_intensity + 1e-6))
            * (original_rgb[..., c] - original_intensity)
            + enhanced_intensity,
        )
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


# ----------------- Global Contrast Adaptive Enhancement -----------------
def global_contrast_adaptive(bgr):
    img = bgr.astype(np.float32)
    Xmin = np.min(img)
    Xmax = np.max(img)
    # Linear stretching
    X_hat = (img - Xmin) / (Xmax - Xmin + 1e-6) * 255.0
    X_hat = np.clip(X_hat, 0, 255)

    # Convert to intensity
    I = 0.299 * X_hat[..., 2] + 0.587 * X_hat[..., 1] + 0.114 * X_hat[..., 0]
    I = np.clip(I, 0, 255)

    # Histogram
    hist, _ = np.histogram(I.flatten(), bins=256, range=(0, 255))
    hist = hist / hist.sum()

    # Uniform histogram
    hU = np.ones(256) / 256.0

    # Weighted sum (lambda can be tuned)
    lam = 0.5  # can be adaptive later
    h_opt = (1 / (1 + lam)) * hist + (lam / (1 + lam)) * hU

    # Mapping function
    cdf = np.cumsum(h_opt)
    T = np.floor(cdf * 255).astype(np.uint8)

    # Apply mapping
    GI = T[I.astype(int)]

    # Hue-preserving color reconstruction
    Gc = hue_preserve(X_hat, I, GI)
    return Gc


# ----------------- Local Contrast Adaptive Enhancement -----------------
def local_contrast_adaptive(bgr):
    img = bgr.astype(np.float32)
    # Convert to intensity
    I = 0.299 * img[..., 2] + 0.587 * img[..., 1] + 0.114 * img[..., 0]
    I = np.clip(I, 0, 255).astype(np.uint8)

    # CLAHE on intensity
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
    LI = clahe.apply(I)

    # Hue-preserving color reconstruction
    Lc = hue_preserve(img, I, LI)
    return Lc


# ----------------- Contrast-Brightness-based Fusion -----------------
def contrast_brightness_fusion(Gc, Lc):
    # Convert to intensity
    IG = 0.299 * Gc[..., 2] + 0.587 * Gc[..., 1] + 0.114 * Gc[..., 0]
    IL = 0.299 * Lc[..., 2] + 0.587 * Lc[..., 1] + 0.114 * Lc[..., 0]

    # Contrast measure via Laplacian
    CG = np.abs(laplace(IG))
    CL = np.abs(laplace(IL))

    # Brightness measure
    BG = np.exp(-((IG / 255 - 0.5) ** 2) / (2 * BRIGHTNESS_SIGMA**2))
    BL = np.exp(-((IL / 255 - 0.5) ** 2) / (2 * BRIGHTNESS_SIGMA**2))

    # Weight maps
    WG = np.minimum(CG, BG)
    WL = np.minimum(CL, BL)

    # Normalize weights
    W_sum = WG + WL + 1e-6
    WG_hat = WG / W_sum
    WL_hat = WL / W_sum

    # Weighted fusion
    Yc = np.zeros_like(Gc, dtype=np.float32)
    for c in range(3):
        Yc[..., c] = WG_hat * Gc[..., c] + WL_hat * Lc[..., c]
    Yc = np.clip(Yc, 0, 255).astype(np.uint8)
    return Yc


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

    out_path = os.path.join(video_folder, f"{video_name}_GLCAE.mp4")
    vw = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    processed = 0
    print(f"Processing {in_path} -> {frame_count} frames")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        Gc = global_contrast_adaptive(frame)
        Lc = local_contrast_adaptive(frame)
        Yc = contrast_brightness_fusion(Gc, Lc)
        vw.write(Yc)

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
