import os
import cv2
import numpy as np

# ----------------- USER PARAMETERS -----------------
input_root = (
    r"/storage/Projects/Gaitly/bsehgal/lower_body_pose_est/HumanEva/S3/Image_Data/"
)
output_root = r"/storage/Projects/Gaitly/bsehgal/Spatial_testing/BPDFHE/Output_Videos"

video_extensions = (".mp4", ".avi", ".mov", ".mkv")

L = 256
EPS = 1e-6
FUZZY_SUPPORT = 1  # spread contribution to neighbors for fuzzy histogram

CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)

W_BPDHE = 0.7
W_CLAHE = 0.3

os.makedirs(output_root, exist_ok=True)


# ----------------- Fuzzy BPDFHE Implementation -----------------
def fuzzy_histogram(gray):
    """Compute fuzzy histogram with triangular membership function."""
    hist = np.zeros(L, dtype=np.float64)
    h, w = gray.shape
    for i in range(h):
        for j in range(w):
            val = gray[i, j]
            for k in range(-FUZZY_SUPPORT, FUZZY_SUPPORT + 1):
                idx = int(val + k)
                if 0 <= idx < L:
                    hist[idx] += max(0, 1 - abs(k))  # triangular weighting
    return hist


def find_local_maxima_fuzzy(hist):
    """Find local maxima in fuzzy histogram."""
    maxima = []
    for i in range(1, L - 1):
        if hist[i] > hist[i - 1] and hist[i] > hist[i + 1]:
            maxima.append(i)
    if 0 not in maxima:
        maxima = [0] + maxima
    if L - 1 not in maxima:
        maxima.append(L - 1)
    return sorted(list(dict.fromkeys(maxima)))


def bpdfhe_frame_fuzzy(bgr, FUZZY_SUPPORT=2, EPS=1e-6, cdf_clip=(0.05, 0.95)):
    """
    Modified BPDFHE (BDFE) with noise reduction:
    - Fuzzy histogram with larger support
    - Clip CDF to prevent over-enhancement
    - Mild Gaussian smoothing
    - Color restoration
    """
    L = 256
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # ---------------- Fuzzy histogram ----------------
    hist = np.zeros(L, dtype=np.float64)
    h, w = gray.shape
    for i in range(h):
        for j in range(w):
            val = gray[i, j]
            for k in range(-FUZZY_SUPPORT, FUZZY_SUPPORT + 1):
                idx = int(val + k)
                if 0 <= idx < L:
                    hist[idx] += max(
                        0, 1 - abs(k) / FUZZY_SUPPORT
                    )  # smoother triangular weighting

    # ---------------- Local maxima ----------------
    maxima = []
    for i in range(1, L - 1):
        if hist[i] > hist[i - 1] and hist[i] > hist[i + 1]:
            maxima.append(i)
    if 0 not in maxima:
        maxima = [0] + maxima
    if L - 1 not in maxima:
        maxima.append(L - 1)
    maxima = sorted(list(dict.fromkeys(maxima)))

    # ---------------- Partition histogram ----------------
    partitions = []
    for i in range(len(maxima) - 1):
        low, high = maxima[i], maxima[i + 1]
        if high < low:
            low, high = high, low
        partitions.append((low, high))

    # ---------------- Dynamic equalization per partition ----------------
    mapping = np.arange(L, dtype=np.uint8)
    for low, high in partitions:
        sub_hist = hist[low : high + 1]
        M = sub_hist.sum()
        if M <= 0:
            for k in range(low, high + 1):
                mapping[k] = np.uint8((k - low) * 255 / max(1, high - low))
            continue
        cdf = np.cumsum(sub_hist) / M
        # Clip CDF to prevent over-enhancement
        cdf = np.clip(cdf, cdf_clip[0], cdf_clip[1])
        for idx, val in enumerate(cdf):
            mapping[low + idx] = np.uint8(round(val * 255))

    # ---------------- Apply mapping ----------------
    enhanced = mapping[gray]

    # ---------------- Mild smoothing ----------------
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)

    # ---------------- Brightness normalization ----------------
    mean_orig = float(np.mean(gray))
    mean_enh = float(np.mean(enhanced)) if np.mean(enhanced) > 0 else 1.0
    scale = mean_orig / (mean_enh + EPS)
    enhanced = np.clip(enhanced.astype(np.float32) * scale, 0, 255).astype(np.uint8)

    # ---------------- Color restoration ----------------
    b, g, r = cv2.split(bgr.astype(np.float32))
    lum_orig = gray.astype(np.float32) + EPS
    lum_enh = enhanced.astype(np.float32) + EPS

    b_out = np.clip(b * (lum_enh / lum_orig), 0, 255).astype(np.uint8)
    g_out = np.clip(g * (lum_enh / lum_orig), 0, 255).astype(np.uint8)
    r_out = np.clip(r * (lum_enh / lum_orig), 0, 255).astype(np.uint8)

    return cv2.merge([b_out, g_out, r_out])


# ----------------- CLAHE + Color Restoration -----------------
def clahe_color_restore_frame(bgr):
    img = bgr.astype(np.float32) / 255.0
    R = img[:, :, 2]
    G = img[:, :, 1]
    B = img[:, :, 0]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Y_8u = np.clip((Y * 255.0).astype(np.uint8), 0, 255)

    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
    Y_clahe = clahe.apply(Y_8u).astype(np.float32) / 255.0

    Tm = float(np.mean(Y_clahe))
    Lbright = Tm
    s = 1.0 - np.tanh(Lbright)

    Ynorm = np.clip(Y + EPS, EPS, None)
    ratioR = np.clip(R / Ynorm, 0.0, 10.0)
    ratioG = np.clip(G / Ynorm, 0.0, 10.0)
    ratioB = np.clip(B / Ynorm, 0.0, 10.0)

    R_out = Y_clahe * (ratioR**s)
    G_out = Y_clahe * (ratioG**s)
    B_out = Y_clahe * (ratioB**s)

    out = np.stack([B_out, G_out, R_out], axis=2)
    out = np.clip(out, 0.0, 1.0)
    return (out * 255.0).astype(np.uint8)


# ----------------- Fusion -----------------
def fuse_frames(bpdhe_bgr, clahe_bgr, w1=W_BPDHE, w2=W_CLAHE):
    a = bpdhe_bgr.astype(np.float32)
    b = clahe_bgr.astype(np.float32)
    fused = a * float(w1) + b * float(w2)
    return np.clip(fused, 0, 255).astype(np.uint8)


# ----------------- Video processing pipeline -----------------
def process_video_file(in_path, out_base_dir, weights=(W_BPDHE, W_CLAHE)):
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        print("Failed to open:", in_path)
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    video_name = os.path.splitext(os.path.basename(in_path))[0]
    video_folder = os.path.join(out_base_dir, video_name)
    os.makedirs(video_folder, exist_ok=True)

    out_bpdhe = os.path.join(video_folder, f"{video_name}_BPDFHE.mp4")
    out_clahe = os.path.join(video_folder, f"{video_name}_CLAHE_colorrest.mp4")
    w1, w2 = weights
    out_fused = os.path.join(
        video_folder, f"{video_name}_FUSED_w{w1:.2f}_w{w2:.2f}.mp4"
    )

    vw_bpdhe = cv2.VideoWriter(out_bpdhe, fourcc, fps, (width, height))
    vw_clahe = cv2.VideoWriter(out_clahe, fourcc, fps, (width, height))
    vw_fused = cv2.VideoWriter(out_fused, fourcc, fps, (width, height))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    processed = 0
    print(f"Processing {in_path} -> {frame_count} frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_bpdhe = bpdfhe_frame_fuzzy(frame)
        frame_clahe = clahe_color_restore_frame(frame)
        frame_fused = fuse_frames(frame_bpdhe, frame_clahe, w1=w1, w2=w2)

        vw_bpdhe.write(frame_bpdhe)
        vw_clahe.write(frame_clahe)
        vw_fused.write(frame_fused)

        processed += 1
        if processed % 100 == 0:
            print(f"  processed {processed}/{frame_count} frames...")

    cap.release()
    vw_bpdhe.release()
    vw_clahe.release()
    vw_fused.release()
    print("Done:", in_path)


def process_folder(input_folder, output_folder, weights=(W_BPDHE, W_CLAHE)):
    for root, dirs, files in os.walk(input_folder):
        for fn in files:
            if fn.lower().endswith(video_extensions):
                in_path = os.path.join(root, fn)
                rel = os.path.relpath(root, input_folder)
                out_dir = os.path.join(output_folder, rel)
                process_video_file(in_path, out_dir, weights=weights)


# ----------------- Run -----------------
if __name__ == "__main__":
    process_folder(input_root, output_root, weights=(W_BPDHE, W_CLAHE))
    print("All done.")
