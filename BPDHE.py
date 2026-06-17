import os
import cv2
import numpy as np

# ----------------- USER PARAMETERS -----------------
input_root = (
    r"/storage/Projects/Gaitly/bsehgal/lower_body_pose_est/HumanEva/S3/Image_Data/"
)
output_root = r"/storage/Projects/Gaitly/bsehgal/Spatial_testing/BPDHE/Output_Videos"

video_extensions = (".mp4", ".avi", ".mov", ".mkv")

# BPDHE parameters
GAUSS_KERNEL_SIZE = 9
GAUSS_SIGMA = 1.0762

L = 256
MIN_PEAK_DIST = 1

CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)

EPS = 1e-6

W_BPDHE = 0.7
W_CLAHE = 0.3

os.makedirs(output_root, exist_ok=True)


# ----------------- BPDHE Implementation -----------------
def smooth_histogram(hist):
    k = cv2.getGaussianKernel(GAUSS_KERNEL_SIZE, GAUSS_SIGMA).flatten()
    if len(k) != GAUSS_KERNEL_SIZE:
        k = k[:GAUSS_KERNEL_SIZE]
    padded = np.pad(hist, (GAUSS_KERNEL_SIZE // 2,), mode="reflect")
    sm = np.convolve(padded, k, mode="valid")
    return sm


def find_local_maxima(hsmoothed):
    arr = hsmoothed
    maxima = []
    for i in range(1, len(arr) - 1):
        if arr[i] > arr[i - 1] and arr[i] > arr[i + 1]:
            maxima.append(i)
    if 0 not in maxima:
        maxima = [0] + maxima
    if (len(arr) - 1) not in maxima:
        maxima = maxima + [len(arr) - 1]
    maxima = sorted(list(dict.fromkeys(maxima)))
    return maxima


def bpdhe_frame(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    hist = cv2.calcHist([gray], [0], None, [L], [0, L]).flatten().astype(np.float64)

    x = np.arange(L)
    nonzero = hist > 0
    if nonzero.sum() < 2:
        return bgr.copy()
    interp_vals = np.interp(x, x[nonzero], hist[nonzero])
    hist_filled = interp_vals

    hsm = smooth_histogram(hist_filled)
    maxima = find_local_maxima(hsm)

    parts = []
    for i in range(len(maxima) - 1):
        low = maxima[i]
        high = maxima[i + 1]
        if high < low:
            low, high = high, low
        parts.append((int(low), int(high)))

    if len(parts) == 0:
        eq = cv2.equalizeHist(gray)
        return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)

    factors = []
    ranges = []
    counts = []

    for low, high in parts:
        low = max(0, low)
        high = min(L - 1, high)
        span = (high - low) if (high - low) > 0 else 1
        M_i = hist[int(low) : int(high) + 1].sum()
        if M_i <= 0:
            M_i = 1.0
        factor = span * np.log10(M_i)
        factors.append(max(factor, 0.0001))
        ranges.append(span)
        counts.append(M_i)

    factors = np.array(factors, dtype=np.float64)
    total_factor = factors.sum()
    if total_factor <= 0:
        total_factor = 1.0
    ranges_out = ((L - 1) * (factors / total_factor)).astype(np.int32)
    ranges_out[ranges_out == 0] = 1

    starts = []
    ends = []
    cur = 0
    for r in ranges_out:
        start = cur
        end = cur + int(r)
        starts.append(start)
        ends.append(end)
        cur = end + 1

    if ends:
        ends[-1] = L - 1

    mapping = np.arange(L, dtype=np.uint8)
    for idx, ((low, high), start, end) in enumerate(zip(parts, starts, ends)):
        low = int(max(0, low))
        high = int(min(L - 1, high))
        if high < low:
            continue
        sub_counts = hist[low : high + 1].astype(np.float64)
        M_sub = sub_counts.sum()
        if M_sub <= 0:
            for k in range(low, high + 1):
                mapping[k] = np.uint8(
                    np.clip(
                        start + (end - start) * (k - low) / max(1, (high - low)), 0, 255
                    )
                )
            continue

        cdf = np.cumsum(sub_counts) / M_sub
        for offset_k, c in enumerate(cdf):
            k = low + offset_k
            mapped = start + (end - start) * c
            mapping[k] = np.uint8(np.clip(round(mapped), 0, 255))

    enhanced = mapping[gray]

    mean_orig = float(np.mean(gray))
    mean_enh = float(np.mean(enhanced)) if np.mean(enhanced) > 0 else 1.0
    scale = mean_orig / (mean_enh + EPS)
    enhanced = np.clip((enhanced.astype(np.float32) * scale), 0, 255).astype(np.uint8)

    b, g, r = cv2.split(bgr.astype(np.float32))
    lum_orig = gray.astype(np.float32) + EPS
    lum_enh = enhanced.astype(np.float32) + EPS

    b_out = np.clip(b * (lum_enh / lum_orig), 0, 255).astype(np.uint8)
    g_out = np.clip(g * (lum_enh / lum_orig), 0, 255).astype(np.uint8)
    r_out = np.clip(r * (lum_enh / lum_orig), 0, 255).astype(np.uint8)
    bgr_out = cv2.merge([b_out, g_out, r_out])
    return bgr_out


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
    out = (out * 255.0).astype(np.uint8)
    return out


# ----------------- Fusion -----------------
def fuse_frames(bpdhe_bgr, clahe_bgr, w1=W_BPDHE, w2=W_CLAHE):
    a = bpdhe_bgr.astype(np.float32)
    b = clahe_bgr.astype(np.float32)
    fused = a * float(w1) + b * float(w2)
    fused = np.clip(fused, 0, 255).astype(np.uint8)
    return fused


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

    # ---------------------- NEW CHANGE ----------------------
    video_folder = os.path.join(out_base_dir, video_name)
    os.makedirs(video_folder, exist_ok=True)
    # -------------------------------------------------------

    out_bpdhe = os.path.join(video_folder, f"{video_name}_BPDHE.mp4")
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
    print(
        f"Processing {in_path} ->\n  {out_bpdhe}\n  {out_clahe}\n  {out_fused}\n frames: {frame_count}"
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_bpdhe = bpdhe_frame(frame)
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
