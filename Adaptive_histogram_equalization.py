import os
import cv2
import numpy as np

# Input and output root folders
input_root = (
    r"/storage/Projects/Gaitly/bsehgal/lower_body_pose_est/HumanEva/S3/Image_Data/"
)
output_root = r"/storage/Projects/Gaitly/bsehgal/Spatial_testing/AHE/Output_Videos"

video_extensions = (".mp4", ".avi", ".mov", ".mkv")


# ---------- AHE Utility (Smooth Version) ----------
def ahe_channel(channel, tile_size=16):
    """Apply classic AHE on a single channel with smoothing to reduce tiling."""
    h, w = channel.shape
    tile_h = h // tile_size
    tile_w = w // tile_size
    out = np.zeros_like(channel)

    # Tile-wise equalization
    for i in range(tile_size):
        for j in range(tile_size):
            y0, y1 = i * tile_h, (i + 1) * tile_h if i != tile_size - 1 else h
            x0, x1 = j * tile_w, (j + 1) * tile_w if j != tile_size - 1 else w
            tile = channel[y0:y1, x0:x1]
            out[y0:y1, x0:x1] = cv2.equalizeHist(tile)

    # 🔥 Smoothing to reduce tile boundaries
    out = cv2.GaussianBlur(out, (5, 5), 0)

    return out


# ---------- AHE per color space ----------
def ahe_ycrcb(frame):
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y = ahe_channel(y)
    ycrcb_eq = cv2.merge([y, cr, cb])
    return cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)


def ahe_hls(frame):
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    h, l, s = cv2.split(hls)
    l = ahe_channel(l)
    hls_eq = cv2.merge([h, l, s])
    return cv2.cvtColor(hls_eq, cv2.COLOR_HLS2BGR)


def ahe_yuv(frame):
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(yuv)
    y = ahe_channel(y)
    yuv_eq = cv2.merge([y, u, v])
    return cv2.cvtColor(yuv_eq, cv2.COLOR_YUV2BGR)


def ahe_hsv(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = ahe_channel(v)
    hsv_eq = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)


def ahe_lab(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = ahe_channel(l)
    lab_eq = cv2.merge([l, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


# ---------- Video processing ----------
def process_video(input_path, output_folder_base):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Failed to open video: {input_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Prepare outputs
    writers = {}
    spaces = [
        ("Original", lambda x: x),
        ("HLS_AHE", ahe_hls),
        ("YCrCb_AHE", ahe_ycrcb),
        ("YUV_AHE", ahe_yuv),
        ("HSV_AHE", ahe_hsv),
        ("LAB_AHE", ahe_lab),
    ]

    for space, func in spaces:
        out_path = os.path.join(output_folder_base, f"{space}.mp4")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        writers[space] = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    # Process frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        for space, func in spaces:
            writers[space].write(func(frame))

    cap.release()
    for w in writers.values():
        w.release()
    print(f"Processed video: {input_path}")


# ---------- Folder processing ----------
def process_folder(input_folder, output_folder):
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(video_extensions):
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_folder)
                output_base = os.path.join(
                    output_folder, relative_path, os.path.splitext(file)[0]
                )
                process_video(input_path, output_base)


# ---------- Run ----------
process_folder(input_root, output_root)
