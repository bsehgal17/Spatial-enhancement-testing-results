import os
import cv2
import numpy as np

# Input and output root folders
input_root = (
    r"/storage/Projects/Gaitly/bsehgal/lower_body_pose_est/HumanEva/S3/Image_Data/"
)
output_root = r"/storage/Projects/Gaitly/bsehgal/Spatial_testing/HE/Output_Videos"

video_extensions = (".mp4", ".avi", ".mov", ".mkv")


def equalize_ycrcb(frame):
    """Enhance contrast using YCrCb color space (preserve colors)."""
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y = cv2.equalizeHist(y)
    ycrcb_eq = cv2.merge([y, cr, cb])
    return cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)


def equalize_hls(frame):
    """Enhance contrast using HLS color space (preserve colors)."""
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    h, l, s = cv2.split(hls)
    l = cv2.equalizeHist(l)
    hls_eq = cv2.merge([h, l, s])
    return cv2.cvtColor(hls_eq, cv2.COLOR_HLS2BGR)


def equalize_yuv(frame):
    """Enhance contrast using YUV color space (preserve colors)."""
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(yuv)
    y = cv2.equalizeHist(y)
    yuv_eq = cv2.merge([y, u, v])
    return cv2.cvtColor(yuv_eq, cv2.COLOR_YUV2BGR)


def equalize_hsv(frame):
    """Equalize only V channel in HSV to preserve colors."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.equalizeHist(v)
    hsv_eq = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)


def equalize_lab(frame):
    """Equalize only L channel in LAB to preserve colors."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    lab_eq = cv2.merge([l, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def process_video(input_path, output_folder_base):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Failed to open video: {input_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Prepare VideoWriters for each color space and original
    writers = {}
    spaces = [
        ("Original", lambda x: x),
        ("HLS", equalize_hls),
        ("YCrCb", equalize_ycrcb),
        ("YUV", equalize_yuv),
        ("HSV", equalize_hsv),
        ("LAB", equalize_lab),
    ]

    for space, func in spaces:
        out_path = os.path.join(output_folder_base, f"{space}.mp4")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        writers[space] = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply equalization for each space
        for space, func in spaces:
            eq_frame = func(frame)
            writers[space].write(eq_frame)

    cap.release()
    for w in writers.values():
        w.release()
    print(f"Processed video: {input_path}")


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


# Run processing
process_folder(input_root, output_root)
