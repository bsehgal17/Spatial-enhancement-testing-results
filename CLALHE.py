import os
import cv2
import numpy as np
import math

# Input and output root folders
input_root = (
    r"/storage/Projects/Gaitly/bsehgal/lower_body_pose_est/HumanEva/S3/Image_Data/"
)
output_root = r"/storage/Projects/Gaitly/bsehgal/Spatial_testing/CLALHE/Output_Videos"
video_extensions = (".mp4", ".avi", ".mov", ".mkv")


# ----------------- CLALHE Utility Functions -----------------
def compute_histogram_peaks_valleys(gray):
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    NPeaks, NValleys, peaks, valleys = 0, 0, [], []

    for i in range(1, 255):
        if hist[i] > hist[i - 1] and hist[i] > hist[i + 1]:
            NPeaks += 1
            peaks.append(hist[i])
        elif hist[i] < hist[i - 1] and hist[i] < hist[i + 1]:
            NValleys += 1
            valleys.append(hist[i])
    return NPeaks, NValleys, peaks, valleys


def compute_parameters(peaks, valleys):
    N_param = sum(valleys) / (len(valleys) + 1e-6)
    NPeaks = len(peaks)
    CBD_dim = max(1, math.ceil(math.log(NPeaks + 1e-6)))  # avoid zero
    CBD = (CBD_dim, CBD_dim)
    I_CL = [(p + N_param) / (max(valleys) + 1e-6) for p in peaks]
    I_CL = min(I_CL)
    return N_param, CBD, I_CL


def calculate_ONS(NValleys):
    ONS = math.ceil(math.log2(NValleys + 1e-6))
    if ONS % 2 != 0:
        ONS += 1
    return ONS


def subdivide_image(img, ONS):
    H, W = img.shape
    rows = ONS // 2
    cols = ONS // 2
    SID_h = math.ceil(H / rows)
    SID_w = math.ceil(W / cols)
    subimages = []
    coords = []
    for r in range(rows):
        for c in range(cols):
            y1, y2 = r * SID_h, min((r + 1) * SID_h, H)
            x1, x2 = c * SID_w, min((c + 1) * SID_w, W)
            subimages.append(img[y1:y2, x1:x2])
            coords.append((y1, y2, x1, x2))
    return subimages, coords


def enhance_subimage_clahe(subimg, I_CL, CBD):
    clahe = cv2.createCLAHE(clipLimit=I_CL, tileGridSize=CBD)
    enhanced = clahe.apply(subimg)
    return enhanced


def clalhe(image):
    gray = (
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if len(image.shape) == 3
        else image.copy()
    )
    NPeaks, NValleys, peaks, valleys = compute_histogram_peaks_valleys(gray)
    N_param, CBD, I_CL = compute_parameters(peaks, valleys)
    ONS = calculate_ONS(NValleys)
    subimages, coords = subdivide_image(gray, ONS)
    enhanced_subimages = [enhance_subimage_clahe(sub, I_CL, CBD) for sub in subimages]
    enhanced_img = np.zeros_like(gray)
    for sub, (y1, y2, x1, x2) in zip(enhanced_subimages, coords):
        enhanced_img[y1:y2, x1:x2] = sub
    return enhanced_img


# ----------------- Video Processing -----------------
def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Failed to open: {input_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        enhanced_frame = clalhe(frame)
        enhanced_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_GRAY2BGR)
        writer.write(enhanced_frame)

    cap.release()
    writer.release()
    print(f"Processed: {input_path}")


def process_folder(input_folder, output_folder):
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(video_extensions):
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_folder)
                output_path = os.path.join(
                    output_folder,
                    relative_path,
                    f"{os.path.splitext(file)[0]}_CLALHE.mp4",
                )
                process_video(input_path, output_path)


# ----------------- Run -----------------
process_folder(input_root, output_root)
