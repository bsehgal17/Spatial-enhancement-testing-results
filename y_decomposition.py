import os
import cv2
import numpy as np

# ----------------- User Parameters -----------------
input_root = (
    r"/storage/Projects/Gaitly/bsehgal/lower_body_pose_est/HumanEva/S3/Image_Data/"
)
output_root = r"/storage/Projects/Gaitly/bsehgal/Spatial_testing/Sigmoid/Output_Videos"
video_extensions = (".mp4", ".avi", ".mov", ".mkv")


# ----------------- Sigmoid Enhancement Functions -----------------
def sigmoid_enhance(Yb, gamma=1.4, alpha=0.5):
    Yb_norm = Yb / 255.0
    Yb_gauss = cv2.GaussianBlur(Yb_norm, (3, 3), 1)

    O = np.zeros_like(Yb_norm)
    for i in range(Yb.shape[0]):
        for j in range(Yb.shape[1]):
            if Yb_norm[i, j] <= Yb_gauss[i, j]:
                O[i, j] = Yb_gauss[i, j] ** (1 - gamma) * Yb_norm[i, j] ** gamma
            else:
                O[i, j] = (
                    1
                    - (1 - Yb_gauss[i, j]) ** (1 - gamma) * (1 - Yb_norm[i, j]) ** gamma
                )

    w = ((O.max() - O.min()) / (Yb_norm.max() - Yb_norm.min() + 1e-6)) * alpha
    Yb_prime = np.clip(w * O * 255, 0, 255).astype(np.uint8)
    return Yb_prime


def enhance_image_YCbCr(img, gamma=1.4, alpha=0.5):
    img_YCbCr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(img_YCbCr)

    # Base and detail layers
    LPF = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16.0
    Yb = cv2.filter2D(Y, -1, LPF)
    Yd = Y - Yb

    # Apply sigmoid enhancement
    Yb_prime = sigmoid_enhance(Yb, gamma, alpha)

    # Enhanced detail layer
    Yd_prime = Yd * (Yb_prime / (Yb + 1e-6))
    Y_prime = np.clip(Yb_prime + Yd_prime, 0, 255).astype(np.uint8)

    # Merge back and convert to BGR
    enhanced_YCbCr = cv2.merge([Y_prime, Cr, Cb])
    enhanced_RGB = cv2.cvtColor(enhanced_YCbCr, cv2.COLOR_YCrCb2BGR)
    return enhanced_RGB


# ----------------- Video Processing -----------------
def process_video(input_path, output_path, gamma=1.4, alpha=0.5):
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
        enhanced_frame = enhance_image_YCbCr(frame, gamma, alpha)
        writer.write(enhanced_frame)

    cap.release()
    writer.release()
    print(f"Processed: {input_path}")


def process_folder(input_folder, output_folder, gamma=1.4, alpha=0.5):
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(video_extensions):
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_folder)
                output_path = os.path.join(
                    output_folder,
                    relative_path,
                    f"{os.path.splitext(file)[0]}_Sigmoid.mp4",
                )
                process_video(input_path, output_path, gamma, alpha)


# ----------------- Run -----------------
process_folder(input_root, output_root)
