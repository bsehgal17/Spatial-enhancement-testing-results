from enlighten_inference import EnlightenOnnxModel
import cv2
import os
import time


# -------------------- single video processing --------------------
def process_video_enlightgan(input_video_path, output_video_path, model):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {frame_count} frames with EnlightenGAN: {input_video_path}")

    start_time = time.time()
    processed = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        enhanced_frame = model.predict(frame)
        out.write(enhanced_frame)
        processed += 1
        if processed % 50 == 0:
            elapsed = time.time() - start_time
            print(
                f"Processed {processed}/{frame_count} frames, FPS: {processed / elapsed:.2f}"
            )

    cap.release()
    out.release()
    total_time = time.time() - start_time
    print(
        f"Finished {input_video_path} → {output_video_path} in {total_time:.2f}s, Avg FPS: {processed / total_time:.2f}"
    )


# -------------------- recursive folder processing --------------------
def enhance_all_videos_enlightgan(input_root, output_root, use_cuda=True):
    video_exts = (".mp4", ".avi", ".mov", ".mkv")

    # Load EnlightenGAN ONNX model once
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if use_cuda
        else ["CPUExecutionProvider"]
    )
    model = EnlightenOnnxModel(providers=providers)
    print("EnlightenGAN model loaded successfully.")

    for root, dirs, files in os.walk(input_root):
        for file in files:
            if file.lower().endswith(video_exts):
                input_video = os.path.join(root, file)
                rel_path = os.path.relpath(root, input_root)
                output_dir = os.path.join(output_root, rel_path)
                os.makedirs(output_dir, exist_ok=True)
                base = os.path.splitext(file)[0]
                output_video = os.path.join(output_dir, f"{base}_enhanced.mp4")

                process_video_enlightgan(input_video, output_video, model)


# -------------------- main --------------------
if __name__ == "__main__":
    input_root = (
        r"/storage/Projects/Gaitly/bsehgal/lower_body_pose_est/HumanEva/S3/Image_Data/"
    )
    output_root = (
        r"/storage/Projects/Gaitly/bsehgal/Spatial_testing/EnlightGAN/Output_Videos"
    )
    enhance_all_videos_enlightgan(input_root, output_root, use_cuda=True)
