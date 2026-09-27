import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path


# ============================================================
# PATH
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

VIDEO_PATH = (
    PROJECT_DIR
    / "dataset"
    / "PushUps"
    / "v_PushUps_g01_c01.avi"
)

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "pose_landmarker.task"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "output"
    / "features"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "v_PushUps_g01_c01.npy"
)


# ============================================================
# CHECK FILE
# ============================================================

if not VIDEO_PATH.exists():
    raise FileNotFoundError(
        f"Không tìm thấy video:\n{VIDEO_PATH}"
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Không tìm thấy model:\n{MODEL_PATH}"
    )


# ============================================================
# MEDIA PIPE
# ============================================================

BaseOptions = mp.tasks.BaseOptions

PoseLandmarker = (
    mp.tasks.vision.PoseLandmarker
)

PoseLandmarkerOptions = (
    mp.tasks.vision.PoseLandmarkerOptions
)

VisionRunningMode = (
    mp.tasks.vision.RunningMode
)


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(
    str(VIDEO_PATH)
)

if not cap.isOpened():
    raise RuntimeError(
        f"Không thể mở video:\n{VIDEO_PATH}"
    )


fps = cap.get(
    cv2.CAP_PROP_FPS
)

if fps <= 0:
    fps = 30.0


frame_count = int(
    cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
)


print("================================")
print("VIDEO INFORMATION")
print("================================")

print(f"FPS    : {fps}")
print(f"Frames : {frame_count}")
print()


# ============================================================
# POSE OPTIONS
# ============================================================

options = PoseLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=str(
            MODEL_PATH
        )
    ),

    running_mode=VisionRunningMode.VIDEO,

    num_poses=1,

    min_pose_detection_confidence=0.5,

    min_pose_presence_confidence=0.5,

    min_tracking_confidence=0.5
)


# ============================================================
# EXTRACT FEATURES
# ============================================================

all_features = []

frame_index = 0

valid_frames = 0


print("================================")
print("EXTRACTING POSE FEATURES")
print("================================")
print()


with PoseLandmarker.create_from_options(
    options
) as landmarker:

    while True:

        success, frame = cap.read()

        if not success:
            break


        # ----------------------------------------------------
        # BGR → RGB
        # ----------------------------------------------------

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        # ----------------------------------------------------
        # MediaPipe Image
        # ----------------------------------------------------

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )


        # ----------------------------------------------------
        # Timestamp
        # ----------------------------------------------------

        timestamp_ms = int(
            frame_index * 1000 / fps
        )


        # ----------------------------------------------------
        # Pose Detection
        # ----------------------------------------------------

        result = (
            landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )
        )


        # ====================================================
        # GET LANDMARKS
        # ====================================================

        if result.pose_landmarks:

            # Người đầu tiên
            landmarks = result.pose_landmarks[0]


            # ------------------------------------------------
            # Extract x, y, z
            # ------------------------------------------------

            feature_vector = []

            for landmark in landmarks:

                feature_vector.extend([
                    landmark.x,
                    landmark.y,
                    landmark.z
                ])


            # ------------------------------------------------
            # Convert to numpy
            # ------------------------------------------------

            feature_vector = np.array(
                feature_vector,
                dtype=np.float32
            )


            # ------------------------------------------------
            # Check dimension
            # ------------------------------------------------

            if feature_vector.shape[0] == 99:

                all_features.append(
                    feature_vector
                )

                valid_frames += 1


        frame_index += 1


# ============================================================
# RELEASE
# ============================================================

cap.release()


# ============================================================
# CONVERT TO ARRAY
# ============================================================

if len(all_features) == 0:

    raise RuntimeError(
        "Không phát hiện được pose trong video."
    )


features = np.array(
    all_features,
    dtype=np.float32
)


# ============================================================
# SAVE
# ============================================================

np.save(
    OUTPUT_PATH,
    features
)


# ============================================================
# INFORMATION
# ============================================================

print()

print("================================")
print("DONE")
print("================================")

print(
    f"Total frames       : {frame_index}"
)

print(
    f"Valid pose frames  : {valid_frames}"
)

print(
    f"Feature dimension  : {features.shape[1]}"
)

print(
    f"Time series shape  : {features.shape}"
)

print(
    f"Output             : {OUTPUT_PATH}"
)