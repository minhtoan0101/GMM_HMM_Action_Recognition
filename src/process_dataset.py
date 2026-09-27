import cv2
import mediapipe as mp
import numpy as np

from pathlib import Path


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_DIR / "dataset"

MODEL_PATH = PROJECT_DIR / "models" / "pose_landmarker.task"

OUTPUT_DIR = PROJECT_DIR / "output" / "dataset_features"


# ============================================================
# SETTINGS
# ============================================================

CLASSES = [
    "BodyWeightSquats",
    "JumpingJack",
    "PullUps",
    "Punch",
    "PushUps"
]

# True  = chỉ xử lý một số video để TEST
# False = xử lý toàn bộ dataset
TEST_MODE = False

# Số video test mỗi class
TEST_VIDEOS_PER_CLASS = 5


# ============================================================
# MEDIAPIPE
# ============================================================

BaseOptions = mp.tasks.BaseOptions

PoseLandmarker = mp.tasks.vision.PoseLandmarker

PoseLandmarkerOptions = (
    mp.tasks.vision.PoseLandmarkerOptions
)

VisionRunningMode = mp.tasks.vision.RunningMode


# ============================================================
# NORMALIZATION
# ============================================================

# MediaPipe Pose landmark indices

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

LEFT_HIP = 23
RIGHT_HIP = 24


def normalize_pose(features):
    """
    Input:
        features shape = (T, 99)

    Output:
        normalized features shape = (T, 99)

    99 features:
        33 landmarks × 3 coordinates (x, y, z)
    """

    if len(features) == 0:

        return np.empty(
            (0, 99),
            dtype=np.float32
        )

    # --------------------------------------------------------
    # (T, 99) -> (T, 33, 3)
    # --------------------------------------------------------

    pose = features.reshape(
        -1,
        33,
        3
    )

    normalized = []

    for frame in pose:

        # ----------------------------------------------------
        # 1. Body center
        # ----------------------------------------------------

        left_hip = frame[LEFT_HIP]

        right_hip = frame[RIGHT_HIP]

        body_center = (
            left_hip + right_hip
        ) / 2.0

        # ----------------------------------------------------
        # 2. Center body
        # ----------------------------------------------------

        centered = (
            frame - body_center
        )

        # ----------------------------------------------------
        # 3. Shoulder distance
        # ----------------------------------------------------

        left_shoulder = (
            frame[LEFT_SHOULDER]
        )

        right_shoulder = (
            frame[RIGHT_SHOULDER]
        )

        shoulder_distance = np.linalg.norm(
            left_shoulder - right_shoulder
        )

        # ----------------------------------------------------
        # 4. Normalize scale
        # ----------------------------------------------------

        if shoulder_distance > 1e-6:

            normalized_frame = (
                centered / shoulder_distance
            )

        else:

            normalized_frame = centered

        normalized.append(
            normalized_frame
        )

    # --------------------------------------------------------
    # Convert to numpy
    # --------------------------------------------------------

    normalized = np.array(
        normalized,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # (T, 33, 3) -> (T, 99)
    # --------------------------------------------------------

    normalized = normalized.reshape(
        -1,
        99
    )

    return normalized


# ============================================================
# PROCESS ONE VIDEO
# ============================================================

def process_video(
    video_path,
    output_path,
    options
):

    print(
        f"[PROCESSING] {video_path.name}"
    )

    # --------------------------------------------------------
    # Open video
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        print(
            f"[ERROR] Cannot open: "
            f"{video_path.name}"
        )

        return False

    # --------------------------------------------------------
    # FPS
    # --------------------------------------------------------

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:

        fps = 30.0

    # --------------------------------------------------------
    # Store features
    # --------------------------------------------------------

    all_features = []

    frame_index = 0

    previous_timestamp_ms = -1

    # ========================================================
    # IMPORTANT
    #
    # Create a NEW PoseLandmarker for EACH VIDEO.
    #
    # This prevents MediaPipe timestamp errors between
    # different videos.
    # ========================================================

    try:

        with PoseLandmarker.create_from_options(
            options
        ) as landmarker:

            while True:

                # ------------------------------------------------
                # Read frame
                # ------------------------------------------------

                ret, frame = cap.read()

                if not ret:

                    break

                # ------------------------------------------------
                # BGR -> RGB
                # ------------------------------------------------

                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                # ------------------------------------------------
                # Create MediaPipe Image
                # ------------------------------------------------

                mp_image = mp.Image(
                    image_format=(
                        mp.ImageFormat.SRGB
                    ),
                    data=rgb_frame
                )

                # ------------------------------------------------
                # Timestamp
                # ------------------------------------------------

                timestamp_ms = int(
                    frame_index * 1000 / fps
                )

                # ------------------------------------------------
                # Make sure timestamp is strictly increasing
                # ------------------------------------------------

                if (
                    timestamp_ms
                    <= previous_timestamp_ms
                ):

                    timestamp_ms = (
                        previous_timestamp_ms
                        + 1
                    )

                previous_timestamp_ms = (
                    timestamp_ms
                )

                # ------------------------------------------------
                # Pose detection
                # ------------------------------------------------

                result = (
                    landmarker.detect_for_video(
                        mp_image,
                        timestamp_ms
                    )
                )

                # ------------------------------------------------
                # Get first detected person
                # ------------------------------------------------

                if result.pose_landmarks:

                    landmarks = (
                        result.pose_landmarks[0]
                    )

                    feature_vector = []

                    # ------------------------------------------------
                    # 33 landmarks × 3 coordinates
                    # ------------------------------------------------

                    for landmark in landmarks:

                        feature_vector.extend(
                            [
                                landmark.x,
                                landmark.y,
                                landmark.z
                            ]
                        )

                    # ------------------------------------------------
                    # Store frame feature
                    # ------------------------------------------------

                    all_features.append(
                        feature_vector
                    )

                frame_index += 1

    except Exception as e:

        print(
            f"[ERROR] MediaPipe failed: "
            f"{video_path.name}"
        )

        print(
            f"       {e}"
        )

        cap.release()

        return False

    finally:

        cap.release()

    # ========================================================
    # CHECK POSE
    # ========================================================

    if len(all_features) == 0:

        print(
            f"[WARNING] No pose found: "
            f"{video_path.name}"
        )

        return False

    # ========================================================
    # CONVERT TO NUMPY
    # ========================================================

    features = np.array(
        all_features,
        dtype=np.float32
    )

    # Expected:
    #
    # (T, 99)
    #
    # T = number of frames with detected pose

    # ========================================================
    # NORMALIZE
    # ========================================================

    normalized_features = (
        normalize_pose(
            features
        )
    )

    # ========================================================
    # SAVE
    # ========================================================

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(
        output_path,
        normalized_features
    )

    # ========================================================
    # RESULT
    # ========================================================

    print(
        f"[OK] {video_path.name:40s} "
        f"-> {normalized_features.shape}"
    )

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    print()

    print(
        "=============================================="
    )

    print(
        "      DATASET POSE FEATURE EXTRACTION"
    )

    print(
        "=============================================="
    )

    print()

    print(
        f"Dataset : {DATASET_DIR}"
    )

    print(
        f"Model   : {MODEL_PATH}"
    )

    print(
        f"Output  : {OUTPUT_DIR}"
    )

    print()

    # ========================================================
    # CHECK PATHS
    # ========================================================

    if not DATASET_DIR.exists():

        print(
            f"[ERROR] Dataset directory not found:"
        )

        print(
            DATASET_DIR
        )

        return

    if not MODEL_PATH.exists():

        print(
            f"[ERROR] MediaPipe model not found:"
        )

        print(
            MODEL_PATH
        )

        return

    # ========================================================
    # MODE
    # ========================================================

    if TEST_MODE:

        print(
            "TEST MODE: "
            f"{TEST_VIDEOS_PER_CLASS} videos/class"
        )

    else:

        print(
            "FULL DATASET MODE"
        )

    print()

    # ========================================================
    # MEDIAPIPE CONFIGURATION
    # ========================================================

    options = PoseLandmarkerOptions(

        base_options=BaseOptions(
            model_asset_path=str(
                MODEL_PATH
            )
        ),

        running_mode=(
            VisionRunningMode.VIDEO
        ),

        num_poses=1
    )

    # ========================================================
    # COUNTERS
    # ========================================================

    total_processed = 0

    total_success = 0

    total_failed = 0

    # ========================================================
    # PROCESS EACH CLASS
    # ========================================================

    for class_name in CLASSES:

        class_dir = (
            DATASET_DIR / class_name
        )

        output_class_dir = (
            OUTPUT_DIR / class_name
        )

        output_class_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # ----------------------------------------------------
        # Check class directory
        # ----------------------------------------------------

        if not class_dir.exists():

            print(
                f"[ERROR] Class directory "
                f"not found: {class_dir}"
            )

            continue

        # ----------------------------------------------------
        # Get videos
        # ----------------------------------------------------

        videos = sorted(
            class_dir.glob("*.avi")
        )

        # ----------------------------------------------------
        # Test mode
        # ----------------------------------------------------

        if TEST_MODE:

            videos = videos[
                :TEST_VIDEOS_PER_CLASS
            ]

        # ----------------------------------------------------
        # Class information
        # ----------------------------------------------------

        print()

        print(
            "----------------------------------------------"
        )

        print(
            f"CLASS: {class_name}"
        )

        print(
            f"Videos: {len(videos)}"
        )

        print(
            "----------------------------------------------"
        )

        # ====================================================
        # PROCESS VIDEOS
        # ====================================================

        for video_path in videos:

            output_path = (
                output_class_dir
                / f"{video_path.stem}.npy"
            )

            # ------------------------------------------------
            # Skip existing output
            # ------------------------------------------------

            if output_path.exists():

                print(
                    f"[SKIP] {video_path.name}"
                )

                total_processed += 1

                total_success += 1

                continue

            # ------------------------------------------------
            # Process
            # ------------------------------------------------

            success = process_video(
                video_path,
                output_path,
                options
            )

            total_processed += 1

            if success:

                total_success += 1

            else:

                total_failed += 1

    # ========================================================
    # SUMMARY
    # ========================================================

    print()

    print(
        "=============================================="
    )

    print(
        "              PROCESSING DONE"
    )

    print(
        "=============================================="
    )

    print(
        f"Processed : {total_processed}"
    )

    print(
        f"Success   : {total_success}"
    )

    print(
        f"Failed    : {total_failed}"
    )

    print(
        f"Output    : {OUTPUT_DIR}"
    )

    print(
        "=============================================="
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()