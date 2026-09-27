import os
import sys
import cv2
import numpy as np
import joblib
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# CONFIG
# ============================================================

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(PROJECT_DIR, "output", "models")
POSE_MODEL = os.path.join(PROJECT_DIR, "models", "pose_landmarker.task")

ACTIONS = [
    "BodyWeightSquats",
    "JumpingJack",
    "PullUps",
    "Punch",
    "PushUps"
]

# Hip landmarks
LEFT_HIP = 23
RIGHT_HIP = 24

# Shoulder landmarks
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12


# ============================================================
# NORMALIZE POSE
# ============================================================

def normalize_pose(features):
    """
    Input:
        features: (T, 99)

    Output:
        normalized_features: (T, 99)

    Mỗi frame:
        33 landmarks x (x, y, z)
    """

    normalized = []

    for frame in features:

        landmarks = frame.reshape(33, 3).copy()

        # ----------------------------------------------------
        # 1. Center tại midpoint của 2 hông
        # ----------------------------------------------------

        left_hip = landmarks[LEFT_HIP]
        right_hip = landmarks[RIGHT_HIP]

        hip_center = (left_hip + right_hip) / 2.0

        landmarks = landmarks - hip_center

        # ----------------------------------------------------
        # 2. Scale theo khoảng cách 2 vai
        # ----------------------------------------------------

        left_shoulder = landmarks[LEFT_SHOULDER]
        right_shoulder = landmarks[RIGHT_SHOULDER]

        shoulder_distance = np.linalg.norm(
            left_shoulder - right_shoulder
        )

        # tránh chia cho 0
        if shoulder_distance < 1e-6:
            shoulder_distance = 1.0

        landmarks = landmarks / shoulder_distance

        normalized.append(landmarks.flatten())

    return np.array(normalized, dtype=np.float32)


# ============================================================
# EXTRACT POSE
# ============================================================

def extract_pose(video_path):

    print("\n[1/4] Extracting human pose...")

    BaseOptions = python.BaseOptions

    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=POSE_MODEL
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1
    )

    landmarker = vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video: {video_path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25.0

    frame_index = 0
    features = []

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # BGR -> RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp_ms = int(
            frame_index * 1000 / fps
        )

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        if result.pose_landmarks:

            landmarks = result.pose_landmarks[0]

            vector = []

            for landmark in landmarks:
                vector.extend([
                    landmark.x,
                    landmark.y,
                    landmark.z
                ])

            features.append(vector)

        frame_index += 1

    cap.release()
    landmarker.close()

    if len(features) == 0:
        raise RuntimeError(
            "Không phát hiện được người trong video."
        )

    features = np.array(
        features,
        dtype=np.float32
    )

    print(f"    Total video frames : {frame_index}")
    print(f"    Valid pose frames  : {len(features)}")
    print(f"    Feature dimension  : {features.shape[1]}")

    return features


# ============================================================
# LOAD MODELS
# ============================================================

def load_models():

    print("\n[2/4] Loading GMM + HMM models...")

    scaler_path = os.path.join(
        MODEL_DIR,
        "gmm_scaler.pkl"
    )

    gmm_path = os.path.join(
        MODEL_DIR,
        "gmm_model.pkl"
    )

    scaler = joblib.load(scaler_path)
    gmm = joblib.load(gmm_path)

    hmm_models = {}

    for action in ACTIONS:

        path = os.path.join(
            MODEL_DIR,
            f"hmm_{action}.pkl"
        )

        hmm_models[action] = joblib.load(path)

    print("    GMM loaded")
    print("    HMM models loaded:", len(hmm_models))

    return scaler, gmm, hmm_models


# ============================================================
# PREDICT
# ============================================================

def predict(features, scaler, gmm, hmm_models):

    print("\n[3/4] GMM + HMM prediction...")

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    normalized = normalize_pose(features)

    print(
        f"    Normalized shape: {normalized.shape}"
    )

    # --------------------------------------------------------
    # StandardScaler
    # --------------------------------------------------------

    scaled = scaler.transform(normalized)

    # --------------------------------------------------------
    # GMM
    # --------------------------------------------------------

    states = gmm.predict(scaled)

    print(
        f"    GMM states: {len(states)}"
    )

    print(
        f"    Unique states: {np.unique(states)}"
    )

    # HMM cần shape (T, 1)
    observation = states.reshape(-1, 1)

    # --------------------------------------------------------
    # HMM
    # --------------------------------------------------------

    scores = {}

    for action, hmm in hmm_models.items():

        try:
            score = hmm.score(observation)
        except Exception as e:
            print(
                f"    Warning: HMM {action} failed: {e}"
            )
            score = -np.inf

        scores[action] = score

    # --------------------------------------------------------
    # Sort scores
    # --------------------------------------------------------

    sorted_scores = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    predicted_action = sorted_scores[0][0]

    print("\n[4/4] RESULT")
    print("=" * 55)

    for action, score in sorted_scores:

        print(
            f"{action:<22} : {score:>12.4f}"
        )

    print("=" * 55)

    print(
        f"\nPredicted action: {predicted_action}"
    )

    return predicted_action, scores


# ============================================================
# MAIN
# ============================================================

def main():

    if len(sys.argv) < 2:

        print(
            "\nUsage:"
        )

        print(
            "python src\\predict_video.py <video_path>"
        )

        print(
            "\nExample:"
        )

        print(
            'python src\\predict_video.py "dataset\\PushUps\\v_PushUps_g01_c01.avi"'
        )

        return

    video_path = sys.argv[1]

    if not os.path.exists(video_path):

        print(
            f"\nERROR: Video not found:"
        )

        print(video_path)

        return

    print("=" * 55)
    print("GMM + HMM ACTION RECOGNITION")
    print("=" * 55)

    print(
        f"\nVideo: {video_path}"
    )

    # 1. Pose
    features = extract_pose(video_path)

    # 2. Models
    scaler, gmm, hmm_models = load_models()

    # 3. Predict
    predicted_action, scores = predict(
        features,
        scaler,
        gmm,
        hmm_models
    )

    print("\nDone.")


if __name__ == "__main__":
    main()