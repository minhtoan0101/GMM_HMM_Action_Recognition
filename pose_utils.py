import cv2
import numpy as np
import mediapipe as mp

from config import (
    LEFT_HIP,
    RIGHT_HIP,
    LEFT_SHOULDER,
    RIGHT_SHOULDER
)


# ============================================================
# EXTRACT ONE POSE FRAME
# ============================================================

def extract_pose_from_frame(
    frame,
    landmarker,
    timestamp_ms
):

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )

    if not result.pose_landmarks:
        return None

    landmarks = result.pose_landmarks[0]

    vector = []

    for landmark in landmarks:

        vector.extend([
            landmark.x,
            landmark.y,
            landmark.z
        ])

    return np.array(
        vector,
        dtype=np.float32
    )


# ============================================================
# NORMALIZE POSE
# ============================================================

def normalize_pose(features):

    normalized = []

    for frame in features:

        landmarks = frame.reshape(
            33,
            3
        ).copy()

        # ----------------------------------------------------
        # Center at hip midpoint
        # ----------------------------------------------------

        hip_center = (
            landmarks[LEFT_HIP]
            +
            landmarks[RIGHT_HIP]
        ) / 2.0

        landmarks -= hip_center

        # ----------------------------------------------------
        # Scale using shoulder distance
        # ----------------------------------------------------

        shoulder_distance = np.linalg.norm(
            landmarks[LEFT_SHOULDER]
            -
            landmarks[RIGHT_SHOULDER]
        )

        if shoulder_distance < 1e-6:
            shoulder_distance = 1.0

        landmarks /= shoulder_distance

        normalized.append(
            landmarks.flatten()
        )

    return np.array(
        normalized,
        dtype=np.float32
    )


# ============================================================
# POSE CONNECTIONS
# ============================================================

POSE_CONNECTIONS = [

    (11, 12),

    (11, 13),
    (13, 15),

    (12, 14),
    (14, 16),

    (11, 23),
    (12, 24),

    (23, 24),

    (23, 25),
    (25, 27),

    (24, 26),
    (26, 28),

    (27, 31),
    (28, 32),

    (15, 17),
    (15, 19),
    (15, 21),

    (16, 18),
    (16, 20),
    (16, 22),

    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),

    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),

    (9, 10),

    (10, 12),
    (9, 11)
]


# ============================================================
# DRAW SKELETON
# ============================================================

def draw_skeleton(
    frame,
    landmarks
):

    height, width = frame.shape[:2]

    points = []

    for landmark in landmarks:

        x = int(
            landmark.x * width
        )

        y = int(
            landmark.y * height
        )

        points.append(
            (x, y)
        )

    # --------------------------------------------------------
    # Connections
    # --------------------------------------------------------

    for start, end in POSE_CONNECTIONS:

        if (
            start < len(points)
            and end < len(points)
        ):

            cv2.line(
                frame,
                points[start],
                points[end],
                (0, 255, 0),
                2
            )

    # --------------------------------------------------------
    # Landmarks
    # --------------------------------------------------------

    for x, y in points:

        cv2.circle(
            frame,
            (x, y),
            3,
            (0, 0, 255),
            -1
        )

    return frame