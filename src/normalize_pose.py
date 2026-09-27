import numpy as np
from pathlib import Path


# ============================================================
# PATH
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    PROJECT_DIR
    / "output"
    / "features"
    / "v_PushUps_g01_c01.npy"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "output"
    / "normalized_features"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "v_PushUps_g01_c01_normalized.npy"
)


# ============================================================
# CHECK INPUT
# ============================================================

if not INPUT_PATH.exists():

    raise FileNotFoundError(
        f"Không tìm thấy file:\n{INPUT_PATH}"
    )


# ============================================================
# LOAD FEATURES
# ============================================================

features = np.load(
    INPUT_PATH
)


print("================================")
print("INPUT INFORMATION")
print("================================")

print(
    f"Shape: {features.shape}"
)

print()


# ============================================================
# CHECK DIMENSION
# ============================================================

if features.ndim != 2:

    raise ValueError(
        "Dữ liệu phải có dạng (frames, features)."
    )


if features.shape[1] != 99:

    raise ValueError(
        f"Feature dimension phải là 99, "
        f"nhưng nhận được {features.shape[1]}."
    )


# ============================================================
# RESHAPE
# ============================================================

# Hiện tại:
#
# (T, 99)
#
# Chuyển thành:
#
# (T, 33, 3)
#
# T = số frame
# 33 = số landmark
# 3 = x, y, z

T = features.shape[0]

pose = features.reshape(
    T,
    33,
    3
)


# ============================================================
# LANDMARK INDEX
# ============================================================

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

LEFT_HIP = 23
RIGHT_HIP = 24


# ============================================================
# NORMALIZATION
# ============================================================

normalized_pose = np.zeros_like(
    pose,
    dtype=np.float32
)


valid_frames = 0


for t in range(T):

    frame = pose[t].copy()


    # --------------------------------------------------------
    # Get hips
    # --------------------------------------------------------

    left_hip = frame[
        LEFT_HIP
    ]

    right_hip = frame[
        RIGHT_HIP
    ]


    # --------------------------------------------------------
    # Body center
    # --------------------------------------------------------

    body_center = (
        left_hip + right_hip
    ) / 2.0


    # --------------------------------------------------------
    # Center the body
    # --------------------------------------------------------

    centered = (
        frame - body_center
    )


    # --------------------------------------------------------
    # Get shoulders
    # --------------------------------------------------------

    left_shoulder = centered[
        LEFT_SHOULDER
    ]

    right_shoulder = centered[
        RIGHT_SHOULDER
    ]


    # --------------------------------------------------------
    # Body scale
    # --------------------------------------------------------

    shoulder_distance = np.linalg.norm(
        left_shoulder - right_shoulder
    )


    # --------------------------------------------------------
    # Avoid division by zero
    # --------------------------------------------------------

    if shoulder_distance < 1e-6:

        # Nếu frame không hợp lệ,
        # giữ nguyên frame centered.

        normalized_pose[t] = centered

        continue


    # --------------------------------------------------------
    # Scale normalization
    # --------------------------------------------------------

    normalized = (
        centered / shoulder_distance
    )


    normalized_pose[t] = normalized

    valid_frames += 1


# ============================================================
# FLATTEN
# ============================================================

normalized_features = normalized_pose.reshape(
    T,
    99
)


# ============================================================
# SAVE
# ============================================================

np.save(
    OUTPUT_PATH,
    normalized_features
)


# ============================================================
# INFORMATION
# ============================================================

print("================================")
print("NORMALIZATION DONE")
print("================================")

print(
    f"Total frames       : {T}"
)

print(
    f"Valid normalized   : {valid_frames}"
)

print(
    f"Feature dimension  : "
    f"{normalized_features.shape[1]}"
)

print(
    f"Output shape       : "
    f"{normalized_features.shape}"
)

print(
    f"Output             : "
    f"{OUTPUT_PATH}"
)