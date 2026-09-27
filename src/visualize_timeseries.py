import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================
# PATH
# ============================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    PROJECT_DIR
    / "output"
    / "normalized_features"
    / "v_PushUps_g01_c01_normalized.npy"
)

OUTPUT_DIR = PROJECT_DIR / "output" / "visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================
# LOAD DATA
# ============================================

features = np.load(INPUT_PATH)

print("================================")
print("TIME SERIES VISUALIZATION")
print("================================")

print(f"Shape: {features.shape}")


# ============================================
# CHECK
# ============================================

num_frames, num_features = features.shape

print(f"Number of frames   : {num_frames}")
print(f"Number of features : {num_features}")


# ============================================
# SELECT SOME FEATURES
# ============================================

# Mỗi landmark có 3 giá trị:
# x, y, z
#
# Landmark index:
# 0 -> x,y,z
# 1 -> x,y,z
# ...
#
# Chọn một số landmark để minh họa.

landmarks = {
    "Nose": 0,
    "Left Shoulder": 11,
    "Right Shoulder": 12,
    "Left Hip": 23,
    "Right Hip": 24,
}


# ============================================
# PLOT X COORDINATE
# ============================================

plt.figure(figsize=(12, 6))

for name, landmark_idx in landmarks.items():

    feature_idx = landmark_idx * 3

    plt.plot(
        range(num_frames),
        features[:, feature_idx],
        label=name
    )

plt.xlabel("Frame")
plt.ylabel("Normalized X coordinate")

plt.title(
    "Normalized Pose Features Over Time - X Coordinate"
)

plt.legend()
plt.grid(True)

output_x = OUTPUT_DIR / "timeseries_x.png"

plt.savefig(output_x, dpi=150, bbox_inches="tight")

plt.show()

print(f"Saved: {output_x}")


# ============================================
# PLOT Y COORDINATE
# ============================================

plt.figure(figsize=(12, 6))

for name, landmark_idx in landmarks.items():

    feature_idx = landmark_idx * 3 + 1

    plt.plot(
        range(num_frames),
        features[:, feature_idx],
        label=name
    )

plt.xlabel("Frame")
plt.ylabel("Normalized Y coordinate")

plt.title(
    "Normalized Pose Features Over Time - Y Coordinate"
)

plt.legend()
plt.grid(True)

output_y = OUTPUT_DIR / "timeseries_y.png"

plt.savefig(output_y, dpi=150, bbox_inches="tight")

plt.show()

print(f"Saved: {output_y}")


# ============================================
# PLOT Z COORDINATE
# ============================================

plt.figure(figsize=(12, 6))

for name, landmark_idx in landmarks.items():

    feature_idx = landmark_idx * 3 + 2

    plt.plot(
        range(num_frames),
        features[:, feature_idx],
        label=name
    )

plt.xlabel("Frame")
plt.ylabel("Normalized Z coordinate")

plt.title(
    "Normalized Pose Features Over Time - Z Coordinate"
)

plt.legend()
plt.grid(True)

output_z = OUTPUT_DIR / "timeseries_z.png"

plt.savefig(output_z, dpi=150, bbox_inches="tight")

plt.show()

print(f"Saved: {output_z}")


# ============================================
# DONE
# ============================================

print()
print("================================")
print("VISUALIZATION DONE")
print("================================")