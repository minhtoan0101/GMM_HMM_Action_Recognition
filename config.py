import os


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ============================================================
# MODEL PATH
# ============================================================

MODEL_DIR = os.path.join(
    PROJECT_DIR,
    "output",
    "models"
)

POSE_MODEL = os.path.join(
    PROJECT_DIR,
    "models",
    "pose_landmarker.task"
)


# ============================================================
# ACTION CLASSES
# ============================================================

ACTIONS = [
    "BodyWeightSquats",
    "JumpingJack",
    "PullUps",
    "Punch",
    "PushUps"
]


# ============================================================
# MEDIAPIPE LANDMARK INDEX
# ============================================================

LEFT_HIP = 23
RIGHT_HIP = 24

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12