import os
import json
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = r"D:\GMM_HMM_Action_Recognition"

SPLIT_FILE = os.path.join(
    BASE_DIR,
    "output",
    "dataset_split.json"
)

FEATURE_DIR = os.path.join(
    BASE_DIR,
    "output",
    "dataset_features"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "output",
    "models"
)

SCALER_FILE = os.path.join(
    MODEL_DIR,
    "gmm_scaler.pkl"
)

GMM_FILE = os.path.join(
    MODEL_DIR,
    "gmm_model.pkl"
)

# Số Gaussian components
N_COMPONENTS = 8

RANDOM_STATE = 42

MAX_ITER = 300

REG_COVAR = 1e-5


# ============================================================
# LOAD TRAIN FILES
# ============================================================

def load_training_data():

    print("=" * 60)
    print("          LOADING TRAINING DATA")
    print("=" * 60)

    with open(
        SPLIT_FILE,
        "r",
        encoding="utf-8"
    ) as f:
        split_data = json.load(f)

    train_data = split_data["train"]

    all_features = []

    total_videos = 0
    total_frames = 0

    for class_name, files in train_data.items():

        print()
        print(f"[{class_name}]")
        print(f"Videos: {len(files)}")

        class_dir = os.path.join(
            FEATURE_DIR,
            class_name
        )

        for filename in files:

            file_path = os.path.join(
                class_dir,
                filename
            )

            if not os.path.exists(file_path):
                print(
                    f"[WARNING] File not found: {file_path}"
                )
                continue

            features = np.load(file_path)

            # Kiểm tra shape
            if features.ndim != 2:
                print(
                    f"[WARNING] Invalid shape: "
                    f"{filename} -> {features.shape}"
                )
                continue

            if features.shape[1] != 99:
                print(
                    f"[WARNING] Invalid feature dimension: "
                    f"{filename} -> {features.shape}"
                )
                continue

            all_features.append(features)

            total_videos += 1
            total_frames += features.shape[0]

    if len(all_features) == 0:
        raise RuntimeError(
            "Không tìm thấy dữ liệu training!"
        )

    X = np.vstack(all_features)

    print()
    print("=" * 60)
    print("          TRAINING DATA SUMMARY")
    print("=" * 60)

    print(f"Videos loaded : {total_videos}")
    print(f"Total frames  : {total_frames}")
    print(f"Feature dim   : {X.shape[1]}")
    print(f"X shape       : {X.shape}")

    return X


# ============================================================
# TRAIN GMM
# ============================================================

def train_gmm(X):

    print()
    print("=" * 60)
    print("          STANDARDIZING FEATURES")
    print("=" * 60)

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    print(
        f"Scaled data shape: {X_scaled.shape}"
    )

    print()
    print("=" * 60)
    print("             TRAINING GMM")
    print("=" * 60)

    print(f"Components    : {N_COMPONENTS}")
    print(f"Covariance    : diag")
    print(f"Max iterations: {MAX_ITER}")
    print(f"Random state  : {RANDOM_STATE}")

    gmm = GaussianMixture(
        n_components=N_COMPONENTS,
        covariance_type="diag",
        random_state=RANDOM_STATE,
        max_iter=MAX_ITER,
        reg_covar=REG_COVAR
    )

    gmm.fit(X_scaled)

    print()
    print("GMM TRAINING COMPLETE")

    print(f"Converged     : {gmm.converged_}")
    print(f"Iterations    : {gmm.n_iter_}")
    print(f"Lower bound   : {gmm.lower_bound_:.6f}")
    print(f"AIC           : {gmm.aic(X_scaled):.2f}")
    print(f"BIC           : {gmm.bic(X_scaled):.2f}")

    return scaler, gmm


# ============================================================
# SAVE MODELS
# ============================================================

def save_models(scaler, gmm):

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    joblib.dump(
        scaler,
        SCALER_FILE
    )

    joblib.dump(
        gmm,
        GMM_FILE
    )

    print()
    print("=" * 60)
    print("             MODELS SAVED")
    print("=" * 60)

    print(
        f"Scaler : {SCALER_FILE}"
    )

    print(
        f"GMM    : {GMM_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("        GMM TRAINING - ACTION RECOGNITION")
    print("=" * 60)

    X = load_training_data()

    scaler, gmm = train_gmm(X)

    save_models(
        scaler,
        gmm
    )

    print()
    print("=" * 60)
    print("              GMM TRAINING DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()