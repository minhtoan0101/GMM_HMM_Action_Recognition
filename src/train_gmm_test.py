from pathlib import Path
import json

import joblib
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(r"D:\GMM_HMM_Action_Recognition")

DATASET_DIR = (
    PROJECT_DIR
    / "output"
    / "dataset_features"
)

SPLIT_FILE = (
    PROJECT_DIR
    / "output"
    / "dataset_split.json"
)

MODEL_DIR = (
    PROJECT_DIR
    / "output"
    / "models"
)

GMM_FILE = MODEL_DIR / "gmm_model.pkl"
SCALER_FILE = MODEL_DIR / "gmm_scaler.pkl"


# ============================================================
# GMM CONFIGURATION
# ============================================================

N_COMPONENTS = 8

COVARIANCE_TYPE = "diag"

RANDOM_STATE = 42

MAX_ITER = 300


# ============================================================
# LOAD TRAINING DATA
# ============================================================

def load_training_data():

    with open(SPLIT_FILE, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    train_data = split_data["train"]

    all_frames = []

    print("=" * 60)
    print("              LOADING TRAINING DATA")
    print("=" * 60)

    total_videos = 0
    total_frames = 0

    for class_name, filenames in train_data.items():

        print(f"\n[{class_name}]")

        for filename in filenames:

            file_path = DATASET_DIR / class_name / filename

            if not file_path.exists():
                print(f"  [ERROR] Missing: {file_path}")
                continue

            data = np.load(file_path)

            print(
                f"  {filename:<45} "
                f"shape={data.shape}"
            )

            if data.ndim != 2:
                print("      [WARNING] Invalid dimensions")
                continue

            if data.shape[1] != 99:
                print(
                    f"      [WARNING] Expected 99 features, "
                    f"got {data.shape[1]}"
                )
                continue

            all_frames.append(data)

            total_videos += 1
            total_frames += data.shape[0]

    if not all_frames:
        raise RuntimeError(
            "No valid training data found."
        )

    X = np.vstack(all_frames)

    print("\n" + "=" * 60)
    print("             TRAINING DATA SUMMARY")
    print("=" * 60)

    print(f"Training videos : {total_videos}")
    print(f"Training frames : {total_frames}")
    print(f"Feature dimension: {X.shape[1]}")
    print(f"Final X shape   : {X.shape}")

    return X


# ============================================================
# TRAIN GMM
# ============================================================

def train_gmm(X):

    print("\n" + "=" * 60)
    print("                 TRAINING GMM")
    print("=" * 60)

    print(f"Components      : {N_COMPONENTS}")
    print(f"Covariance type : {COVARIANCE_TYPE}")
    print(f"Max iterations  : {MAX_ITER}")
    print(f"Random state    : {RANDOM_STATE}")

    # --------------------------------------------------------
    # Standardization
    # --------------------------------------------------------

    print("\n[1/2] Standardizing features...")

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    print(
        f"Scaled data shape: {X_scaled.shape}"
    )

    # --------------------------------------------------------
    # GMM
    # --------------------------------------------------------

    print("\n[2/2] Fitting Gaussian Mixture Model...")

    gmm = GaussianMixture(
        n_components=N_COMPONENTS,
        covariance_type=COVARIANCE_TYPE,
        random_state=RANDOM_STATE,
        max_iter=MAX_ITER,
        reg_covar=1e-5
    )

    gmm.fit(X_scaled)

    print("\nGMM training completed.")

    print(f"Converged : {gmm.converged_}")
    print(f"Iterations: {gmm.n_iter_}")
    print(f"BIC       : {gmm.bic(X_scaled):.4f}")
    print(f"AIC       : {gmm.aic(X_scaled):.4f}")

    return scaler, gmm


# ============================================================
# SAVE MODELS
# ============================================================

def save_models(scaler, gmm):

    MODEL_DIR.mkdir(
        parents=True,
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

    print("\n" + "=" * 60)
    print("                 MODELS SAVED")
    print("=" * 60)

    print(f"Scaler : {SCALER_FILE}")
    print(f"GMM    : {GMM_FILE}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("             GMM TRAINING PIPELINE")
    print("=" * 60)

    X = load_training_data()

    scaler, gmm = train_gmm(X)

    save_models(
        scaler,
        gmm
    )

    print("\n" + "=" * 60)
    print("                 GMM TRAINING DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()