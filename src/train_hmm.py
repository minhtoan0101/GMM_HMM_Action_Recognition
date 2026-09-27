import os
import json
import numpy as np
import joblib

from hmmlearn.hmm import CategoricalHMM


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

# ------------------------------------------------------------
# HMM CONFIG
# ------------------------------------------------------------

N_STATES = 8

RANDOM_STATE = 42

N_ITER = 100


CLASSES = [
    "BodyWeightSquats",
    "JumpingJack",
    "PullUps",
    "Punch",
    "PushUps"
]


# ============================================================
# LOAD MODELS
# ============================================================

def load_gmm_models():

    print("=" * 60)
    print("             LOADING GMM")
    print("=" * 60)

    scaler = joblib.load(
        SCALER_FILE
    )

    gmm = joblib.load(
        GMM_FILE
    )

    print("Scaler loaded : OK")
    print("GMM loaded    : OK")

    print(
        f"GMM components: {gmm.n_components}"
    )

    return scaler, gmm


# ============================================================
# LOAD SPLIT
# ============================================================

def load_split():

    with open(
        SPLIT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        split_data = json.load(f)

    return split_data


# ============================================================
# CONVERT VIDEO TO GMM STATES
# ============================================================

def video_to_states(
    file_path,
    scaler,
    gmm
):

    features = np.load(
        file_path
    )

    # --------------------------------------------------------
    # Check feature shape
    # --------------------------------------------------------

    if features.ndim != 2:
        raise ValueError(
            f"Invalid feature shape: {features.shape}"
        )

    if features.shape[1] != 99:
        raise ValueError(
            f"Expected 99 features, "
            f"got {features.shape[1]}"
        )

    # --------------------------------------------------------
    # Standardization
    # --------------------------------------------------------

    features_scaled = scaler.transform(
        features
    )

    # --------------------------------------------------------
    # GMM prediction
    # --------------------------------------------------------

    states = gmm.predict(
        features_scaled
    )

    # HMM expects:
    # shape = (T, 1)
    states = states.reshape(
        -1,
        1
    )

    return states


# ============================================================
# TRAIN ONE HMM
# ============================================================

def train_class_hmm(
    class_name,
    files,
    scaler,
    gmm
):

    print()
    print("=" * 60)
    print(f"TRAINING HMM: {class_name}")
    print("=" * 60)

    sequences = []

    total_frames = 0

    successful_videos = 0

    for index, filename in enumerate(files):

        file_path = os.path.join(
            FEATURE_DIR,
            class_name,
            filename
        )

        if not os.path.exists(file_path):

            print(
                f"[WARNING] Missing: {filename}"
            )

            continue

        try:

            states = video_to_states(
                file_path,
                scaler,
                gmm
            )

            if len(states) == 0:

                print(
                    f"[WARNING] Empty sequence: {filename}"
                )

                continue

            sequences.append(states)

            total_frames += len(states)

            successful_videos += 1

        except Exception as e:

            print(
                f"[ERROR] {filename}: {e}"
            )

    if len(sequences) == 0:

        raise RuntimeError(
            f"No valid sequences for {class_name}"
        )

    # --------------------------------------------------------
    # Concatenate sequences
    # --------------------------------------------------------

    X = np.concatenate(
        sequences,
        axis=0
    )

    lengths = [
        len(seq)
        for seq in sequences
    ]

    print()
    print(f"Videos      : {successful_videos}")
    print(f"Total frames: {total_frames}")
    print(f"X shape     : {X.shape}")

    print()
    print("Training CategoricalHMM...")

    # --------------------------------------------------------
    # Create HMM
    # --------------------------------------------------------

    model = CategoricalHMM(
        n_components=N_STATES,
        random_state=RANDOM_STATE,
        n_iter=N_ITER,
        verbose=False
    )

    # Number of possible observations
    # = number of GMM components
    model.n_features = gmm.n_components

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model.fit(
        X,
        lengths
    )

    print()
    print("HMM TRAINING COMPLETE")

    print(
        f"Converged : {model.monitor_.converged}"
    )

    print(
        f"Iterations: {model.monitor_.iter}"
    )

    print(
        f"Final log likelihood: "
        f"{model.monitor_.history[-1]:.6f}"
    )

    return model


# ============================================================
# SAVE HMM
# ============================================================

def save_hmm(
    class_name,
    model
):

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    filename = (
        f"hmm_{class_name}.pkl"
    )

    output_path = os.path.join(
        MODEL_DIR,
        filename
    )

    joblib.dump(
        model,
        output_path
    )

    print(
        f"Saved: {output_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("       HMM TRAINING - GMM + HMM")
    print("=" * 60)

    # --------------------------------------------------------
    # Load GMM
    # --------------------------------------------------------

    scaler, gmm = load_gmm_models()

    # --------------------------------------------------------
    # Load train/test split
    # --------------------------------------------------------

    split_data = load_split()

    train_data = split_data["train"]

    print()
    print("=" * 60)
    print("TRAIN DATA")
    print("=" * 60)

    total_train = 0

    for class_name in CLASSES:

        count = len(
            train_data.get(
                class_name,
                []
            )
        )

        print(
            f"{class_name:<20}: {count}"
        )

        total_train += count

    print(
        f"\nTotal train videos: {total_train}"
    )

    # --------------------------------------------------------
    # Train one HMM per class
    # --------------------------------------------------------

    trained_models = {}

    for class_name in CLASSES:

        files = train_data.get(
            class_name,
            []
        )

        if len(files) == 0:

            print(
                f"[WARNING] No training data "
                f"for {class_name}"
            )

            continue

        model = train_class_hmm(
            class_name,
            files,
            scaler,
            gmm
        )

        save_hmm(
            class_name,
            model
        )

        trained_models[class_name] = model

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("             HMM TRAINING DONE")
    print("=" * 60)

    print(
        f"Models trained: "
        f"{len(trained_models)}/{len(CLASSES)}"
    )

    print()
    print("Saved models:")

    for class_name in trained_models:

        print(
            f"  - hmm_{class_name}.pkl"
        )

    print()
    print(
        f"Model directory:"
    )

    print(
        MODEL_DIR
    )

    print("=" * 60)


if __name__ == "__main__":
    main()