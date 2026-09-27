import os
import json
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

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

def load_models():

    print("=" * 60)
    print("              LOADING MODELS")
    print("=" * 60)

    scaler = joblib.load(
        SCALER_FILE
    )

    gmm = joblib.load(
        GMM_FILE
    )

    hmm_models = {}

    for class_name in CLASSES:

        path = os.path.join(
            MODEL_DIR,
            f"hmm_{class_name}.pkl"
        )

        hmm_models[class_name] = joblib.load(
            path
        )

    print("Scaler : OK")
    print("GMM    : OK")

    for class_name in CLASSES:
        print(
            f"HMM {class_name:<20}: OK"
        )

    return scaler, gmm, hmm_models


# ============================================================
# LOAD TEST SPLIT
# ============================================================

def load_test_data():

    with open(
        SPLIT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        split_data = json.load(f)

    return split_data["test"]


# ============================================================
# CONVERT VIDEO TO GMM STATES
# ============================================================

def features_to_states(
    features,
    scaler,
    gmm
):

    # Standardize
    features_scaled = scaler.transform(
        features
    )

    # GMM state
    states = gmm.predict(
        features_scaled
    )

    # CategoricalHMM requires:
    # shape = (T, 1)
    states = states.reshape(
        -1,
        1
    )

    return states


# ============================================================
# PREDICT ONE VIDEO
# ============================================================

def predict_video(
    file_path,
    scaler,
    gmm,
    hmm_models
):

    features = np.load(
        file_path
    )

    if features.ndim != 2:
        raise ValueError(
            f"Invalid feature shape: {features.shape}"
        )

    if features.shape[1] != 99:
        raise ValueError(
            f"Expected 99 features, "
            f"got {features.shape[1]}"
        )

    if len(features) == 0:
        raise ValueError(
            "Empty feature sequence"
        )

    states = features_to_states(
        features,
        scaler,
        gmm
    )

    scores = {}

    for class_name in CLASSES:

        model = hmm_models[class_name]

        score = model.score(
            states
        )

        scores[class_name] = score

    predicted_class = max(
        scores,
        key=scores.get
    )

    return predicted_class, scores


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    print()
    print("=" * 60)
    print("        GMM + HMM ACTION RECOGNITION")
    print("                  EVALUATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    scaler, gmm, hmm_models = load_models()

    # --------------------------------------------------------
    # Load test data
    # --------------------------------------------------------

    test_data = load_test_data()

    print()
    print("=" * 60)
    print("                 TEST DATA")
    print("=" * 60)

    total_test = 0

    for class_name in CLASSES:

        count = len(
            test_data.get(
                class_name,
                []
            )
        )

        print(
            f"{class_name:<20}: {count}"
        )

        total_test += count

    print()
    print(
        f"Total test videos: {total_test}"
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    y_true = []
    y_pred = []

    results = []

    print()
    print("=" * 60)
    print("                 PREDICTIONS")
    print("=" * 60)

    processed = 0
    failed = 0

    for true_class in CLASSES:

        files = test_data.get(
            true_class,
            []
        )

        for filename in files:

            file_path = os.path.join(
                FEATURE_DIR,
                true_class,
                filename
            )

            try:

                predicted_class, scores = predict_video(
                    file_path,
                    scaler,
                    gmm,
                    hmm_models
                )

                y_true.append(
                    true_class
                )

                y_pred.append(
                    predicted_class
                )

                processed += 1

                correct = (
                    true_class == predicted_class
                )

                symbol = "OK" if correct else "WRONG"

                print(
                    f"[{symbol:<5}] "
                    f"{filename:<45} "
                    f"True={true_class:<20} "
                    f"Pred={predicted_class}"
                )

                results.append({
                    "file": filename,
                    "true": true_class,
                    "predicted": predicted_class,
                    "scores": scores
                })

            except Exception as e:

                failed += 1

                print(
                    f"[ERROR] "
                    f"{filename}: {e}"
                )

    # --------------------------------------------------------
    # Check results
    # --------------------------------------------------------

    if len(y_true) == 0:

        raise RuntimeError(
            "Không có video nào được đánh giá."
        )

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    report = classification_report(
        y_true,
        y_pred,
        labels=CLASSES,
        target_names=CLASSES,
        digits=4,
        zero_division=0
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=CLASSES
    )

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("                 EVALUATION")
    print("=" * 60)

    print(
        f"Processed : {processed}"
    )

    print(
        f"Failed    : {failed}"
    )

    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Accuracy  : {accuracy * 100:.2f}%"
    )

    print()
    print("Classification Report")
    print("-" * 60)
    print(report)

    print()
    print("Confusion Matrix")
    print("-" * 60)

    print(
        "Rows = True class"
    )

    print(
        "Cols = Predicted class"
    )

    print()

    print(
        f"{'':20}",
        end=""
    )

    for class_name in CLASSES:

        print(
            f"{class_name[:12]:>14}",
            end=""
        )

    print()

    for i, class_name in enumerate(CLASSES):

        print(
            f"{class_name:<20}",
            end=""
        )

        for j in range(len(CLASSES)):

            print(
                f"{cm[i][j]:>14}",
                end=""
            )

        print()

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    output_file = os.path.join(
        BASE_DIR,
        "output",
        "evaluation_results.json"
    )

    evaluation = {
        "processed": processed,
        "failed": failed,
        "accuracy": float(accuracy),
        "classes": CLASSES,
        "confusion_matrix": cm.tolist(),
        "y_true": y_true,
        "y_pred": y_pred,
        "results": results
    }

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            evaluation,
            f,
            indent=4,
            ensure_ascii=False
        )

    print()
    print("=" * 60)
    print("          EVALUATION COMPLETE")
    print("=" * 60)

    print(
        f"Results saved to:"
    )

    print(
        output_file
    )

    print("=" * 60)


if __name__ == "__main__":
    main()