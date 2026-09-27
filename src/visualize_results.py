import os
import json

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = r"D:\GMM_HMM_Action_Recognition"

RESULT_FILE = os.path.join(
    BASE_DIR,
    "output",
    "evaluation_results.json"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "output",
    "visualizations"
)

CLASSES = [
    "BodyWeightSquats",
    "JumpingJack",
    "PullUps",
    "Punch",
    "PushUps"
]


# ============================================================
# LOAD RESULTS
# ============================================================

def load_results():

    with open(
        RESULT_FILE,
        "r",
        encoding="utf-8"
    ) as f:
        data = json.load(f)

    return data


# ============================================================
# CONFUSION MATRIX
# ============================================================

def plot_confusion_matrix(data):

    cm = np.array(
        data["confusion_matrix"]
    )

    fig, ax = plt.subplots(
        figsize=(9, 7)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASSES
    )

    display.plot(
        ax=ax,
        values_format="d",
        xticks_rotation=30
    )

    ax.set_title(
        "Confusion Matrix - GMM + HMM"
    )

    ax.set_xlabel(
        "Predicted Class"
    )

    ax.set_ylabel(
        "True Class"
    )

    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        "confusion_matrix.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


# ============================================================
# F1 SCORE
# ============================================================

def calculate_f1_scores(data):

    y_true = data["y_true"]
    y_pred = data["y_pred"]

    f1_scores = []

    for class_name in CLASSES:

        tp = 0
        fp = 0
        fn = 0

        for true, pred in zip(
            y_true,
            y_pred
        ):

            if true == class_name and pred == class_name:
                tp += 1

            elif true != class_name and pred == class_name:
                fp += 1

            elif true == class_name and pred != class_name:
                fn += 1

        if tp == 0:
            f1 = 0.0
        else:

            precision = (
                tp / (tp + fp)
                if (tp + fp) > 0
                else 0
            )

            recall = (
                tp / (tp + fn)
                if (tp + fn) > 0
                else 0
            )

            if precision + recall == 0:
                f1 = 0
            else:
                f1 = (
                    2 * precision * recall
                    / (precision + recall)
                )

        f1_scores.append(f1)

    return f1_scores


# ============================================================
# F1 BAR CHART
# ============================================================

def plot_f1_scores(data):

    f1_scores = calculate_f1_scores(
        data
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    bars = ax.bar(
        CLASSES,
        f1_scores
    )

    ax.set_title(
        "F1-score by Action Class"
    )

    ax.set_xlabel(
        "Action"
    )

    ax.set_ylabel(
        "F1-score"
    )

    ax.set_ylim(
        0,
        1
    )

    ax.tick_params(
        axis="x",
        rotation=25
    )

    # Hiển thị giá trị trên cột
    for bar, score in zip(
        bars,
        f1_scores
    ):

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            score + 0.02,
            f"{score:.3f}",
            ha="center"
        )

    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        "f1_scores.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("       VISUALIZING EVALUATION RESULTS")
    print("=" * 60)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    data = load_results()

    print(
        f"Accuracy: "
        f"{data['accuracy'] * 100:.2f}%"
    )

    print()

    plot_confusion_matrix(
        data
    )

    plot_f1_scores(
        data
    )

    print()
    print("=" * 60)
    print("          VISUALIZATION DONE")
    print("=" * 60)

    print(
        f"Output directory:"
    )

    print(
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()