import os
import json
import random

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = r"D:\GMM_HMM_Action_Recognition"

FEATURE_DIR = os.path.join(
    BASE_DIR,
    "output",
    "dataset_features"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "output",
    "dataset_split.json"
)

CLASSES = [
    "BodyWeightSquats",
    "JumpingJack",
    "PullUps",
    "Punch",
    "PushUps"
]

TRAIN_RATIO = 0.8
RANDOM_SEED = 42


# ============================================================
# MAIN
# ============================================================

def main():

    random.seed(RANDOM_SEED)

    train_data = {}
    test_data = {}

    total_train = 0
    total_test = 0

    print("=" * 60)
    print("       CREATING TRAIN / TEST SPLIT")
    print("=" * 60)

    for class_name in CLASSES:

        class_dir = os.path.join(
            FEATURE_DIR,
            class_name
        )

        if not os.path.exists(class_dir):
            print(f"[ERROR] Folder not found: {class_name}")
            continue

        files = [
            f for f in os.listdir(class_dir)
            if f.endswith(".npy")
        ]

        files.sort()

        # Shuffle nhưng cố định kết quả
        random.shuffle(files)

        split_index = int(len(files) * TRAIN_RATIO)

        train_files = files[:split_index]
        test_files = files[split_index:]

        train_data[class_name] = train_files
        test_data[class_name] = test_files

        total_train += len(train_files)
        total_test += len(test_files)

        print()
        print(f"{class_name}")
        print(f"  Total : {len(files)}")
        print(f"  Train : {len(train_files)}")
        print(f"  Test  : {len(test_files)}")

    # ========================================================
    # SAVE
    # ========================================================

    split_data = {
        "random_seed": RANDOM_SEED,
        "train_ratio": TRAIN_RATIO,
        "train": train_data,
        "test": test_data
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            split_data,
            f,
            indent=4,
            ensure_ascii=False
        )

    print()
    print("=" * 60)
    print("              SPLIT DONE")
    print("=" * 60)

    print(f"Total Train : {total_train}")
    print(f"Total Test  : {total_test}")
    print(f"Total       : {total_train + total_test}")

    print()
    print(f"Saved to:")
    print(OUTPUT_FILE)

    print("=" * 60)


if __name__ == "__main__":
    main()