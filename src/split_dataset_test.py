from pathlib import Path
import json


PROJECT_DIR = Path(r"D:\GMM_HMM_Action_Recognition")
DATASET_DIR = PROJECT_DIR / "output" / "dataset_features"
SPLIT_FILE = PROJECT_DIR / "output" / "dataset_split.json"


CLASSES = [
    "BodyWeightSquats",
    "JumpingJack",
    "PullUps",
    "Punch",
    "PushUps"
]


# Các video dùng cho TEST.
# Mỗi video được giữ nguyên thành một sequence.
TEST_FILES = {
    "BodyWeightSquats": [
        "v_BodyWeightSquats_g02_c01.npy"
    ],

    "JumpingJack": [
        "v_JumpingJack_g01_c05.npy"
    ],

    "PullUps": [
        "v_PullUps_g02_c01.npy"
    ],

    "Punch": [
        "v_Punch_g01_c04.npy"
    ],

    "PushUps": [
        "v_PushUps_g01_c05.npy"
    ]
}


def main():

    train_files = {}
    test_files = {}

    total_train = 0
    total_test = 0

    print("=" * 60)
    print("              TRAIN / TEST SPLIT")
    print("=" * 60)

    for class_name in CLASSES:

        class_dir = DATASET_DIR / class_name

        if not class_dir.exists():
            print(f"[ERROR] Missing directory: {class_dir}")
            return

        all_files = sorted(class_dir.glob("*.npy"))

        test_names = set(TEST_FILES[class_name])

        current_train = []
        current_test = []

        for file_path in all_files:

            if file_path.name in test_names:
                current_test.append(file_path.name)
            else:
                current_train.append(file_path.name)

        train_files[class_name] = current_train
        test_files[class_name] = current_test

        total_train += len(current_train)
        total_test += len(current_test)

        print(f"\n{class_name}")
        print(f"  Train: {len(current_train)}")
        for name in current_train:
            print(f"    - {name}")

        print(f"  Test : {len(current_test)}")
        for name in current_test:
            print(f"    - {name}")

    split_data = {
        "train": train_files,
        "test": test_files
    }

    with open(SPLIT_FILE, "w", encoding="utf-8") as f:
        json.dump(
            split_data,
            f,
            indent=4,
            ensure_ascii=False
        )

    print("\n" + "=" * 60)
    print("                 SPLIT DONE")
    print("=" * 60)

    print(f"Train videos : {total_train}")
    print(f"Test videos  : {total_test}")
    print(f"Output       : {SPLIT_FILE}")

    print("=" * 60)


if __name__ == "__main__":
    main()