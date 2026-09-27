from pathlib import Path
import numpy as np


PROJECT_DIR = Path(r"D:\GMM_HMM_Action_Recognition")
DATASET_DIR = PROJECT_DIR / "output" / "dataset_features"

CLASSES = [
    "BodyWeightSquats",
    "JumpingJack",
    "PullUps",
    "Punch",
    "PushUps"
]


def main():

    print("=" * 60)
    print("           CHECK DATASET FEATURES")
    print("=" * 60)

    total_files = 0
    total_frames = 0

    for class_name in CLASSES:

        class_dir = DATASET_DIR / class_name

        if not class_dir.exists():
            print(f"\n[WARNING] Missing folder: {class_name}")
            continue

        files = sorted(class_dir.glob("*.npy"))

        print(f"\n[{class_name}]")
        print(f"Videos: {len(files)}")

        class_frames = 0

        for file_path in files:

            try:
                data = np.load(file_path)

                print(
                    f"  {file_path.name:<45} "
                    f"shape={data.shape}"
                )

                if data.ndim != 2:
                    print("      [WARNING] Data is not 2D")

                elif data.shape[1] != 99:
                    print(
                        f"      [WARNING] Expected 99 features, "
                        f"got {data.shape[1]}"
                    )

                else:
                    frames = data.shape[0]
                    class_frames += frames
                    total_frames += frames
                    total_files += 1

            except Exception as e:
                print(f"  [ERROR] {file_path.name}")
                print(f"      {e}")

        print(f"Total frames in class: {class_frames}")

    print("\n" + "=" * 60)
    print("                 SUMMARY")
    print("=" * 60)

    print(f"Valid files : {total_files}")
    print(f"Total frames: {total_frames}")

    print("=" * 60)


if __name__ == "__main__":
    main()