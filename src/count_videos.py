from pathlib import Path

DATASET_DIR = Path("dataset")

for class_dir in sorted(DATASET_DIR.iterdir()):
    if class_dir.is_dir():
        videos = list(class_dir.glob("*.avi"))
        print(f"{class_dir.name}: {len(videos)} videos")