import shutil
import re
from pathlib import Path
from collections import defaultdict
import random

# ---------------- CONFIG ----------------
DATASET_ROOT = Path("pta-dataset")
TRAIN_DIR = DATASET_ROOT / "train"
VAL_DIR = DATASET_ROOT / "val"
TEST_DIR = DATASET_ROOT / "test"

CLASSES = ["0", "1", "2", "3", "4"]
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
RANDOM_SEED = 42
# ----------------------------------------

random.seed(RANDOM_SEED)

PAGE_RE = re.compile(r"page_(\d+)_row_(\d+)")


def ensure_dirs():
    for base in [VAL_DIR, TEST_DIR]:
        for cls in CLASSES:
            (base / cls).mkdir(parents=True, exist_ok=True)


def main():
    ensure_dirs()

    for cls in CLASSES:
        class_dir = TRAIN_DIR / cls
        if not class_dir.exists():
            continue

        # Group images by page number
        pages = defaultdict(list)

        for img in class_dir.iterdir():
            if img.suffix.lower() not in IMAGE_EXTS:
                continue

            m = PAGE_RE.match(img.stem)
            if not m:
                continue

            page_num = int(m.group(1))
            pages[page_num].append(img)

        # Process each page group
        for page_num, imgs in pages.items():
            if len(imgs) < 3:
                # Too few → keep all in train
                continue

            random.shuffle(imgs)

            val_img = imgs[0]
            test_img = imgs[1]

            shutil.move(
                str(val_img),
                str(VAL_DIR / cls / val_img.name)
            )

            shutil.move(
                str(test_img),
                str(TEST_DIR / cls / test_img.name)
            )

    print("✅ Page-aware split completed")


if __name__ == "__main__":
    main()
