import shutil
from pathlib import Path
import re

# ---------------- CONFIG ----------------
DEBUG_ROOT = Path("app/debug_out/2025-12-27_00-07-40_pta_free")          # <-- change if needed
OUTPUT_DIR = Path("app/pta_dataset/all_rows/pdf 3")  # temp holding folder
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
# ----------------------------------------


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    page_re = re.compile(r"page_(\d+)")
    row_re = re.compile(r"row_(\d+)_rating_strip")

    copied = 0
    skipped = 0

    for page_dir in sorted(DEBUG_ROOT.iterdir()):
        if not page_dir.is_dir():
            continue

        m_page = page_re.match(page_dir.name)
        if not m_page:
            continue

        page_idx = int(m_page.group(1))

        for img in page_dir.iterdir():
            if img.suffix.lower() not in IMAGE_EXTS:
                continue

            m_row = row_re.match(img.stem)
            if not m_row:
                skipped += 1
                continue

            row_idx = int(m_row.group(1))

            new_name = f"page_{page_idx:03d}_row_{row_idx:02d}.png"
            dst = OUTPUT_DIR / new_name

            if dst.exists():
                print(f"⚠️ Skipping existing: {dst.name}")
                skipped += 1
                continue

            shutil.copy2(img, dst)
            copied += 1

    print("\n✅ Done")
    print(f"Copied  : {copied}")
    print(f"Skipped : {skipped}")
    print(f"Output  : {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
