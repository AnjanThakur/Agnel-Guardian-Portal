import os
import sys

# Ensure app directory is in path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.services.marks_ingest import process_marks_csv

def main():
    csv_path = "test_marks.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    print(f"Reading {csv_path}...")
    with open(csv_path, "rb") as f:
        file_content = f.read()

    print("Processing marks and inserting into MongoDB...")
    result = process_marks_csv(file_content)
    
    print("Success!")
    print(f"Result: {result}")

if __name__ == "__main__":
    main()
