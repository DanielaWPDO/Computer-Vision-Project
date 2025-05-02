import csv
from pathlib import Path
import config

# Define base path
base_dir = Path("dataset")
output_csv = config.CSV_PATH

# Collect image paths and labels
rows = []

for label in config.CLASS_NAMES :
    class_dir = base_dir / label
    for img_path in class_dir.iterdir():
        if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            rows.append([str(img_path), label])

# Write to CSV
with open(output_csv, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filepath", "label"])
    writer.writerows(rows)

print(f"Annotation CSV saved to: {output_csv}")
