import csv
import os

def append_log_row(log_path, row_dict, fieldnames):
    file_exists = os.path.exists(log_path)

    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)