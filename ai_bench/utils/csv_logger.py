import csv
import os


class CSVLogger:
    def __init__(self, csv_path, fieldnames):
        self.csv_path = csv_path
        self.fieldnames = fieldnames
        self._ensure_header()

    def _ensure_header(self):
        write_header = True
        if os.path.isfile(self.csv_path):
            with open(self.csv_path, "r", newline="") as csvfile:
                write_header = csvfile.read(1) == ""
        if write_header:
            with open(self.csv_path, "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                writer.writeheader()

    def log(self, row):
        with open(self.csv_path, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writerow(row)
