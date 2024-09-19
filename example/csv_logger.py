import csv
import os

class CSVLogger:
    def __init__(self, filename, fieldnames):
        self.filename = filename
        self.fieldnames = fieldnames
        
        # If the file doesn't exist, create it and write the header
        if not os.path.exists(self.filename):
            with open(self.filename, mode='w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=self.fieldnames)
                writer.writeheader()

    def log(self, data):
        with open(self.filename, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.fieldnames)
            writer.writerow(data)