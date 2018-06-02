import numpy as np
import csv

# reads the csv and returns it as an array
def read_csv(filename):
    with open(filename, 'rb') as csvfile:
        try:
            reader = csv.reader(csvfile, delimiter=',')
            next(reader, None)  # skip the header
            rows = [r for r in reader]
            return rows
        finally:
            csvfile.close()
