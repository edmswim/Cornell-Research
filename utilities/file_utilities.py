import os
import glob

def create_folder(path):
    try:
        if os.path.isdir(path):
            print("Error: The directory you're attempting to create already exists") # or just pass
        else:
            os.makedirs(path)
    except IOError as exception:
        raise IOError('%s: %s' % (path, exception.strerror))
    return None


'''
returns what the next version of the file is within that
subdirectory(i.e. file0.csv, file1.csv -> next one is file2.csv)
'''
def find_next_version_file(path):
    arr = glob.glob(path + "*.csv")
    if len(arr) == 0:
        return 0
    else:
        return len(arr)
