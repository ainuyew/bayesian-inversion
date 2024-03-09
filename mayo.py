from pydicom import dcmread
from pydicom.data import get_testdata_file
import numpy as np
from pathlib import Path
import os
import zipfile
from io import StringIO

def unzip(zip_path):
    zip_file = zipfile.ZipFile(zip_path, 'r')
    files = []
    for file_name in zip_file.namelist():
        file_object = zip_file.open(file_name, 'r')
        file_string = StringIO(file_object.read())
        file_object.close()
        file_string.seek(0)
        name = os.path.basename(file_name)
        files.append((name, file_string))
    return files

def get_training_data(folder):
    ds = []
    pathlist = Path(folder).rglob('*.zip')
    for zip_path in pathlist:
        files_in_zip = unzip(zip_path)
        for file_name, file_string in files_in_zip:
            f = dcmread(fp=filepath)
            ds.append(f.pixel_array)
    return np.stack(ds)


def main():
  path='/Volumes/SEAGATE_1TB/Huiyuan/projects/Mayo_Grand_Challenge/Patient_Data/Training_Projection_Data/L067.zip'
  print(unzip(path))

if __name__ == '__main__':
    main()
