from pydicom import dcmread
from pydicom.data import get_testdata_file
import numpy as np
from pathlib import Path
import os
import zipfile
from io import TextIOWrapper

def unzip(zip_path):
    with zipfile.ZipFile(zip_path) as zf:
        files = []
        for file_name in zf.namelist():
            print(file_name)
            if file_name.endswith('.dcm'):
                with TextIOWrapper(zf.open(file_name)) as f:
                    #file_string = f.read()
                    #file_string.seek(0)
                    dcm = dcmread(fp=f)
                name = os.path.basename(file_name)
                files.append((name, dcm))
    return files

def get_training_data(folder):
    ds = []
    pathlist = Path(folder).rglob('DICOM-CT-PD_QD.zip')
    for zip_path in pathlist:
        files_in_zip = unzip(zip_path)
        for file_name, file_string in files_in_zip:
            f = dcmread(fp=filepath)
            ds.append(f.pixel_array)
    return np.stack(ds)


def main():
  path='/Volumes/SEAGATE_1TB/Huiyuan/projects/Mayo_Grand_Challenge/Patient_Data/Training_Projection_Data/L067/DICOM-CT-PD_QD.zip'
  print(unzip(path))

if __name__ == '__main__':
    main()
