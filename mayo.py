from pydicom import dcmread
from pydicom.data import get_testdata_file
import numpy as np
from pathlib import Path
import os
import zipfile
from io import BytesIO, TextIOWrapper

def unzip(zip_path):
    with zipfile.ZipFile(zip_path) as zf:
        files = []
        for file_name in zf.namelist():
            if not file_name.endswith('.dcm'):
                continue

            with zf.open(file_name) as file_object:
                file_bytes = file_object.read()
                dcm = dcmread(BytesIO(file_bytes))
            name = os.path.basename(file_name)
            files.append((name, dcm))
    return files

def get_training_data(folder):
    fd_ima = []
    qd_ima = []
    fd_path=f'{folder}/full_3mm'
    qd_path=f'{folder}/quarter_3mm'
    pathlist = Path(folder).rglob('*.ima')
    for zip_path in pathlist:
        zip_file_name = os.path.basename(zip_path)
        dcm_files = unzip(zip_path)
        for file_name, dcm in dcm_files:
            ds.append((zip_file_name, file_name, dcm))
    return ds


def main():
    #path='/Volumes/SEAGATE_1TB/Huiyuan/projects/Mayo_Grand_Challenge/Patient_Data/Training_Projection_Data/L067/DICOM-CT-PD_QD.zip'
    path='/Users/huiyuanchua/Documents/Mayo_Grand_Challenge/Patient_Data/Training_Image_Data/3mm B30'
    #path='/media/huiyuanchua/SEAGATE_1TB/Huiyuan/projects/Mayo_Grand_Challenge/Patient_Data/Training_Projection_Data/L067/DICOM-CT-PD_QD.zip'
    training_data = get_training_data(path)
    print(len(training_data))

if __name__ == '__main__':
    main()
