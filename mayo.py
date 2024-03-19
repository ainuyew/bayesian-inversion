import pydicom
import numpy as np
import os
import zipfile
import tqdm
from pathlib import Path
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

def get_training_data(folder, slice_start=0, slice_end=-1):
    fd_imas = []
    qd_imas = []
    fd_path=f'{folder}/full_3mm'
    qd_path=f'{folder}/quarter_3mm'

    for search_path, imas in [(fd_path, fd_imas), (qd_path, qd_imas)]:
        file_paths = sorted(list(Path(search_path).rglob('*.IMA')))[slice_start:slice_end]
        for ima_file_path in tqdm.tqdm(file_paths, f'loading IMA files from {search_path}'):
            ima = pydicom.read_file(ima_file_path)
            rows = ima.Rows
            cols = ima.Columns

            pixel_array = ima.pixel_array

            # convert to HU
            hu_values = ima.RescaleSlope * pixel_array + ima.RescaleIntercept
            densities = (hu_values + 1000)/1000

            imas.append(densities)

    return list(zip(fd_imas, qd_imas))


def main():
    #path='/Volumes/SEAGATE_1TB/Huiyuan/projects/Mayo_Grand_Challenge/Patient_Data/Training_Projection_Data/L067/DICOM-CT-PD_QD.zip'
    path='/Users/huiyuanchua/Documents/data/Mayo_Grand_Challenge/Patient_Data/Training_Image_Data/3mm B30'
    #path='/media/huiyuanchua/SEAGATE_1TB/Huiyuan/projects/Mayo_Grand_Challenge/Patient_Data/Training_Projection_Data/L067/DICOM-CT-PD_QD.zip'
    training_data = get_training_data(path)
    print(len(training_data))

if __name__ == '__main__':
    main()
