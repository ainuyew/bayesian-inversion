import pydicom
import numpy as np
import os
import zipfile
import tqdm
from pathlib import Path
from io import BytesIO, TextIOWrapper
from skimage.transform import resize

import utils

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

def get_pixel_arrays(file_paths):
    pixel_arrays = []
    for ima_file_path in file_paths:
        ima = pydicom.read_file(ima_file_path)

        pixel_array = ima.pixel_array

        # convert to HU
        hu_values = ima.RescaleSlope * pixel_array + ima.RescaleIntercept
        densities = hu_values + 1000 # this rescales -1000 HU to -1000 and 0 HU to 0

        # resize image to run with smaller ram/vram
        densities = resize(densities, (densities.shape[0] // 4, densities.shape[1] // 4), anti_aliasing=True)

        pixel_arrays.append(densities.reshape((densities.shape[0], densities.shape[1], 1)))

    return pixel_arrays

def get_training_data(folder, slice_start=0, slice_end=-1):
    fd_data = []
    ld_data = []
    fd_path=f'{folder}/full_3mm'
    ld_path=f'{folder}/quarter_3mm'

    patients=sorted([p for p in os.listdir(fd_path) if not p.startswith('.')])

    for patient in tqdm.tqdm(patients, f'loading patient data'):
        fd_file_paths = sorted(list(Path(f'{fd_path}/{patient}').rglob('*.IMA')))[slice_start:slice_end]
        ld_file_paths = sorted(list(Path(f'{ld_path}/{patient}').rglob('*.IMA')))[slice_start:slice_end]

        fd_pixel_arrays = get_pixel_arrays(fd_file_paths)
        ld_pixel_arrays = get_pixel_arrays(ld_file_paths)

        fd_data[0:0] = fd_pixel_arrays # concatenate two lists
        ld_data[0:0] = ld_pixel_arrays

    return np.array(list(zip(fd_data, ld_data)))

def main():
    #path='/Volumes/SEAGATE_1TB/Huiyuan/projects/Mayo_Grand_Challenge/Patient_Data/Training_Projection_Data/L067/DICOM-CT-PD_QD.zip'
    path='/Users/huiyuanchua/Documents/data/Mayo_Grand_Challenge/Patient_Data/Training_Image_Data/3mm B30'
    #path='/media/huiyuanchua/SEAGATE_1TB/Huiyuan/projects/Mayo_Grand_Challenge/Patient_Data/Training_Projection_Data/L067/DICOM-CT-PD_QD.zip'
    training_data = get_training_data(path)
    print(len(training_data))

if __name__ == '__main__':
    main()
