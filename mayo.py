import pydicom
import numpy as np
import os
import zipfile
import tqdm
from pathlib import Path
from io import BytesIO, TextIOWrapper
from skimage.transform import resize, radon, rescale, iradon

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

def add_poisson_noise(image, N0=30000, slice_thickness=.03, epsilon=5, filter_name='ramp'):

    theta = np.linspace(0., 180., max(image.shape), endpoint=False)
    sinogram = radon(image, theta=theta, circle=False)

    sinogram_in = N0 * np.exp(-sinogram * slice_thickness)
    sinogram_noisy = np.random.poisson(sinogram_in)
    sinogram_out = -np.log(sinogram_noisy/N0)/slice_thickness

    # update inf values
    idx = np.isinf(sinogram_out)
    sinogram_out[idx] = -np.log(epsilon/N0)/slice_thickness

    fbp_image = iradon(sinogram_out, theta=theta, filter_name=filter_name, circle=False)
    return fbp_image

def add_simple_noise(image, peak=1):
    image_in = (image - image.min())/(image.max() - image.min()) * 255
    noise_mask = np.random.poisson(image_in/255 * peak)/peak * 255
    return image_in + noise_mask

def get_pixel_arrays(file_paths):
    pixel_arrays = []
    for ima_file_path in file_paths:
        ima = pydicom.read_file(ima_file_path)

        pixel_array = ima.pixel_array

        # convert to HU
        hu_values = ima.RescaleSlope * pixel_array + ima.RescaleIntercept

        # resize image to run with smaller ram/vram
        #hu_values = resize(hu_values, (hu_values.shape[0] // 4, hu_values.shape[1] // 4), anti_aliasing=True)

        # rescale 1/1000 HU
        hu_values = hu_values / 1000.

        pixel_arrays.append(hu_values.reshape((hu_values.shape[0], hu_values.shape[1], 1)))

    return pixel_arrays

def get_data(folder, patients, slice_start=0, slice_end=-1):
    fd_data = []
    ld_data = []
    fd_path=f'{folder}/full_3mm'
    ld_path=f'{folder}/quarter_3mm'

    #patients=sorted([p for p in os.listdir(fd_path) if not p.startswith('.')])

    for patient in tqdm.tqdm(patients, f'loading patient data'):
        fd_file_paths = sorted(list(Path(f'{fd_path}/{patient}').rglob('*.IMA')))[slice_start:slice_end]
        ld_file_paths = sorted(list(Path(f'{ld_path}/{patient}').rglob('*.IMA')))[slice_start:slice_end]

        fd_pixel_arrays = get_pixel_arrays(fd_file_paths)
        ld_pixel_arrays = get_pixel_arrays(ld_file_paths)

        fd_data[0:0] = fd_pixel_arrays # concatenate two lists
        ld_data[0:0] = ld_pixel_arrays

    return np.array(list(zip(fd_data, ld_data)))

def get_training_data(folder, slice_start=0, slice_end=-1):
    fd_path=f'{folder}/full_3mm'
    patients=sorted([p for p in os.listdir(fd_path) if not p.startswith('.')])
    n = len(patients)//10 * 9
    return get_data(folder, patients[:n], slice_start, slice_end)

def get_test_data(folder, slice_start=0, slice_end=-1):
    fd_path=f'{folder}/full_3mm'
    patients=sorted([p for p in os.listdir(fd_path) if not p.startswith('.')])
    n = len(patients)//10

    return get_data(folder, patients[-n:], slice_start, slice_end)


def main():
    #path='/Volumes/SEAGATE_1TB/Huiyuan/projects/Mayo_Grand_Challenge/Patient_Data/Training_Projection_Data/L067/DICOM-CT-PD_QD.zip'
    path='/Users/huiyuanchua/Documents/data/Mayo_Grand_Challenge/Patient_Data/Training_Image_Data/3mm B30'
    #path='/media/huiyuanchua/SEAGATE_1TB/Huiyuan/projects/Mayo_Grand_Challenge/Patient_Data/Training_Projection_Data/L067/DICOM-CT-PD_QD.zip'
    training_data = get_training_data(path)
    print(len(training_data))

if __name__ == '__main__':
    main()
