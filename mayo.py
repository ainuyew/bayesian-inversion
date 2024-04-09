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

def add_noise(image, N0=30000, slice_thickness=.03, epsilon=5, filter_name='ramp'):
    assert len(image.shape) == 2

    image_rescaled = (image - image.min())/(image.max() - image.min())

    theta = np.linspace(0., 180., max(image.shape), endpoint=False)
    sinogram = radon(image_rescaled, theta=theta, circle=False)

    sinogram_in = N0 * np.exp(-sinogram * slice_thickness)
    sinogram_noisy = np.random.poisson(sinogram_in) # this has to within [0., 1.]
    sinogram_out = -np.log(sinogram_noisy/N0)/slice_thickness

    # update inf values
    idx = np.isinf(sinogram_out)
    sinogram_out[idx] = -np.log(epsilon/N0)/slice_thickness

    # what is the standard deviation for eletrical noise?
    gaussian_noise = np.random.normal(0., 1e-5, sinogram_out.shape)

    reconstructed_image = iradon(sinogram_out + gaussian_noise, theta=theta, filter_name=filter_name, circle=False)

    return reconstructed_image * (image.max() - image.min()) + image.min()

def add_simple_noise(image, peak=1):
    image_rescaled = (image - image.min())/(image.max() - image.min())
    image_noise = np.random.poisson(image_rescaled * 255 * peak)/peak / 255
    return np.clip(image_noise, 0., 1.) * (image.max() - image.min()) + image.min()

def get_pixel_arrays(file_paths):
    pixel_arrays = []
    for ima_file_path in file_paths:
        ima = pydicom.read_file(ima_file_path)

        pixel_array = ima.pixel_array

        # convert to HU
        hu_values = ima.RescaleSlope * pixel_array + ima.RescaleIntercept
        hu_rescaled = utils.window_image(hu_values, 350, 50, out_range=(-1., 1.))

        # resize image to run with smaller ram/vram
        hu_rescaled = resize(hu_rescaled, (hu_rescaled.shape[0] // 4, hu_rescaled.shape[1] // 4), anti_aliasing=True)

        pixel_arrays.append(hu_rescaled.reshape((hu_rescaled.shape[0], hu_rescaled.shape[1], 1)))

    return pixel_arrays

def get_data(folder, patients, slice_start=0, slice_end=-1):
    fd_data = []
    ld_data = []
    uld_data = []
    fd_path=f'{folder}/full_3mm'
    ld_path=f'{folder}/quarter_3mm'

    #patients=sorted([p for p in os.listdir(fd_path) if not p.startswith('.')])

    for patient in tqdm.tqdm(patients, f'loading patient data'):
        fd_file_paths = sorted(list(Path(f'{fd_path}/{patient}').rglob('*.IMA')))[slice_start:slice_end]
        ld_file_paths = sorted(list(Path(f'{ld_path}/{patient}').rglob('*.IMA')))[slice_start:slice_end]

        fd_pixel_arrays = get_pixel_arrays(fd_file_paths)
        ld_pixel_arrays = get_pixel_arrays(ld_file_paths)
        uld_pixel_arrays = []
        for pixel_array in fd_pixel_arrays:
            uld_pixel_arrays.append(add_noise(pixel_array.reshape((pixel_array.shape[0], pixel_array.shape[1]))).reshape((pixel_array.shape)))

        fd_data[0:0] = fd_pixel_arrays # concatenate two lists
        ld_data[0:0] = ld_pixel_arrays
        uld_data[0:0] = uld_pixel_arrays

    return np.array(list(zip(fd_data, ld_data, uld_data)))

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
    path='/Users/huiyuanchua/Documents/data/Mayo_Grand_Challenge/Patient_Data/Training_Image_Data/3mm B30'
    training_data = get_training_data(path)
    print(len(training_data))

if __name__ == '__main__':
    main()
