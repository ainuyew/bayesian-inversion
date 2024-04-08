import matplotlib.pyplot as plt
import optax
from jax import random
import jax.numpy as jnp
from tqdm import tqdm
import numpy as np
import h5py
from pathlib import Path
import os

import utils
from unet import Unet
import mayo

def sample(state, condition, n, ts, key):
    # random white noise X_T
    key, subkey = random.split(key)
    x_t = random.normal(subkey, shape=(n, condition.shape[0], condition.shape[1], 1))

    condition = np.repeat(condition.reshape((1, condition.shape[0], condition.shape[1], 1)), n, axis=0)

    step=0

    for k in range(len(ts))[::-1]:
        key, subkey = random.split(key)
        z = random.normal(subkey, shape=x_t.shape)

        t = ts[k]
        dt = jnp.where(k > 0, t - ts[k-1], 0.)

        inputs = jnp.concatenate((x_t, condition), axis=-1)


        f_theta = state.apply_fn(state.params, inputs, t * jnp.ones((n,)))

        # equation (40)
        s_theta = jnp.where(k > 0, x_t/(1-jnp.exp(-t))  - jnp.exp(-t/2)/(1-jnp.exp(-t)) * f_theta,  0.)

        # equation (24)
        x_t_bar = x_t - dt * s_theta
        x_t = jnp.exp(dt/2) * x_t_bar + jnp.sqrt(1-jnp.exp(-dt)) * z

        x_t = jnp.clip(x_t, -1., 1.) # should we clip ...
        #x_t = normalize_to_neg_one_to_one(x_t) # or scale?

        step=step+1

    return x_t

def main(n_samples=1000):
    key = random.PRNGKey(SEED)

    # use the best params
    file_path, epoch, step, loss = utils.find_latest_pytree(f'{PROJECT_DIR}/cem_params_*.npy')
    cem_state = utils.create_training_state(params_file=file_path, param_shape=(1, 128, 128, 2))
    print(f'using parameters from epoch {epoch} with loss {loss}')

    ts = utils.exponential_time_schedule(T, K)

    with h5py.File(f'{Path.home()}/Documents/data/mayo.hdf5', 'r') as hf:
        fd_data, _, uld_data = hf['test'][36]

    # create empty samples HDF5 to store samples
    samples_path = f'{PROJECT_DIR}/cem_samples.hdf5'
    with h5py.File(samples_path, 'w') as hf:
        samples = hf.create_dataset('samples', data=np.zeros((n_samples, 128, 128, 1)), compression='gzip', chunks=True)

    for i in tqdm(range(n_samples // BATCH_SIZE)):
        # generate x_0 from noise
        key, subkey = random.split(key)
        x_0_tilde = sample(cem_state, uld_data, BATCH_SIZE, ts, subkey)

        with h5py.File(samples_path, 'a') as hf:
            hf['samples'][(i * BATCH_SIZE):((i + 1) * BATCH_SIZE)] = x_0_tilde

SEED=42
T=10.
K=1000
BATCH_SIZE = 5
PROJECT_DIR=os.path.abspath('.')

if __name__ == '__main__':
    try:
        main(n_samples=100)
    except Exception as e:
        print(f'Unexpected {e}, {type(e)}')
