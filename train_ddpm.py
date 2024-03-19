from jax import jit, random, value_and_grad
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
import os
import time

import unet
import utils
import mayo

@jit
def forward_process(x_0, a_bar_k, eta):
    assert x_0.shape[0] == a_bar_k.shape[0]
    assert x_0.shape == eta.shape
    assert len(a_bar_k.shape) == 4

    return jnp.sqrt(a_bar_k) * x_0 + jnp.sqrt(1 - a_bar_k) * eta

def fit(state, training_data, key, ks, alpha_bars, batch_size, n_epoch, patience, step=0, epoch_start=0):
    @jit
    def mse_loss(params, inputs, k, targets) -> jnp.float32:
        assert inputs.shape[0] == k.shape[0]
        assert inputs.shape == targets.shape

        n = targets.shape[0]
        predictions = state.apply_fn(params, inputs, k)
        diffs = predictions.reshape((n, - 1)) - targets.reshape((n, -1)) # differences of the flattened images (n, 28*28)
        return (diffs * diffs).mean(axis=1).mean()

    n_batch=training_data.shape[0] // batch_size

    loss_log = None
    best_loss = 1.
    n_stale_epoch=1

    for epoch in range(epoch_start, n_epoch):
      key, subkey = random.split(key)
      perms = random.permutation(subkey, training_data.shape[0])
      perms = perms[: n_batch * batch_size] # skip incomplete batch
      perms = perms.reshape((n_batch, batch_size))

      loss_log = []

      for perm in tqdm(perms, desc=f'epoch {epoch}'):

          # randomly pick a subset of the entire sample size
          x_0_batch = training_data[perm, ...]

          # regenerate a new random keys
          key, key2, key3 = random.split(key, 3)

          n = x_0_batch.shape[0]

          k = random.choice(key2, ks, shape=(n,))
          alpha_bar_k = alpha_bars[k, None, None, None] # (n, ) -> (n, 1, 1, 1)
          eta = random.normal(key3, shape=x_0_batch.shape) # (n, 28, 28, 1)

          x_k = forward_process(x_0_batch, alpha_bar_k, eta)

          loss, grads = value_and_grad(mse_loss)(state.params, x_k, k, eta)

          state = state.apply_gradients(grads=grads)

          step = step+1
          loss_log.append((epoch, step, loss))

      #utils.save_checkpoint(CHECKPOINT_DIR, state, epoch, step)
      utils.save_loss_log(loss_log, LOSS_LOG)

      epoch_loss = np.mean([loss for _, _, loss in loss_log])

      if epoch_loss < best_loss:
          best_loss = epoch_loss
          utils.save_pytree(state.params, f'{PROJECT_DIR}/ddpm_params_{epoch}_{step}_{best_loss:.5f}')
          n_stale_epoch=1
      elif n_stale_epoch < patience:
          n_stale_epoch += 1
      else:
          print(f'stop training early after {epoch} epochs with a best loss of {best_loss} ')
          return state

    return state

if __name__ == '__main__':
    PROJECT_DIR=os.path.abspath('.')
    CHECKPOINT_DIR=os.path.abspath('/tmp/ddpm')
    LOSS_LOG= f'{PROJECT_DIR}/ddpm_loss_log.npy'
    SEED=42
    BATCH_SIZE=10
    N_EPOCH=10
    MIN_BETA=1e-4
    MAX_BETA=.02
    K = 200
    PATIENCE=5

    key = random.PRNGKey(SEED)
    key, key2, key3 = random.split(key, 3)

    RESUME=False
    epoch_start=-1
    step=-1

    if RESUME:
        state, epoch_start, step = utils.restore_checkpoint(CHECKPOINT_DIR)
        print(f'restore checkpoint from epoch {epoch_start} and step {step}')
    else:
        state = utils.create_training_state(key=key2)
        #utils.save_checkpoint(CHECKPOINT_DIR, state, epoch_start, step)

    path='/Users/huiyuanchua/Documents/data/Mayo_Grand_Challenge/Patient_Data/Training_Image_Data/3mm B30'
    training_data = mayo.get_training_data(path, 1089, 1289)
    training_data = jnp.array([np.concatenate((fd.reshape(512, 512, 1), qd.reshape(512, 512, 1)), axis=2) for fd, qd in training_data])

    betas = jnp.linspace(MIN_BETA, MAX_BETA, K, dtype=jnp.float32) # noise variance
    alphas = 1- betas
    alpha_bars = jnp.cumprod(alphas)
    ks = jnp.array(range(len(betas))) # noise variance indexes

    start = time.time()
    state = fit(state, training_data, key3, ks, alpha_bars, BATCH_SIZE, N_EPOCH, PATIENCE, step=step+1, epoch_start=epoch_start+1)
    end = time.time()

    print(f'elapsed: {end - start}s')
