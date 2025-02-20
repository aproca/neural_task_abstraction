# 2024-04-28
# This script contains src for the MNIST subproject

# %% LIRBARY IMPORT

import equinox as eqx
import jax
import jax.numpy as jnp
import optax  # https://github.com/deepmind/optax
import torch  # https://pytorch.org
import torchvision  # https://pytorch.org
from jaxtyping import Array, Float, Int, PyTree  # https://github.com/google/jaxtyping
import numpy as np
import os
from lcs.plotting_utils import *
import equinox as eqx
from typing import Any, Dict

#from torch.utils.data import Dataset, DataLoaders

from lcs.models import linear_model
from lcs.models import linear_model as model
from lcs.joint_learning import regularization_loss_fn
from lcs_mnist.mnist_src import cross_entropy, CNN
from lcs.configs import Config


# %% COMBINED MODEL

### FROM-SCRATCH MODEL
# class CombinedModel(eqx.Module):
#     conv1: eqx.nn.Conv2d
#     pool1: eqx.nn.MaxPool2d
#     conv2: eqx.nn.Conv2d
#     pool2: eqx.nn.MaxPool2d
#     fc1: eqx.nn.Linear
#     fc2: eqx.nn.Linear
#     cfg: Config
#     linear_params: Dict[str, jnp.ndarray]

    # def __init__(self, key, params_init, cfg):
    #     key1, key2, key3, key4, key5, key6 = jax.random.split(key, 6)
    #     self.conv1 = eqx.nn.Conv2d(1, 32, kernel_size=3, key=key1)
    #     self.pool1 = eqx.nn.MaxPool2d(kernel_size=2)
    #     self.conv2 = eqx.nn.Conv2d(32, 64, kernel_size=3, key=key2)
    #     self.pool2 = eqx.nn.MaxPool2d(kernel_size=2)
    #     self.fc1 = eqx.nn.Linear(1600, 128, key=key3)
    #     self.fc2 = eqx.nn.Linear(128, 64, key=key4)
    #     self.linear_params = params_init
    #     self.cfg = cfg

    # def __call__(self, x):
    #     x = jax.nn.relu(self.conv1(x))
    #     x = self.pool1(x)
    #     x = jax.nn.relu(self.conv2(x))
    #     x = self.pool2(x)
    #     x = jnp.ravel(x)
    #     x = jax.nn.relu(self.fc1(x))
    #     x = jax.nn.relu(self.fc2(x))
    #     W1 = self.linear_params['W1']
    #     c1 = self.linear_params['c1']
    #     #x = jnp.dot(x, W1) + c1
    #     x = jnp.einsum("p,pij,...j->...i", c1, W1, x)
    #     x = jax.nn.log_softmax(x)
    #     return x

### COMBINE EXPLICITLY
class CombinedModel(eqx.Module):
    cnn: Any
    linear_params: Dict
    cfg: Any

    def __init__(self, key, linear_params, cfg):
        key1, key2 = jax.random.split(key)
        self.cnn = CNN(key1)
        self.linear_params = linear_params
        self.cfg = cfg

    def __call__(self, x):
        cnn_output = self.cnn(x)
        cnn_output = jnp.expand_dims(cnn_output, axis=0)
        linear_output = linear_model(cnn_output, self.linear_params, self.cfg)
        softmax_output = jax.nn.softmax(linear_output, axis=-1)
        softmax_output = jnp.squeeze(softmax_output, axis=0)
        return softmax_output

#I would like to improve the model definition by combining the two models more smoothly into one. A particular feature is that in the second-to-last layer the model should take two different paths. Is this a good solution?

# %% DATASET

def create_combined_dataset(labels1, permutation2 = "standard", permutation1 = None):
    
    labels2 = permute_labels(labels1, permutation2)
    print("original labels: ", labels1)
    print("permuted labels: ", labels2)
    if permutation1 is not None:
        labels1 = permute_labels(labels1.copy())
    ys1 = np.eye(10)[labels1]
    ys2 = np.eye(10)[labels2]

    ys = jnp.stack([ys1, ys2], axis=0)
    labels = jnp.stack([labels1, labels2], axis=0)

    return ys, labels


#@jax.jit
def permute_labels(y, permutation = "standard"):
    if permutation == "standard":
        return y // 2 + 5 * (y % 2)
    elif permutation == "upper_lower":
        # Define the new order for the labels
        new_order = {
            0: 0, # T-shirt/top
            2: 1, # Pullover
            4: 2, # Coat
            6: 3, # Shirt
            8: 4, # Bag
            1: 5, # Trouser
            3: 6, # Dress
            5: 7, # Sandal
            7: 8, # Sneaker
            9: 9  # Ankle boot
        }
        
        # Map the labels to the new order
        remapped_labels = np.vectorize(new_order.get)(y)
        return remapped_labels
    elif permutation == "warm_cool":
        # Define the new order for the labels
        
        # new_order = {
        #     0: 0, # T-shirt/top
        #     3: 1, # Dress
        #     5: 2, # Sandal
        #     7: 3, # Sneaker
        #     8: 4, # Bag
        #     1: 5, # Trouser
        #     2: 6, # Pullover
        #     4: 7, # Coat
        #     6: 8, # Shirt
        #     9: 9  # Ankle boot
        # }

        ### NEW SUBMISSION FOR FINAL CAMERA READY
        new_order = {
            0: 0, # T-shirt/top
            1: 1, # Trouser
            5: 2, # Sandal
            3: 3, # Dress
            7: 4, # Sneaker
            6: 5, # Shirt
            2: 6, # Pullover
            4: 7, # Coat
            8: 8, # Bag
            9: 9  # Ankle boot
        }
        
        # Map the labels to the new order
        remapped_labels = np.vectorize(new_order.get)(y)
        
        return remapped_labels
    else:
        raise ValueError("Unknown permutation")
    
# %% LOSS FUNCTION

def mnist_loss_fn(params, X, Y_tgt, cfg):
    loss = cross_entropy( jnp.argmax(Y_tgt, axis=1), jax.nn.log_softmax(model(X, params, cfg), axis=-1)) 

    regularization_loss = regularization_loss_fn(params, cfg)
    
    loss += cfg.regularization_strength * regularization_loss

    return loss
