# 2024-04-22
# This script generates the hidden layer dataset for the MNIST dataset.
# It builds on the following tutorial: https://docs.kidger.site/equinox/examples/mnist/

# %% LIBRARY IMPORT

import equinox as eqx
import jax
import jax.numpy as jnp
import os
import numpy as np
import optax  # https://github.com/deepmind/optax
import torch  # https://pytorch.org
import torchvision  # https://pytorch.org
from jaxtyping import Array, Float, Int, PyTree  # https://github.com/google/jaxtyping

from mnist_src import CNN, FashionCNN, loss, evaluate, train, evaluate_return_hidden

from lcs.utils import get_timestamp
import argparse

# %%  PARAMETERS

BATCH_SIZE = 512
LEARNING_RATE = 1e-3
STEPS = 3000
PRINT_EVERY = 30
SEED = 5678

# %%

if __name__ == '__main__':

    key = jax.random.PRNGKey(SEED)
    parser = argparse.ArgumentParser(description='Generate hidden layer dataset for MNIST or FashionMNIST.')
    parser.add_argument('--bottleneck_size', type=int, default=10, help='Size of the bottleneck layer')
    parser.add_argument('--dataset', type=str, choices=['mnist', 'fashion_mnist'], default='mnist', help='Dataset to use: mnist or fashion_mnist')
    args = parser.parse_args()
    bottleneck_size = args.bottleneck_size
    dataset_name = args.dataset

    data_folder = os.path.join('data', dataset_name)
    os.makedirs(data_folder, exist_ok=True)

    timestamp = get_timestamp()

    # %%

    if dataset_name == "mnist":
        official_dataset_name = "MNIST"
        dataset_function = torchvision.datasets.MNIST
    elif dataset_name == "fashion_mnist":
        official_dataset_name = "FashionMNIST"
        dataset_function = torchvision.datasets.FashionMNIST
    else:
        raise ValueError("Invalid dataset name")

    normalise_data = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    train_dataset = dataset_function(
        official_dataset_name,
        train=True,
        download=True,
        transform=normalise_data,
    )
    test_dataset = dataset_function(
        official_dataset_name,
        train=False,
        download=True,
        transform=normalise_data,
    )
    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    testloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=True
    )

    # Checking our data a bit (by now, everyone knows what the MNIST dataset looks like)
    dummy_x, dummy_y = next(iter(trainloader))
    dummy_x = dummy_x.numpy()
    dummy_y = dummy_y.numpy()
    print(dummy_x.shape)  # 64x1x28x28
    print(dummy_y.shape)  # 64
    print(dummy_y)

    # %%

    key, subkey = jax.random.split(key, 2)
    model = CNN(subkey, bottleneck_size=bottleneck_size)
    #model = FashionCNN(subkey)
    print(model)

    # %% TRAINING
    loss_value = loss(model, dummy_x, dummy_y)
    print(loss_value.shape)  # scalar loss
    output = jax.vmap(model)(dummy_x)
    print(output.shape)  # batch of predictions

    # %%
    value, grads = eqx.filter_value_and_grad(loss)(model, dummy_x, dummy_y)
    print(value)

    # %%

    evaluate(model, testloader)

    # %% 

    optim = optax.adamw(LEARNING_RATE)

    model = train(model, trainloader, testloader, optim, STEPS, PRINT_EVERY)

    # %% EVALUATE AFTER TRAINING

    evaluate(model, testloader)

    # %% EXTRACT FINAL HIDDEN LAYER

    xs, ys, pred_ys, hiddens, losses, accs = evaluate_return_hidden(model, testloader)

    # %%

    save_appendix = '_CNN_bottleneck%d' %bottleneck_size

    np.save(os.path.join(data_folder, '%s_xs%s.npy' %(timestamp, save_appendix)), xs)
    np.save(os.path.join(data_folder, '%s_ys%s.npy' %(timestamp, save_appendix)), ys)
    np.save(os.path.join(data_folder, '%s_pred_ys%s.npy' %(timestamp, save_appendix)), pred_ys)
    np.save(os.path.join(data_folder, '%s_hiddens%s.npy' %(timestamp, save_appendix)), hiddens)
    np.save(os.path.join(data_folder, '%s_losses%s.npy' %(timestamp, save_appendix)), losses)
    np.save(os.path.join(data_folder, '%s_accs%s.npy' %(timestamp, save_appendix)), accs)

    # %%

    xs_train, ys_train, pred_ys_train, hiddens_train, losses_train, accs_train = evaluate_return_hidden(model, trainloader)

    # %%

    np.save(os.path.join(data_folder, '%s_xs_train%s.npy' %(timestamp, save_appendix)), xs_train)
    np.save(os.path.join(data_folder, '%s_ys_train%s.npy' %(timestamp, save_appendix)), ys_train)
    np.save(os.path.join(data_folder, '%s_pred_ys_train%s.npy' %(timestamp, save_appendix)), pred_ys_train)
    np.save(os.path.join(data_folder, '%s_hiddens_train%s.npy' %(timestamp, save_appendix)), hiddens_train)
    np.save(os.path.join(data_folder, '%s_losses_train%s.npy' %(timestamp, save_appendix)), losses_train)
    np.save(os.path.join(data_folder, '%s_accs_train%s.npy' %(timestamp, save_appendix)), accs_train)

    # %%
