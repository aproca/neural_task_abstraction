import equinox as eqx
import jax
import jax.numpy as jnp
import optax  
import torch  
import torchvision  
from jaxtyping import Array, Float, Int, PyTree  
import numpy as np

class CNN(eqx.Module):
    ## TODO: Separate out a clean second class for the part that we need - or rather just use return hidden?!?!?!
    layers: list

    def __init__(self, key, bottleneck_size=None):
        if bottleneck_size is None:
            key1, key2, key3, key4 = jax.random.split(key, 4)
            # Standard CNN setup: convolutional layer, followed by flattening,
            # with a small MLP on top.
            self.layers = [
                eqx.nn.Conv2d(1, 3, kernel_size=4, key=key1),
                eqx.nn.MaxPool2d(kernel_size=2),
                jax.nn.relu,
                jnp.ravel,
                eqx.nn.Linear(1728, 512, key=key2),
                jax.nn.sigmoid,
                eqx.nn.Linear(512, 64, key=key3),
                jax.nn.relu,
                
                ### COMMENT OUT FOR COMBINED MODEL
                eqx.nn.Linear(64, 10, key=key4),
                jax.nn.log_softmax,
            ]
        else:
            key1, key2, key3, key4, key5 = jax.random.split(key, 5)
            # Standard CNN setup: convolutional layer, followed by flattening,
            # with a small MLP on top.
            self.layers = [
                eqx.nn.Conv2d(1, 3, kernel_size=4, key=key1),
                eqx.nn.MaxPool2d(kernel_size=2),
                jax.nn.relu,
                jnp.ravel,
                eqx.nn.Linear(1728, 512, key=key2),
                jax.nn.sigmoid,
                eqx.nn.Linear(512, 64, key=key3),
                jax.nn.relu,

                eqx.nn.Linear(64, bottleneck_size, key=key3),
                jax.nn.relu,
                
                ### COMMENT OUT FOR COMBINED MODEL
                eqx.nn.Linear(bottleneck_size, 10, key=key4),
                jax.nn.log_softmax,
            ]

    def __call__(self, x: Float[Array, "1 28 28"]) -> Float[Array, "10"]:
        for layer in self.layers:
            x = layer(x)
        return x
    
    def call_return_hidden(self, x):
        for layer in self.layers[:-2]:
            x = layer(x)
        relu_output = x
        for layer in self.layers[-2:]:
            x = layer(x)
        return x, relu_output
    
# %% MORE COMPLICATED CNN FOR MNIST

class FashionCNN(eqx.Module):
    """ built accoring to https://github.com/guilhermedom/cnn-fashion-mnist/blob/main/notebooks/1.0-gdfs-cnn-fashion-mnist.ipynb """
    layers: list

    def __init__(self, key):
        keys = jax.random.split(key, 8)
        self.layers = [
            eqx.nn.Conv2d(1, 32, kernel_size=3, key=keys[0]),  # Conv2D layer with 32 filters and a 3x3 kernel
            jax.nn.relu,                                       # ReLU activation after each Conv2D
            eqx.nn.MaxPool2d(kernel_size=2, stride=2),                   # MaxPooling with 2x2 pool size
            eqx.nn.Conv2d(32, 64, kernel_size=3, key=keys[1]), # Conv2D layer with 64 filters and a 3x3 kernel
            jax.nn.relu,                                       # ReLU activation after each Conv2D
            eqx.nn.MaxPool2d(kernel_size=2, stride=2),                   # MaxPooling with 2x2 pool size
            eqx.nn.Conv2d(64, 64, kernel_size=3, key=keys[2]), # Conv2D layer with 64 filters and a 3x3 kernel
            jax.nn.relu,                                       # ReLU activation after each Conv2D
            jnp.ravel,                                         # Flatten the output before passing to dense layers
            eqx.nn.Linear(576, 250, key=keys[3]),              # Dense layer with 250 units
            jax.nn.relu,                                       # ReLU activation after first Dense layer
            eqx.nn.Linear(250, 125, key=keys[4]),              # Dense layer with 125 units
            jax.nn.relu,                                       # ReLU activation after second Dense layer
            eqx.nn.Linear(125, 60, key=keys[5]),               # Dense layer with 60 units
            jax.nn.relu,                                       # ReLU activation after third Dense layer

            ### COMMENT OUT FOR COMBINED MODEL
            eqx.nn.Linear(60, 10, key=keys[6]),                # Final Dense layer with 10 units
            jax.nn.log_softmax                                 # Log Softmax activation for classification
        ]

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for layer in self.layers:
            x = layer(x)
            #print(f"Shape after layer {layer}: {x.shape}")
        return x
    
    def call_return_hidden(self, x):
        for layer in self.layers[:-2]:
            x = layer(x)
        relu_output = x
        for layer in self.layers[-2:]:
            x = layer(x)
        return x, relu_output

# %% FUNCTIONS

def loss(
    #model: CNN, x: Float[Array, "batch 1 28 28"], y: Int[Array, " batch"]
    model, x: Float[Array, "batch 1 28 28"], y: Int[Array, " batch"]
    ) -> Float[Array, ""]:
    # Our input has the shape (BATCH_SIZE, 1, 28, 28), but our model operations on
    # a single input input image of shape (1, 28, 28).
    #
    # Therefore, we have to use jax.vmap, which in this case maps our model over the
    # leading (batch) axis.
    pred_y = jax.vmap(model)(x)
    return cross_entropy(y, pred_y)


@eqx.filter_jit
def cross_entropy(
    y: Int[Array, " batch"], pred_y: Float[Array, "batch 10"]
    ) -> Float[Array, ""]:
    # y are the true targets, and should be integers 0-9.
    # pred_y are the log-softmax'd predictions.
    pred_y = jnp.take_along_axis(pred_y, jnp.expand_dims(y, 1), axis=1)
    return -jnp.mean(pred_y)

@eqx.filter_jit
def compute_accuracy(
    model: CNN, x: Float[Array, "batch 1 28 28"], y: Int[Array, " batch"]
) -> Float[Array, ""]:
    """This function takes as input the current model
    and computes the average accuracy on a batch.
    """
    pred_y = jax.vmap(model)(x)
    pred_y = jnp.argmax(pred_y, axis=1)
    return jnp.mean(y == pred_y)

def evaluate(model: CNN, testloader: torch.utils.data.DataLoader):
    """This function evaluates the model on the test dataset,
    computing both the average loss and the average accuracy.
    """
    avg_loss = 0
    avg_acc = 0
    for x, y in testloader:
        x = x.numpy()
        y = y.numpy()
        # Note that all the JAX operations happen inside `loss` and `compute_accuracy`,
        # and both have JIT wrappers, so this is fast.
        avg_loss += loss(model, x, y)
        avg_acc += compute_accuracy(model, x, y)
    return avg_loss / len(testloader), avg_acc / len(testloader)

def evaluate_return_hidden(model: CNN, testloader: torch.utils.data.DataLoader):
    """This function evaluates the model on the test dataset,
    computing both the average loss and the average accuracy.
    """
    
    xs = []
    ys = []
    hiddens = []
    pred_ys = []
    accs = []
    losses = []

    for x, y in testloader:
        x = x.numpy()
        y = y.numpy()

        xs.append(x)
        ys.append(y)

        # Note that all the JAX operations happen inside `loss` and `compute_accuracy`,
        # and both have JIT wrappers, so this is fast.
        losses.append(loss(model, x, y))

        pred_y, hidden = jax.vmap(model.call_return_hidden)(x)
        pred_y = jnp.argmax(pred_y, axis=1)
        pred_ys.append(pred_y)
        hiddens.append(hidden)

        accs.append(jnp.mean(y == pred_y))


    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0), np.concatenate(pred_ys, axis=0), np.concatenate(hiddens, axis=0), np.stack(losses), np.stack(accs)

def train(
    model: CNN,
    trainloader: torch.utils.data.DataLoader,
    testloader: torch.utils.data.DataLoader,
    optim: optax.GradientTransformation,
    steps: int,
    print_every: int,
) -> CNN:
    # Just like earlier: It only makes sense to train the arrays in our model,
    # so filter out everything else.
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    # Always wrap everything -- computing gradients, running the optimiser, updating
    # the model -- into a single JIT region. This ensures things run as fast as
    # possible.
    @eqx.filter_jit
    def make_step(
        model: CNN,
        opt_state: PyTree,
        x: Float[Array, "batch 1 28 28"],
        y: Int[Array, " batch"],
    ):
        loss_value, grads = eqx.filter_value_and_grad(loss)(model, x, y)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value

    # Loop over our training dataset as many times as we need.
    def infinite_trainloader():
        while True:
            yield from trainloader

    for step, (x, y) in zip(range(steps), infinite_trainloader()):
        # PyTorch dataloaders give PyTorch tensors by default,
        # so convert them to NumPy arrays.
        x = x.numpy()
        y = y.numpy()
        model, opt_state, train_loss = make_step(model, opt_state, x, y)
        if (step % print_every) == 0 or (step == steps - 1):
            test_loss, test_accuracy = evaluate(model, testloader)
            print(
                f"{step=}, train_loss={train_loss.item()}, "
                f"test_loss={test_loss.item()}, test_accuracy={test_accuracy.item()}"
            )
    return model
# %%
