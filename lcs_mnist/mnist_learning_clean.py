from collections import namedtuple
import pprint
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
import pickle, yaml
from lcs.plotting_utils import *
import jax.random as random
import logging
logger = logging.getLogger(__name__)
import os
#os.chdir('lcs_mnist')
from lcs.configs import Config
from lcs.joint_learning import get_tape_module
from lcs.models import linear_model as model
from lcs.utils import get_timestamp
#from mnist_src import loss as mnist_loss
from lcs_mnist.mnist_src import cross_entropy
from lcs_mnist.mnist_lcs_src import create_combined_dataset, mnist_loss_fn
import argparse

# %% INITIALIZATION OF MODEL

def simulate_(model, init_key, tape, traindata, testdata, cfg, params_init=None):

    params_init = dict()
    init_key, *init_keys_layers = random.split(init_key, cfg.num_layers + 1)
    for l in range(cfg.num_layers):
        if cfg.control == 'N_diag_mono':
            params_init['c' + str(l+1)] = jnp.full((cfg.num_paths*cfg.hidden_size,), 0.5)
        else:
            params_init['c' + str(l+1)] = jnp.full((cfg.num_paths,), 0.5)

        # here we use initialization that keeps y_i = W_ij x_j << 1 to allow for the W to grow
        std = .01*(cfg.initialization_scale / cfg.input_size**0.5)
        if cfg.control == '2_diag_mono' or cfg.control == 'N_diag_mono' or cfg.control == 'deep_mono':
            params_init['W' + str(l+1)] = std*random.normal(init_keys_layers[l], (cfg.layer_sizes[l+1], cfg.layer_sizes[l]))
        else:
            params_init['W' + str(l+1)] = std*random.normal(init_keys_layers[l], (cfg.num_paths, cfg.layer_sizes[l+1], cfg.layer_sizes[l]))
            
    learning_rates = dict()
    for k, v in params_init.items():
        if 'W' in k:
            learning_rates[k] = cfg.W_lr
        elif 'c' in k:
            learning_rates[k] = cfg.c_lr
        else:
            raise ValueError

    ts_out = jnp.linspace(0, cfg.t_tot, cfg.T_tape)
    dt_tape = jnp.diff(ts_out)[0]
    dt = cfg.dt

    current_batch_indices = jnp.array([0, 0])

    def loop_body(tape, params, ti, ti_tape, c_gt, traindata, testdata, data_key, current_batch_indices):  # in-out for each loop iteration (c_gt needs not be returned though)
        # for orthogonal teachers

        trainhiddens, trainys = traindata
        n_batches = trainhiddens.shape[0] // cfg.batch_size

        ## reshuffle if at the beginning of a batch
        current_batch_index = current_batch_indices[c_gt]

        def reshuffle_traindata(args):
            trainhiddens, trainys_context, data_key = args
            indices = jnp.arange(trainhiddens.shape[0])
            shuffled_indices = random.permutation(data_key, indices)
            trainhiddens = jnp.take(trainhiddens, shuffled_indices, axis=0)
            trainys_context = jnp.take(trainys_context, shuffled_indices, axis=0)
            return (trainhiddens, trainys_context, data_key)

        trainhiddens, trainys_context, data_key = lax.cond((current_batch_index % n_batches) == 0, reshuffle_traindata, lambda x: x, (trainhiddens, trainys[c_gt], data_key))
        trainys = trainys.at[c_gt].set(trainys_context)

        # get the current batch
        #batch_indices = jnp.arange((current_batch_index % n_batches)*cfg.batch_size, ((current_batch_index % n_batches)+1)*cfg.batch_size)
        start_index = (current_batch_index % n_batches) * cfg.batch_size
        end_index = start_index + cfg.batch_size
        batch_indices = jax.lax.dynamic_slice(jnp.arange(trainhiddens.shape[0]), [start_index], [cfg.batch_size])

        X = trainhiddens[batch_indices]
        Y_tgt = trainys[c_gt, batch_indices]
        
        # Compute the predicted Y values
        Y_pred = model(X, params, cfg)

        ## add softmax
        Y_pred = jax.nn.log_softmax(Y_pred, axis=-1)

        # Compute the loss
        loss_fn_X = lambda params, X, Y_tgt, cfg: mnist_loss_fn(params, X, Y_tgt, cfg)
        loss = loss_fn_X(params, X, Y_tgt, cfg)
        
        # Compute the gradients
        loss_grad = jax.grad(loss_fn_X, 0)
        grads = loss_grad(params, X, Y_tgt, cfg)

        T_tape = cfg.T_tape

        # logging
        def write_to_tape(tape, ti_tape, loss, params, grads, c_gt, testdata):
            
            ## evaluate test data
            Y_tgt = testdata[1][c_gt]
            Y_pred = model(testdata[0], params, cfg)
            Y_pred = jax.nn.log_softmax(Y_pred, axis=-1)

            tape = eqx.tree_at(lambda tape: tape.t, tape, tape.t.at[ti_tape].set(ts_out[ti_tape]))
            for l in range(cfg.num_layers):
                
                tape = eqx.tree_at(lambda tape: getattr(tape, f"c{l+1}"), tape,  getattr(tape, f"c{l+1}").at[ti_tape].set(params[f'c{l+1}']))
                tape = eqx.tree_at(lambda tape: getattr(tape, f"W{l+1}"), tape,  getattr(tape, f"W{l+1}").at[ti_tape].set(params[f'W{l+1}']))

                tape = eqx.tree_at(lambda tape: getattr(tape, f"grad_norm_c{l+1}"), tape,  getattr(tape, f"grad_norm_c{l+1}").at[ti_tape].set(jnp.linalg.norm(grads[f'c{l+1}'])))
                tape = eqx.tree_at(lambda tape: getattr(tape, f"grad_norm_W{l+1}"), tape,  getattr(tape, f"grad_norm_W{l+1}").at[ti_tape].set(jnp.linalg.norm(grads[f'W{l+1}'])))

                tape = eqx.tree_at(lambda tape: getattr(tape, f"norm_c{l+1}"), tape,  getattr(tape, f"norm_c{l+1}").at[ti_tape].set(jnp.linalg.norm(params[f'c{l+1}'])))
                tape = eqx.tree_at(lambda tape: getattr(tape, f"norm_W{l+1}"), tape,  getattr(tape, f"norm_W{l+1}").at[ti_tape].set(jnp.linalg.norm(params[f'W{l+1}'])))

                ## adding cgt to tape (since it hasn't been included previously?)
                tape = eqx.tree_at(lambda tape: getattr(tape, f"c_gt1"), tape,  getattr(tape, f"c_gt1").at[ti_tape].set(c_gt))
                
                if cfg.num_layers == 1:
                   ## FOR NOW, I AM USING THIS FIELD TO STORE ACCURACY
                   tape = eqx.tree_at(lambda tape: getattr(tape, f"context_alignment{l+1}"), tape,  getattr(tape, f"context_alignment{l+1}").at[ti_tape].set(jnp.mean(jnp.argmax(Y_tgt, axis=1) == jnp.argmax(Y_pred, axis=1))))

            ## TODO: Update to match mnist
            if not cfg.context_model and cfg.num_layers > 1:
                tape = eqx.tree_at(lambda tape: getattr(tape, f"context_alignment{l}"), tape,  getattr(tape, f"context_alignment{l}").at[ti_tape].set(jnp.mean(jnp.argmax(Y_tgt, axis=1) == jnp.argmax(Y_pred, axis=1))))
           
            #loss_without_reg = jnp.mean((Y_pred - Y_tgt)**2, axis=(-2, -1))
            loss_without_reg = cross_entropy( jnp.argmax(Y_tgt, axis=1), Y_pred) 
            ## ---- Ignoring regularization term in the saved loss ----
            tape = eqx.tree_at(lambda tape: getattr(tape, f"loss"), tape,  getattr(tape, f"loss").at[ti_tape].set(loss_without_reg))

            return tape, ti_tape + 1
        
        # log whenever t becomes larger than the next time point in the tape
        ## dt_tape is the time step in the tape, i.e. until next logging
        ## dt is simulation timestep
        tape, ti_tape = lax.cond((ti*dt >= ti_tape*dt_tape) & (ti_tape < T_tape), lambda _: write_to_tape(tape, ti_tape, loss, params, grads, c_gt, testdata), lambda _: (tape, ti_tape), None)
#
        # Update c and W
        params_n = jax.tree_util.tree_map(lambda param, lr, grad: param - dt * lr * grad, 
                                        params, learning_rates, grads)
            
        ti += 1
        current_batch_indices = current_batch_indices.at[c_gt].set(current_batch_index + 1)
        
        return tape, params_n, ti, ti_tape, c_gt, traindata, testdata, data_key, current_batch_indices

    def c_gt_(t): # TODO: code in a smarter way using the string formatting
        # JB: TODO think about how this looks in continuous time, some things needs to be multipled or divided by dt
        switch_point = int(0.75*cfg.num_blocks)
        if cfg.shared_teachers:
            if cfg.c_gt_curriculum == 'B_AB__A_B_AB__':
                c_gt = lax.select((t < switch_point), (t % cfg.num_contexts) + 1, t % (cfg.num_contexts + cfg.num_shared_contexts))
            elif cfg.c_gt_curriculum == 'B_AB__A_B__':
                c_gt = lax.select((t < switch_point), (t % cfg.num_contexts) + 1, t % (cfg.num_contexts))
            elif cfg.c_gt_curriculum == 'A_B__AB__':
                c_gt = lax.select(((t < switch_point)), (t % cfg.num_contexts), 2) 
            elif cfg.c_gt_curriculum == 'AB_BC__':
                c_gt = (t % (cfg.num_contexts-1)) + cfg.num_contexts 
            elif cfg.c_gt_curriculum == 'A_B_C__AB_BC_CA__':
                c_gt = lax.select(((t < switch_point)), (t % cfg.num_contexts), (t % cfg.num_contexts)+3) 
            elif cfg.c_gt_curriculum == 'AB_BC_CA__':
                c_gt = ((t % (cfg.num_contexts)) + cfg.num_contexts)

        if cfg.c_gt_curriculum == 'A_B__':
            c_gt = t % cfg.num_contexts
        return c_gt

    def step(carry, t2):
        """ iterate within a block """
        carry = loop_body(*carry)
        return carry, None

    def outer_step(carry, t1):
        """ iterate over blocks """
        tape, params, ti, ti_tape, c_gt, traindata, testdata, data_key, current_batch_indices = carry
        c_gt = c_gt_(t1)
        #c_gt = 1
        initial_carry_inner = (tape, params, ti, ti_tape, c_gt, traindata, testdata, data_key, current_batch_indices)
        final_carry_inner, _ = jax.lax.scan(step, initial_carry_inner, jnp.arange(cfg.T_tot // cfg.num_blocks))
        return final_carry_inner, None

    # Initialize your variables (tape, params, ti, ti_tape) appropriately
    init_data_key, init_key = random.split(init_key)
    initial_carry = (tape, params_init, 0, 0, 0, traindata, testdata, init_data_key, current_batch_indices)

    blocks = jnp.arange(cfg.num_blocks + 1)
    # step = scan_tqdm(len(blocks))(outer_step)  # prev ## DELETED - TODO: Ask Jan, WHAT DOES THIS DO?
    final_carry, _ = jax.lax.scan(outer_step, initial_carry, blocks)
    tape, params, ti, ti_tape, c_gt, traindata, testdata, data_key, current_batch_indices = final_carry

    return tape

def simulate(model, tape, traindata, testdata, cfg, params_init=None):
    # parallelize the operation
    init_keys = random.split(random.PRNGKey(0), cfg.num_seeds)
    # make a named tuple out of cfg's dict
    cfg_dict = cfg.__dict__
    MyNamedTuple = namedtuple('Config', cfg_dict.keys())
    cfg_tuple = MyNamedTuple(**cfg_dict)

    logger.info(f'Running {cfg.num_seeds} seeds with config {pprint.pformat(cfg)}')
    # only if they are different across seeds:
    vmap_dim_p = None if params_init is None or params_init['c1'].ndim == 1 else 0
    print(init_keys)
    tape = jax.vmap(simulate_, in_axes=(None, 0, 0, None, None, None, vmap_dim_p))(model, init_keys, tape, traindata, testdata, cfg_tuple, params_init)  # does this iterate over multiple seeds?

    return tape

def run_cfg(cfg, data_timestamp):

    print('initializing...')

    ## INITIALIZE TAPE        
    tape = get_tape_module(cfg)
    timestamp = get_timestamp()

    ## LOAD IN HIDDEN LAYER MNIST DATASET
    trainlabels1 = np.load(os.path.join(cfg.data_folder, '%s_ys_train%s.npy' % (data_timestamp, cfg.data_appendix)))
    trainhiddens = jnp.array(np.load(os.path.join(cfg.data_folder, '%s_hiddens_train%s.npy' %  (data_timestamp, cfg.data_appendix))))
    hiddens = jnp.array(np.load(os.path.join(cfg.data_folder, '%s_hiddens%s.npy' % (data_timestamp, cfg.data_appendix))))
    labels1 = np.load(os.path.join(cfg.data_folder, '%s_ys%s.npy' %  (data_timestamp, cfg.data_appendix)))

    trainys, trainlabels = create_combined_dataset(trainlabels1, cfg.permutation2, cfg.permutation1)
    ys, labels = create_combined_dataset(labels1, cfg.permutation2, cfg.permutation1)

    print('training...')

    # MAIN CALL: run the simulations over many seeds
    with jax.disable_jit(False):
        tape = simulate(model, tape, (trainhiddens, trainys), (hiddens, ys), cfg)
    print('done with training')

    # do some checks
    for l in range(cfg.num_layers):
        max_grad_norm_W  = getattr(tape, f'grad_norm_W{l+1}').max()
        if not max_grad_norm_W*cfg.W_lr*cfg.dt <= 1e-1:
            logger.warning(f"grad norm W{l+1} x lr is too large at {max_grad_norm_W*cfg.W_lr*cfg.dt}, not operating in Gradient Flow regime. Consider reducing dt={cfg.dt:.4f}. ")
        max_grad_norm_c  = getattr(tape, f'grad_norm_c{l+1}').max()
        if not getattr(tape, f'grad_norm_c{l+1}').max()*cfg.c_lr*cfg.dt <= 1e-1:
            logger.warning(f"grad norm c{l+1} x lr is too large at {max_grad_norm_c*cfg.c_lr*cfg.dt}, not operating in Gradient Flow regime. Consider reducing dt={cfg.dt:4f}. ")

    # monkey-patch getitem
    tape = jax.tree_util.tree_map(lambda x: np.array(x).squeeze(), tape)

    ## save information from tape
    print('saving...')
    run_info = {
        'cfg': cfg,
        'loss': tape.loss,
        't': tape.t,
        'c1': tape.c1,
        'W1': tape.W1,
        'grad_norm_c1': tape.grad_norm_c1,
        'grad_norm_W1': tape.grad_norm_W1,
        'norm_c1': tape.norm_c1,
        'norm_W1': tape.norm_W1,
        'c_gt1': tape.c_gt1,
        'context_alignment1': tape.context_alignment1,
        'data_timestamp': data_timestamp,
    }

    with open(os.path.join(cfg.results_folder, f'{timestamp}_run_info.pkl'), 'wb') as f:
        pickle.dump(run_info, f)

    ## also save info from the tape as a yaml
    with open(os.path.join(cfg.results_folder, f'{timestamp}_run_info.yaml'), 'w') as f:
        yaml.dump({**cfg.__dict__, **{'data_timestamp': data_timestamp}}, f)

    print('all done')


# %%

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MNIST Learning Script')

    parser.add_argument('--dataset_name', type=str, default='mnist', choices=['mnist', 'fashion_mnist'],
                        help="Dataset name, one of ['mnist', 'fashion_mnist']")
    parser.add_argument('--data_timestamp', type=str, default='20240513175956',
                        help="Data timestamp, one of ['20240513175956', '20240802131757']")
    parser.add_argument('--data_appendix', type=str, default='_CNN_bottleneck10',
                        help="Data appendix")
    parser.add_argument('--permutation1', type=str, default=None, choices=[None, 'upper_lower', 'standard', 'warm_cool'],
                        help="Permutation 1, one of [None, 'upper_lower', 'standard', 'warm_cool']")
    parser.add_argument('--permutation2', type=str, default='standard', choices=['standard', 'upper_lower', 'warm_cool'],
                        help="Permutation 2, one of ['standard', 'upper_lower', 'warm_cool']")
    parser.add_argument('--model', type=str, default='linear', choices=['NTA', 'control'],)

    args = parser.parse_args()

    dataset_name = args.dataset_name
    data_timestamp = args.data_timestamp
    data_appendix = args.data_appendix
    permutation1 = args.permutation1
    permutation2 = args.permutation2
    model_type = args.model

    data_folder = os.path.join('data', dataset_name)
    results_folder = os.path.join('results', dataset_name, "%s_%s"%(str(permutation1), str(permutation2)))
    os.makedirs(results_folder, exist_ok=True)

    if model_type == "NTA":
        # ## NTA 10/14
        cfg = Config(**{
            'input_size': 64,
            'output_size': 10,
            'num_seeds': 10,
            'num_contexts': 2,
            'num_paths': 2,
            'batch_size': 100,
            'W_lr': 0.001,
            'c_lr': 2,
            'initialization_scale': 0.01,
            'num_blocks': 10,
            'block_duration': 100,
            'regularization_strength': 0.1,
            'num_layers': 1,
            'hidden_size': 32,
            'context_model': True,
            'name': 'mnist_cfg',
            'regularization_type': ['gating_manifold_L1', 'nonnegative'],
            'shared_teachers': False,
            'c_gt_curriculum': 'A_B__',
            'dt': 0.1,
            'T_tape': -1,
            'dataset_name': args.dataset_name,
            'data_appendix': args.data_appendix,
            'permutation1': args.permutation1,
            'permutation2': args.permutation2,
            'data_folder': os.path.join('data', args.dataset_name),
            'results_folder': os.path.join('results', args.dataset_name, "%s_%s"%(str(args.permutation1), str(args.permutation2)))
        })
        
    elif model_type == "control":
        
        # CONTROL 10/14
        cfg = Config(**{
            'input_size': 64,
            'output_size': 10,
            'num_seeds': 10,
            'num_contexts': 2,
            'num_paths': 2,
            'batch_size': 100,
            'W_lr': 0.001,
            'c_lr': 0.001,
            'initialization_scale': 0.01,
            'num_blocks': 10,
            'block_duration': 100,
            'regularization_strength': np.finfo(float).eps,
            'num_layers': 1,
            'hidden_size': 32,
            'context_model': True,
            'name': 'mnist_cfg',
            'regularization_type': ['gating_manifold_L1', 'nonnegative'],
            'shared_teachers': False,
            'c_gt_curriculum': 'A_B__',
            'dt': 0.1,
            'T_tape': -1,
            'dataset_name': args.dataset_name,
            'data_appendix': args.data_appendix,
            'permutation1': args.permutation1,
            'permutation2': args.permutation2,
            'data_folder': os.path.join('data', args.dataset_name),
            'results_folder': os.path.join('results', args.dataset_name, "%s_%s"%(str(args.permutation1), str(args.permutation2)))
        })
    
    run_cfg(cfg, data_timestamp)

# %%
