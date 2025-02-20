from collections import namedtuple
import copy
from dataclasses import asdict
from functools import partial
import os
import pprint
import jax
from jax import lax
import jax.numpy as jnp
import jax.random as random
import numpy as np
from scipy.stats import ortho_group
import equinox as eqx
import logging
from dataclasses import asdict
from types import SimpleNamespace
from lcs.utils import compute_similarity, rdm_bunch_of_xx
from lcs.models import linear_model

logger = logging.getLogger(__name__)
cpu_cores = os.cpu_count()
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={max(1, cpu_cores - 2)}'

def get_tape_module(cfg, loading=False):
    """
    Preallocate the tape with the right shapes. 
    """
    tape_var_specs = dict(
        loss = (), 
        t = (),
    )
    for l in range(cfg.num_layers):
        if cfg.control == '2_diag_mono' or cfg.control == 'N_diag_mono' or cfg.control=='deep_mono':
            tape_var_specs['W' + str(l+1)] = (cfg.layer_sizes[l+1], cfg.layer_sizes[l])
            tape_var_specs['norm_W' + str(l+1)] = (cfg.layer_sizes[l+1],)
            tape_var_specs['grad_norm_W' + str(l+1)] = (cfg.layer_sizes[l+1],)
        else:
            tape_var_specs['W' + str(l+1)] = (cfg.num_paths, cfg.layer_sizes[l+1], cfg.layer_sizes[l])
            tape_var_specs['norm_W' + str(l+1)] = (cfg.num_paths, cfg.layer_sizes[l+1])
            tape_var_specs['grad_norm_W' + str(l+1)] = (cfg.num_paths, cfg.layer_sizes[l+1])

        if cfg.control == 'N_diag_mono':
            tape_var_specs['c' + str(l+1)] = (cfg.num_paths*cfg.hidden_size,)
        elif cfg.control == 'c_hadamard':
            tape_var_specs['c' + str(l+1)] = (cfg.num_paths, cfg.output_size)
        else:
            tape_var_specs['c' + str(l+1)] = (cfg.num_paths,)

        tape_var_specs['c_gt' + str(l+1)] = ()
        tape_var_specs['norm_c' + str(l+1)] = ()
        tape_var_specs['grad_norm_c' + str(l+1)] = ()
        tape_var_specs[f'cos_sim{l+1}'] = (cfg.num_contexts, cfg.num_paths)
        tape_var_specs['SVD_similarity'] = (cfg.num_contexts, cfg.num_paths)
        tape_var_specs[f'context_alignment{l+1}'] = ()
        tape_var_specs['SVD_alignment'] = ()
        tape_var_specs[f'current_context'] = () 
        tape_var_specs[f'grad_c{l+1}'] = (cfg.num_paths,) if cfg.control != 'c_hadamard' else (cfg.num_paths, cfg.output_size)
        tape_var_specs[f'grad_W{l+1}'] = (cfg.num_paths, cfg.layer_sizes[l+1], cfg.layer_sizes[l])
        tape_var_specs[f'y'] = (cfg.output_size,)
        tape_var_specs['concat_cossim'] = ()
        tape_var_specs['concat_SVD'] = ()

    if cfg.control == '2_diag_mono' or cfg.control == 'N_diag_mono':
        tape_var_specs['W2'] = (cfg.layer_sizes[2], cfg.layer_sizes[1])
        tape_var_specs['norm_W2'] = (cfg.layer_sizes[2],)
        tape_var_specs['grad_norm_W2'] = (cfg.layer_sizes[2],)
        tape_var_specs['sorted_c_student'] = (cfg.num_paths, cfg.hidden_size,)

    if cfg.control == 'N_diag_mono' or cfg.control == 'deep_mono' or cfg.control == '2_diag_mono': 
        tape_var_specs['c_alignment'] = (cfg.num_contexts + cfg.num_shared_contexts,)
        tape_var_specs['SVD_c_alignment'] = (cfg.num_contexts + cfg.num_shared_contexts,)
        tape_var_specs['row_cossim'] = (cfg.num_contexts*cfg.hidden_size, cfg.num_paths*cfg.hidden_size,)
        tape_var_specs['row_SVD'] = (cfg.num_contexts*cfg.hidden_size, cfg.num_paths*cfg.hidden_size,)
        tape_var_specs['sorted_cossim'] = (cfg.num_contexts, cfg.num_paths,)
        tape_var_specs['sorted_SVD'] = (cfg.num_contexts, cfg.num_paths,)

    if cfg.control == 'deep_mono':
        tape_var_specs['sorted_W2_student'] = (cfg.num_paths, cfg.output_size, cfg.hidden_size)

    if not loading: 
        tape_var_specs['W_teachers'] = (cfg.num_contexts, cfg.output_size, cfg.input_size)

    Tape = type('Tape', (eqx.Module,), {k: None for k in tape_var_specs.keys()} | {"__annotations__": {k: v for k, v in tape_var_specs.items()}} | {"__getitem__": lambda self, k: getattr(self, k)})

    ts_out = jnp.linspace(0, cfg.t_tot, cfg.T_tape, dtype=int)  
    tape_module = Tape(**{k: jnp.zeros((cfg.num_seeds, len(ts_out)) + v) for k, v in tape_var_specs.items()})
    return tape_module

def generate_teachers(d_in, d_out, n_contexts, seed=567456245, seed_U=None, seed_V=None, mode="orthogonal", svd_kwargs=None, return_svd=False, xx=0., scale=None, rotate=True):
    key = random.key(seed)
    if seed_U is None: seed_U = seed
    if seed_V is None: seed_V = seed

    """
    We can choose teachers W to have SVs of scale 1 / sqrt(d_in). 
    This will with be compatible with the natural scaling of 
    x_i = O(1)
    y_j = O(1)

    as

    y_j y_j = \sum_i,i' W_ij x_i x_i' W_i'j 
            = \sum_i,i' W_ij W_i'j \delta_{ii'}  (Gaussian iid x_i)
            = \sum_i W_ij W_ij
            = \sum_i W_ij^2
            = \sum_i 1/d_in
            = O(1)

    Alternatively, we may choose the teachers to have SVs of scale 1.
    """

    n_SVs = min(d_out, d_in)
    svd_kwargs_dflt = dict(
        U="ortho,",
        V="ortho,",
        rotate=rotate,
    )

    if svd_kwargs is None:
        svd_kwargs = dict()
    svd_kwargs = svd_kwargs_dflt | svd_kwargs  # update defaults with passed kwargs

    if scale is None:
        scale = 'd_in'
    if "unit" in scale.lower():
        scale = 1.0  # s_alpha = O(1)
    elif "d_in" in scale.lower():
        scale = 1.0 / d_in**0.5  # for y_j = O(1), x_i = O(1)
    elif "num_sv" in scale.lower():
        num_nonzero_sv_per_teacher = min(d_in, d_out) // n_contexts
        scale = 1.0 / num_nonzero_sv_per_teacher**0.5 
    else:
        raise ValueError

    def ortho_vecs(d, seed=seed, rotate=rotate):
        if xx == 0.:
            basis = jnp.eye(d)
        else:
            basis = rdm_bunch_of_xx(N=d, K=d, xx=xx)

        # random rotation matrix
        R = ortho_group.rvs(d, random_state=seed) if d > 1 else jnp.eye(1).reshape(1, 1)
        if rotate:
            basis = jnp.einsum('ij,bj->bi', R, basis)  # b is the dimension enumerating the basis
        return basis

    if "orthogonal" in mode.lower():
        assert d_in >= n_contexts, "d_in must be larger than n_contexts"
        basis = ortho_vecs(d_in)

        basis = basis[:n_contexts * d_out]
        basis *= scale

        # the resulting SVs will *not* have scale, but rather be smaller by a factor of sqrt(d_in)

        # split along the rows
        W = jnp.split(basis[:d_out*n_contexts], n_contexts, axis=0)
        W = jnp.array(W)  # wrap back into ndarray

        # NOTE that in contrast to the svd mode, here each path has its own SV-basis and hence is full_rank

        # some expensive test code, can be commented out
        if xx == 0.:
            for p1 in range(n_contexts):
                for p2 in range(n_contexts):
                    if p1 != p2:
                        assert jnp.isclose(jnp.trace(W[p1].T @ W[p2]), 0, atol=1e-5), "not orthogonal"
        U, S, VT = jnp.linalg.svd(W)
    elif "iid" in mode.lower():
        W = random.normal(key, (n_contexts, d_out, d_in)) * scale
        U, S, VT = jnp.linalg.svd(W)
    elif "svd" in mode.lower():
        # generate joint "feature detectors VT"

        key, key_v, key_u, key_s = random.split(key, 4)

        # 1. U SINGULAR VECTORS
        if "ortho" in svd_kwargs['U'].lower():
            UT = ortho_vecs(d_out, seed_U, svd_kwargs['rotate'])
        elif "random" in svd_kwargs['U'].lower():
            UT = random.normal(key_u, (n_contexts, d_out, d_out))
        else:
            raise ValueError
        UT = UT / jnp.linalg.norm(UT, axis=-1, keepdims=True)

        if 'share' in svd_kwargs['U'].lower():
            UT = jnp.repeat(UT[None, :, :], n_contexts, axis=0)
        else:
            UT = jnp.array(jnp.split(UT, n_contexts, axis=0))  # (n_context, vecs_per_context, d_out)

        assert UT.shape[0] == n_contexts
        U = jnp.swapaxes(UT, -1, -2)

        # 2. V SINGULAR VECTORS
        if "ortho" in svd_kwargs['V'].lower():
            VT = ortho_vecs(d_in, seed_V, svd_kwargs['rotate'])
        elif "random" in svd_kwargs['V'].lower():
            VT = random.normal(key_v, (n_contexts, d_in, d_in))
        else:
            raise ValueError
        VT = VT / jnp.linalg.norm(VT, axis=-1, keepdims=True)

        if 'share' in svd_kwargs['V'].lower():
            VT = jnp.repeat(VT[None, :, :], n_contexts, axis=0)
        else:
            VT = jnp.array(jnp.split(VT, n_contexts, axis=0))  # (n_context, vecs_per_context, d_out)

        V = jnp.swapaxes(VT, -1, -2)
        assert VT.shape[0] == n_contexts

        # 3. SYNTHESIZE U and VT with an S

        # generate joint singular values
        # for now this is the only mode implemented
        # S = random.uniform(key_s, (1, d_out)) * sv_scale
        S = scale * jnp.full((min(d_in, d_out),), 1.0)
        n_SVs = min(U.shape[-1], V.shape[-1], S.shape[-1])
        S = S[None][:1].repeat(n_contexts, axis=0)  
        S = S.at[:, n_SVs:].set(0.)

        assert jnp.isclose(jnp.linalg.norm(UT, axis=-1), 1).all(), "SV U not normalized"
        assert jnp.isclose(jnp.linalg.norm(VT, axis=-1), 1).all(), "SV VT not normalized"

        # orthogonality
        for p1 in range(n_contexts):
            for p2 in range(n_contexts):
                if p1 != p2:
                    if 'u,' in mode.lower():
                        assert jnp.isclose(jnp.trace(U[p1].T @ U[p2]), 0, atol=1e-5), "not orthogonal"
                    elif 'v,' in mode.lower():
                        assert jnp.isclose(jnp.trace(VT[p1] @ VT[p2].T), 0, atol=1e-5), "not orthogonal"

        # assemble the teachers
        logger.info(f"Generated teachers with mode {mode} and n_SVs {n_SVs} per context, Up,Sp,VTp={U[0, :, :n_SVs].shape},{S[0, :n_SVs].shape},{VT[0, :n_SVs].shape}")

        W = [jnp.einsum('ia,a,ak->ik', U[p, ..., :n_SVs], S[p, :n_SVs], VT[p, :n_SVs]) for p in range(n_contexts)]
        W = jnp.array(W)
    else: 
        raise ValueError

    if not return_svd:
        return W
    else:
        return W, (U[:, :, :n_SVs], S[:, :n_SVs], VT[:, :n_SVs]) 

# Define the loss function
def regularization_loss_fn(params, cfg):
    c = params['c1'].reshape(-1)
    W = params['W1']
    if cfg.control == 'deep_mono' and cfg.num_layers >= 2:
        c = params['W2'].flatten()

    sum_ = 0
    if type(cfg.regularization_type) is not list:
        regularization = [cfg.regularization_type]
    else:
        regularization = cfg.regularization_type
    
    reg_weight_sum = 0
    for i, reg in enumerate(regularization):
        if type(reg) is not tuple:
            regularization[i] = (reg, 1.)  # compatibility, default weight 1
        else:
            if len(reg) == 1:
                regularization[i] = (reg[0], 1.)

        reg_weight_sum += regularization[i][1]

    for reg in regularization:
        reg_name = reg[0]
        reg_weight = (reg[1] / reg_weight_sum)  # relative strength of the regularization term scaled to [0,1]. Global scaling happens outisde this function 

        if reg_name == 'nonnegative':
            term = 10 * jnp.max(jnp.array([-c, c*0.]), axis=0).sum()  # strict constraint, so it has to be quite strong
        elif reg_name == 'gating_manifold_L1':
            term = 10*0.5*jnp.abs(jnp.linalg.norm(c, ord=1, axis=-1) - 1)**2
        elif reg_name == 'gating_manifold_L2':
            term = 0.5*jnp.abs(jnp.linalg.norm(c, ord=2, axis=-1) - 1)**2
        elif reg_name == 'W_norm':
            term = jnp.linalg.norm(W)
        elif (("w" in reg_name.lower()) or ("c" in reg_name.lower())) and ("l" in reg_name.lower()):
            reg_name = reg_name.lower()
            idx_L = reg_name.index("l")
            p = int(reg_name[idx_L+1:idx_L+2])
            if 'w' in reg_name:
                term = 1 * 0.5*jnp.mean(jnp.linalg.norm(W, ord=p, axis=-2))
            if 'c' in reg_name:
                term = 1 * 0.5*jnp.mean(jnp.linalg.norm(c, ord=p, axis=-1))
        elif (reg_name.lower() == 'none') or (reg_name is None):
            term = 0.
        else:
            raise ValueError
        
        term *= reg_weight
        sum_ += term

    return sum_


def loss_fn(params, Y_pred, Y, cfg, reg_on):

    """
    regluarization_type: str or list of str or tuple
    tuple is (reg_name, reg_weight, reg_order)
    reg_weight is the relative weight of the regularization term, which is auto-normalized
    """
    
    reg_loss = lax.select((reg_on), regularization_loss_fn(params, cfg), 0.)
    error_summed_avg = jnp.sum((Y_pred - Y)**2, axis=-1).mean(0)  # ! scales with the output dimension
    return 1/2 * error_summed_avg + cfg.regularization_strength * reg_loss  # sum over output dimensions, mean over batch

def simulate_(model, init_key, tape, W_teachers, cfg, params_init=None, record_grads=False):
    # initialization
    params_init_ = dict()
    init_key, *init_keys_layers = random.split(init_key, cfg.num_layers + 1)
    init_keys_layers_c = random.split(init_key, cfg.num_layers + 1)
    std_c = 0.0
    for l in range(cfg.num_layers):
        if cfg.control == 'N_diag_mono':
            params_init_['c' + str(l+1)] = jnp.full((cfg.num_paths*cfg.hidden_size,), 0.5) 
        elif cfg.control == 'c_hadamard':
            params_init_['c' + str(l+1)] = jnp.full((cfg.num_paths, cfg.output_size), 0.5) 
        else:
            params_init_['c' + str(l+1)] = jnp.full((cfg.num_paths,), 0.5) 

        # here we use initialization that keeps y_i = W_ij x_j << 1 to allow for the W to grow
        std = cfg.initialization_scale / cfg.input_size**0.5
        if cfg.control == '2_diag_mono' or cfg.control == 'N_diag_mono' or cfg.control == 'deep_mono':
            params_init_['W' + str(l+1)] = std*random.normal(init_keys_layers[l], (cfg.layer_sizes[l+1], cfg.layer_sizes[l]))
        else:
            params_init_['W' + str(l+1)] = std*random.normal(init_keys_layers[l], (cfg.num_paths, cfg.layer_sizes[l+1], cfg.layer_sizes[l]))

    if params_init is None:
        params_init = dict()
    params_init = params_init_ | params_init  # update the defaults with passed

    learning_rates = dict()
    for k, v in params_init.items():
        if cfg.control == 'deep_mono' and 'W' in k:
            if 'W1' in k:
                learning_rates[k] = cfg.W1_tau ** -1 if cfg.W1_tau else None
            elif 'W2' in k:
                learning_rates[k] = cfg.W2_tau ** -1 if cfg.W2_tau else None
        elif 'W' in k:
            learning_rates[k] = cfg.W_tau ** -1 if cfg.W_tau else None
        elif 'c' in k:
            learning_rates[k] = cfg.c_tau ** -1 if cfg.c_tau else None
        else:
            raise ValueError
    
    ts_out = jnp.linspace(0, cfg.t_tot, cfg.T_tape)
    dt_tape = jnp.diff(ts_out)[0]
    dt = cfg.dt

    def loop_body(tape, params, ti, ti_tape, c_gt, data_key):  # in-out for each loop iteration (c_gt needs not be returned though)
        
        # INPUT
        data_key = random.split(data_key)[0]
        if cfg.use_X == True:
            X = random.normal(data_key, (cfg.batch_size, cfg.input_size))
        elif cfg.use_X == False: 
            X = jnp.ones((cfg.batch_size, cfg.input_size))
        elif cfg.use_X == 'eye':
            X = jnp.eye(cfg.input_size)  # this will amount to a gradient that affects both contexts
        else:
            raise ValueError
        
        # PREDICTION
        Y_pred = model(X, params, cfg)
    
        # LABEL
        if not hasattr(cfg.Y_tgt, '__call__'):
            if hasattr(cfg.c_gt_curriculum, '__call__'):
                c_gt_vec = cfg.c_gt_curriculum(ti*dt)  # vector of form (.9, .1) that sums to one
                c_vals = jnp.arange(cfg.num_contexts + cfg.num_shared_contexts)
                c_gt = (c_gt_vec / c_gt_vec.sum() * c_vals).sum()  # will give float in [0, num_contexts-1]
            else:
                c_gt_vec = jnp.eye(cfg.num_contexts + cfg.num_shared_contexts)[c_gt]
                    
            # picks out current teacher as a weighted sum
            current_teacher = jnp.einsum("p,pij->ij", c_gt_vec, W_teachers)  
            Y_tgt = jnp.einsum("ij,bj->bi", current_teacher, X)

        else:
            Y_tgt = cfg.Y_tgt(ti*dt)
            c_gt = jnp.argmax(Y_tgt, axis=-1).mean()
            current_teacher = cfg.Y_tgt(ti*dt).reshape(cfg.output_size, 1)

        reg_on = True
        if cfg.turn_off_regularization:
            reg_on = lax.select((ti < 0.5*(1/cfg.dt)*(cfg.num_blocks-1)), True, False)

        # Compute the loss
        loss_fn_X = lambda params, X, Y_tgt, cfg, reg_on: loss_fn(params, model(X, params, cfg), Y_tgt, cfg, reg_on)  # need to redefine this here to take gradient wrt model
        loss = loss_fn_X(params, X, Y_tgt, cfg, reg_on)

        # GRADIENTS
        loss_grad = jax.grad(loss_fn_X, 0)
        grads = loss_grad(params, X, Y_tgt, cfg, reg_on)
        W = params['W1'][0]

        T_tape = cfg.T_tape

        if cfg.control == '2_diag_mono':
            W2 = jnp.zeros((cfg.output_size, cfg.num_paths*cfg.hidden_size))
            for p in range(cfg.num_paths):
                for i in range(min(cfg.output_size, cfg.hidden_size)):
                    W2 = W2.at[i,i+p*cfg.hidden_size].set(params['c1'][p])
        elif cfg.control == 'N_diag_mono':
            W2 = jnp.zeros((cfg.output_size, cfg.num_paths*cfg.hidden_size))
            c_idx = 0
            for p in range(cfg.num_paths):
                for i in range(min(cfg.output_size, cfg.hidden_size)):
                    W2 = W2.at[i,i+p*cfg.hidden_size].set(params['c1'][c_idx])
                    c_idx+=1
        else:
            pass

        # LOGGING
        def write_to_tape(tape, ti_tape, t, loss, params, grads, c_gt, W_teachers):
            tape = eqx.tree_at(lambda tape: tape.t, tape, tape.t.at[ti_tape].set(t))
            for l in range(cfg.num_layers):
                tape = eqx.tree_at(lambda tape: getattr(tape, f"c{l+1}"), tape,  getattr(tape, f"c{l+1}").at[ti_tape].set(params[f'c{l+1}']))
                tape = eqx.tree_at(lambda tape: getattr(tape, f"W{l+1}"), tape,  getattr(tape, f"W{l+1}").at[ti_tape].set(params[f'W{l+1}']))

                tape = eqx.tree_at(lambda tape: getattr(tape, f"grad_norm_c{l+1}"), tape,  getattr(tape, f"grad_norm_c{l+1}").at[ti_tape].set(jnp.linalg.norm(grads[f'c{l+1}'])))
                tape = eqx.tree_at(lambda tape: getattr(tape, f"grad_norm_W{l+1}"), tape,  getattr(tape, f"grad_norm_W{l+1}").at[ti_tape].set(jnp.linalg.norm(grads[f'W{l+1}'])))

                if record_grads:
                    tape = eqx.tree_at(lambda tape: getattr(tape, f"grad_c{l+1}"), tape,  getattr(tape, f"grad_c{l+1}").at[ti_tape].set(grads[f'c{l+1}']))
                    tape = eqx.tree_at(lambda tape: getattr(tape, f"grad_W{l+1}"), tape,  getattr(tape, f"grad_W{l+1}").at[ti_tape].set(grads[f'W{l+1}']))

                tape = eqx.tree_at(lambda tape: getattr(tape, f"norm_c{l+1}"), tape,  getattr(tape, f"norm_c{l+1}").at[ti_tape].set(jnp.linalg.norm(params[f'c{l+1}'])))
                tape = eqx.tree_at(lambda tape: getattr(tape, f"norm_W{l+1}"), tape,  getattr(tape, f"norm_W{l+1}").at[ti_tape].set(jnp.linalg.norm(params[f'W{l+1}'])))

                tape = eqx.tree_at(lambda tape: getattr(tape, f"y"), tape,  getattr(tape, f"y").at[ti_tape].set(Y_pred.mean(0)))
                tape = eqx.tree_at(lambda tape: getattr(tape, f"c_gt{l+1}"), tape,  getattr(tape, f"c_gt{l+1}").at[ti_tape].set(c_gt))

                tape = eqx.tree_at(lambda tape: getattr(tape, "current_context"), tape,  getattr(tape, "current_context").at[ti_tape].set(c_gt))
                
                if cfg.context_model or cfg.control == "c_hadamard": 
                    W_students_ = params['W'+ str(l+1)]
                    current_student = jnp.sum(jnp.einsum("p,pij->pij" if cfg.control != "c_hadamard" else 'pi,pij->pij', params[f'c{l+1}'], params[f'W{l+1}']), axis=0)
                    cos_sim = compute_similarity(W_teachers[:cfg.num_contexts][:,None], W_students_[None, :], metric=cfg.metric)
                    tape = eqx.tree_at(lambda tape: getattr(tape, f"cos_sim{l+1}"), tape,  getattr(tape, f"cos_sim{l+1}").at[ti_tape].set(cos_sim))
                    
                    if cfg.use_X: # not toy model
                        context_alignment = compute_similarity(current_teacher, current_student, metric='cosine')
                        tape = eqx.tree_at(lambda tape: getattr(tape, f"context_alignment{l+1}"), tape,  getattr(tape, f"context_alignment{l+1}").at[ti_tape].set(context_alignment))
                        
                        if cfg.log_aux:
                            SVD_alignment = compute_similarity(current_teacher, current_student, metric='SVD')
                            tape = eqx.tree_at(lambda tape: getattr(tape, "SVD_alignment"), tape,  getattr(tape, "SVD_alignment").at[ti_tape].set(SVD_alignment))

                            SVD_similarity = compute_similarity(W_teachers[:cfg.num_contexts][:,None], W_students_[None, :], metric='SVD')
                            tape = eqx.tree_at(lambda tape: getattr(tape, "SVD_similarity"), tape,  getattr(tape, "SVD_similarity").at[ti_tape].set(SVD_similarity))
                            W_teachers_ = W_teachers[:cfg.num_contexts].reshape(cfg.num_contexts*cfg.output_size, cfg.input_size)
                            W_student = []
                            for c in range(cfg.num_contexts): # sorting for concatenated cossimilarity metric
                                if cfg.teacher_mode == 'svd':
                                    path = jnp.argmax(SVD_similarity[c])
                                else:
                                    path = jnp.argmax(cos_sim[c])
                                W_student.append(params['W1'][path])
                        
                            W_student = jnp.array(W_student).reshape(cfg.num_paths*cfg.output_size, cfg.input_size)
                            concat_cossim = compute_similarity(W_teachers_, W_student, metric='cosine')
                            tape = eqx.tree_at(lambda tape: getattr(tape, "concat_cossim"), tape,  getattr(tape, "concat_cossim").at[ti_tape].set(concat_cossim))
                            concat_SVD = compute_similarity(W_teachers_, W_student, metric='SVD')
                            tape = eqx.tree_at(lambda tape: getattr(tape, "concat_SVD"), tape,  getattr(tape, "concat_SVD").at[ti_tape].set(concat_SVD))
                
                elif hasattr(cfg.c_gt_curriculum, '__call__') and not cfg.control == "deep_mono":  # TODO MNIST flag
                    W_students_ = params['W'+ str(l+1)]
                    rectify = (lambda x: jnp.abs(x))
                    cos_sim = rectify(compute_similarity(W_teachers[:, None], W_students_[None, :], metric=cfg.metric))
                    tape = eqx.tree_at(lambda tape: getattr(tape, f"cos_sim{l+1}"), tape,  getattr(tape, f"cos_sim{l+1}").at[ti_tape].set(cos_sim))
                    
                    current_student = jnp.sum(params[f'W{l+1}'],axis=0)
                    
                    context_alignment = (jnp.abs(jnp.einsum("ij,ij->i", current_teacher, current_student)) / (jnp.linalg.norm(current_teacher, axis=-1) * jnp.linalg.norm(current_student, axis=-1))).max() 
                    tape = eqx.tree_at(lambda tape: getattr(tape, f"context_alignment{l+1}"), tape,  getattr(tape, f"context_alignment{l+1}").at[ti_tape].set(context_alignment))

            if cfg.control == 'N_diag_mono' or cfg.control == 'deep_mono' or cfg.control =='2_diag_mono': 
                W_student = params['W1']
                if cfg.control == 'N_diag_mono' or cfg.control == '2_diag_mono':
                    current_student = jnp.einsum("ij,jk->ik", W2, params['W1'])   
                else: 
                    current_student = jnp.einsum("ij,jk->ik", params['W2'], params['W1'])
                context_alignment = compute_similarity(current_teacher, current_student, metric='cosine')
                tape = eqx.tree_at(lambda tape: getattr(tape, f"context_alignment{1}"), tape,  getattr(tape, f"context_alignment{1}").at[ti_tape].set(context_alignment))
                SVD_alignment = compute_similarity(current_teacher, current_student, metric='SVD')
                tape = eqx.tree_at(lambda tape: getattr(tape, "SVD_alignment"), tape,  getattr(tape, "SVD_alignment").at[ti_tape].set(SVD_alignment))
                    

                cossims_per_row = jnp.zeros((cfg.num_paths*cfg.hidden_size, cfg.num_contexts*cfg.hidden_size))
                SVD_per_row = jnp.zeros((cfg.num_paths*cfg.hidden_size, cfg.num_contexts*cfg.hidden_size))
                for i in range(W_student.shape[0]): # row of W1
                    for j in range(cfg.num_contexts): # teacher
                        for k in range(W_teachers[j].shape[0]): # row of teacher
                            cos_sim = compute_similarity(W_teachers[j][k][None,...], W_student[i][None,...], metric='cosine')
                            cossims_per_row = cossims_per_row.at[i,k+(j*cfg.hidden_size)].set(cos_sim)
                            SVD_sim = compute_similarity( W_teachers[j][k][None,...], W_student[i][None,...], metric='SVD')
                            SVD_per_row = SVD_per_row.at[i,k+(j*cfg.hidden_size)].set(SVD_sim)

                tape = eqx.tree_at(lambda tape: getattr(tape, f"row_cossim"), tape,  getattr(tape, f"row_cossim").at[ti_tape].set(cossims_per_row))
                tape = eqx.tree_at(lambda tape: getattr(tape, f"row_SVD"), tape,  getattr(tape, f"row_SVD").at[ti_tape].set(SVD_per_row))

            
            if cfg.control == '2_diag_mono' or cfg.control == 'N_diag_mono':
                tape = eqx.tree_at(lambda tape: getattr(tape, f"W{2}"), tape,  getattr(tape, f"W{2}").at[ti_tape].set(W2))
                tape = eqx.tree_at(lambda tape: getattr(tape, f"norm_W{2}"), tape,  getattr(tape, f"norm_W{2}").at[ti_tape].set(jnp.linalg.norm(W2)))

            
            loss_without_reg = 1/2 * jnp.mean(jnp.sum((Y_pred - Y_tgt)**2, axis=-1), axis=0)  # sum over output dimensions, mean over batch)
            ## ---- Ignoring regularization term in the saved loss ----
            tape = eqx.tree_at(lambda tape: getattr(tape, f"loss"), tape,  getattr(tape, f"loss").at[ti_tape].set(loss_without_reg))

            return tape, ti_tape + 1
        
        # log whenever t becomes larger than the next time point in the tape
        tape, ti_tape = lax.cond((ti*dt >= ti_tape*dt_tape) & (ti_tape < T_tape), lambda _: write_to_tape(tape, ti_tape, ti*dt, loss, params, grads, c_gt, W_teachers), lambda _: (tape, ti_tape), None)

        # Update c and W
        params_n = jax.tree_util.tree_map(lambda param, lr, grad: param - dt * lr * grad, 
                                        params, learning_rates, grads)
            
        ti += 1
        
        return tape, params_n, ti, ti_tape, c_gt, data_key
    
    def c_gt_(block_step): 
        switch_point = int(0.75*cfg.num_blocks)
        if cfg.shared_teachers or cfg.shared_concat_teachers:
            if cfg.c_gt_curriculum == 'B_AB__A_B_AB__':
                c_gt = lax.select((block_step < switch_point), (block_step % cfg.num_contexts) + 1, block_step % (cfg.num_contexts + cfg.num_shared_contexts))
            elif cfg.c_gt_curriculum == 'B_AB__A_B__':
                c_gt = lax.select((block_step < switch_point), (block_step % cfg.num_contexts) + 1, block_step % (cfg.num_contexts))
            elif cfg.c_gt_curriculum == 'A_B__AB__':
                c_gt = lax.select(((block_step < switch_point)), (block_step % cfg.num_contexts), 2) 
            elif cfg.c_gt_curriculum == 'AB_BC__':
                c_gt = (block_step % (cfg.num_contexts-1)) + cfg.num_contexts 
            elif cfg.c_gt_curriculum == 'A_B_C__AB_BC_CA__':
                c_gt = lax.select(((block_step < switch_point)), (block_step % cfg.num_contexts), (block_step % cfg.num_contexts)+cfg.num_contexts) 
            elif cfg.c_gt_curriculum == 'AB_BC_CA__':
                c_gt = ((block_step % (cfg.num_contexts)) + cfg.num_contexts)
            elif cfg.c_gt_curriculum == 'AB_BC_CA__A_B_C__':
                c_gt = lax.select(((block_step < switch_point)), (block_step % cfg.num_contexts)+cfg.num_contexts, (block_step % cfg.num_contexts)) 
            elif cfg.c_gt_curriculum == 'AB_CD__AD__': 
                c_gt = lax.select((block_step < switch_point), lax.select(((block_step % 2)==0), 4, 6), 7) 
            elif cfg.c_gt_curriculum == 'AB_BC_CD_DA__AC_BD__': 
                c_gt = lax.select(((block_step < switch_point)), (block_step % cfg.num_contexts) + cfg.num_contexts, (block_step % 2) + 2*cfg.num_contexts) 

        if cfg.c_gt_curriculum == 'A_B__':
            c_gt = block_step % cfg.num_contexts
        return c_gt

    def step(carry, t2):
        carry = loop_body(*carry)
        return carry, None

    # Initialize your variables (tape, params, ti, ti_tape) appropriately
    init_data_key, init_key = random.split(init_key)
    initial_carry = (tape, params_init, 0, 0, 0, init_data_key)

    # Using scan for both outer and inner loops
    if not hasattr(cfg.c_gt_curriculum, '__call__'):
        
        # iterate over a complete blocks
        def outer_step(carry, block_step):
            tape, params, ti, ti_tape, c_gt, data_key = carry
            c_gt = c_gt_(block_step) if cfg.Y_tgt is None else 0
            initial_carry_inner = (tape, params, ti, ti_tape, c_gt, data_key)
            final_carry_inner, _ = jax.lax.scan(step, initial_carry_inner, jnp.arange(cfg.block_duration // cfg.dt))
            return final_carry_inner, None
        
        block_steps = jnp.arange(cfg.num_blocks + 1)
        #outer_step = scan_tqdm(len(block_steps))(outer_step)  # prev
        final_carry, _ = jax.lax.scan(outer_step, initial_carry, block_steps)
        tape, params, ti, ti_tape, c_gt, data_key = final_carry
    else:
        ## take a single step
        #step = scan_tqdm((cfg.T_tot + 1))(step)  # prev
        final_carry, _ = jax.lax.scan(step, initial_carry, jnp.arange(cfg.T_tot + 1))
        tape, params, ti, ti_tape, c_gt, data_key = final_carry

    return tape

def simulate(model, tape, W_teachers, cfg, params_init=None, record_grads=False):
    # parallelize the operation
    init_keys = random.split(random.key(8897), cfg.num_seeds)
    # make a named tuple out of cfg's dict
    cfg_dict = cfg.__dict__
    MyNamedTuple = namedtuple('Config', cfg_dict.keys())
    cfg_tuple = MyNamedTuple(**cfg_dict)

    device = jax.devices()[0]
    logger.info(f'Running {cfg.num_seeds} seeds on device {device} with config {pprint.pformat(cfg)} and dtype {jax.tree_flatten(params_init)[0][0].dtype if params_init is not None else jnp.array(1.).dtype}')
    
    # only if the init parameters have a batch axis – we try to infer this from the parameter shapes
    c_shape_atomic = tape['c1'].shape[2:] # strips batch and time
    vmap_dim_params = None if not params_init or (params_init['c1'].shape == c_shape_atomic) else 0
    vmap_dim_W_t = None if W_teachers is None or W_teachers.ndim == 3 else 0 

    simulate_vbatch_seeds = jax.vmap(partial(simulate_, record_grads=record_grads), 
                    in_axes=(None, 0, 0, vmap_dim_W_t, None, vmap_dim_params))
    
    with jax.disable_jit(False):
        tape = simulate_vbatch_seeds(model, init_keys, tape, W_teachers, cfg_tuple, params_init)  # does this iterate over multiple seeds?

    if hasattr(W_teachers, 'shape'):
        tape = eqx.tree_at(lambda tape: tape.W_teachers, tape, W_teachers if W_teachers.ndim == 4 else W_teachers[None, ...])  # (B, M, O, I)
    return tape


def sort_it(tape, cfg):
    tape = jax.tree.map(lambda x: x[None], tape) if tape.t.ndim == 1 else tape
    tape = jax.tree.map(lambda x: np.array(x), tape)  # to make arrays mutable

    # assign the right context because of the permutation degeneracy
    tape = asdict(tape)
    tape2 = copy.deepcopy(tape)

    if cfg.num_paths > 1:
        for i_tr in range(tape['t'].shape[0]):
                for l in range(cfg.num_layers):
                    for p in range(cfg.num_paths):
                        if cfg.teacher_mode == 'orthogonal':
                            matched_context = np.argmax(tape[f'cos_sim{l+1}'][i_tr, :, :, p].mean(0), axis=-1)
                        elif ('svd' in cfg.teacher_mode.lower()) or ('iid' in cfg.teacher_mode.lower()):
                            matched_context = np.argmax(tape['SVD_similarity'][i_tr, :, :, p].mean(0), axis=-1)
                        if p != matched_context and p < cfg.num_contexts:
                            tape2[f'cos_sim{l+1}'][i_tr, :, p, p] = tape[f'cos_sim{l+1}'][i_tr, :, matched_context, p]
                            tape2[f'cos_sim{l+1}'][i_tr, :, matched_context, p] = tape[f'cos_sim{l+1}'][i_tr, :, p, p]
                            tape2['SVD_similarity'][i_tr, :, p, p] = tape['SVD_similarity'][i_tr, :, matched_context, p]
                            tape2['SVD_similarity'][i_tr, :, matched_context, p] = tape['SVD_similarity'][i_tr, :, p, p]
                            tape2[f'c{l+1}'][i_tr, :, p] = tape[f'c{l+1}'][i_tr, :, matched_context]
                            tape2[f'c{l+1}'][i_tr, :, matched_context] = tape[f'c{l+1}'][i_tr, :, p]
                            tape2[f'W{l+1}'][i_tr, :, p] = tape[f'W{l+1}'][i_tr, :, matched_context]
                            tape2[f'W{l+1}'][i_tr, :, matched_context] = tape[f'W{l+1}'][i_tr, :, p]
                            tape2[f'norm_W{l+1}'][i_tr, :, p] = tape[f'norm_W{l+1}'][i_tr, :, matched_context]
                            tape2[f'norm_W{l+1}'][i_tr, :, matched_context] = tape[f'norm_W{l+1}'][i_tr, :, p]

    Tape = type('Tape', (eqx.Module,), {k: None for k in tape2.keys()} | {"__annotations__": {k: None for k, v in tape2.items()}} | {"__getitem__": lambda self, k: getattr(self, k)})

    tape2 = Tape(**tape2)
    return tape2

def remove_teachers(tape):
    tape = asdict(tape)
    del tape['W_teachers']
    Tape = type('Tape', (eqx.Module,), {k: None for k in tape.keys()} | {"__annotations__": {k: None for k, v in tape.items()}} | {"__getitem__": lambda self, k: getattr(self, k)})
    tape = Tape(**tape)
    return tape

def compute_monolithic_alignment(tape, cfg, W_teachers):
    tape = asdict(tape)
    C_star = list()
    trained_contexts = np.unique(tape['current_context'])
    for i in range(len(trained_contexts)):
        last_c_idx = jnp.where(tape['current_context'][0] == trained_contexts[i])[0][-1]
        C_star.append(tape['W2'][:,last_c_idx])
    C_star = jnp.array(C_star)  
    C_star = jnp.swapaxes(C_star, 0, 1) # has shape (time, batch, ...) -> so we swap for consistency
    

    def set_alignment(current_context, W_teachers, C_star, current_student, c_alignment, SVD_c_alignment):
        """
        After this function is vectorized, it sees inputs that lack the batch and time axes
        """
        current_teacher = W_teachers[(current_context).astype(int)]
        for i in range(len(trained_contexts)):
            ctx_student = jnp.einsum("ij,jk->ik", C_star[i], current_student)
            ctx_align = compute_similarity(current_teacher[:,...],ctx_student[:,...], metric='cosine')
            c_alignment = c_alignment.at[i].set(ctx_align)
            SVD_align = compute_similarity(current_teacher[:,...], ctx_student[:,...], metric='SVD')
            SVD_c_alignment = SVD_c_alignment.at[i].set(SVD_align)

        return c_alignment, SVD_c_alignment
    
    c_alignment = jnp.zeros((cfg.num_seeds, tape['t'].shape[1], len(trained_contexts)))
    SVD_c_alignment = jnp.zeros((cfg.num_seeds, tape['t'].shape[1], len(trained_contexts)))
    
    # vmap over batch (0) and time (1) axes
    set_alignment_v = jax.vmap(set_alignment, in_axes=(0, 0, 0, 0, 0, 0,), out_axes=(0, 0))
    set_alignment_vv = jax.vmap(set_alignment_v, in_axes=(1, None, None, 1, 1, 1,), out_axes=(1, 1))  # dont map teachers over time
    c_alignment, SVD_c_alignment = set_alignment_vv(tape['current_context'], W_teachers, C_star, tape['W1'], c_alignment, SVD_c_alignment)

    tape['c_alignment'] = c_alignment
    tape['SVD_c_alignment'] = SVD_c_alignment
    Tape = type('Tape', (eqx.Module,), {k: None for k in tape.keys()} | {"__annotations__": {k: None for k, v in tape.items()}} | {"__getitem__": lambda self, k: getattr(self, k)})
    tape = Tape(**tape)
    return tape


def compute_row_cossim(tape, cfg, W_teachers, teacher_mode): 
    tape = asdict(tape)
    if teacher_mode == 'orthogonal':
        mean_row_cossim = jnp.mean(tape['row_cossim'], axis=1)
    elif ('svd' in teacher_mode.lower()) or ('iid' in teacher_mode.lower()):
        mean_row_cossim = jnp.mean(tape['row_SVD'], axis=1)
    student_idx = jnp.argmax(mean_row_cossim, axis=1)
    if cfg.control == 'deep_mono':
        sorted_W2_student = jnp.zeros((cfg.num_seeds, tape['t'].shape[1], cfg.num_paths, cfg.output_size, cfg.hidden_size))
    elif cfg.control == 'N_diag_mono':
        sorted_c_student = jnp.zeros((cfg.num_seeds, tape['t'].shape[1], cfg.num_paths, cfg.hidden_size))
    sorted_W_student = jnp.zeros((cfg.num_seeds, tape['t'].shape[1], cfg.num_paths, cfg.hidden_size, cfg.input_size))
    for s in range(cfg.num_seeds):
        for p in range(cfg.num_paths):
            for row in range(cfg.hidden_size):
                sorted_W_student = sorted_W_student.at[s,:,p,row].set(tape['W1'][s,:,student_idx[s][row + p*cfg.hidden_size]])
                if cfg.control == 'deep_mono':
                    sorted_W2_student = sorted_W2_student.at[s,:,p,:,row].set(tape['W2'][s,:,:,student_idx[s][row + p*cfg.hidden_size]])
                elif cfg.control == 'N_diag_mono':
                    sorted_c_student = sorted_c_student.at[s,:,p,row].set(tape['c1'][s,:,student_idx[s][row + p*cfg.hidden_size]])

    def sort_cossim(W_teachers, sorted_W_student, sorted_cossim, sorted_SVD_sim, concat_cossim, concat_SVD):
        for p in range(cfg.num_paths):
            for c in range(cfg.num_contexts):
                sort_cossim = compute_similarity(W_teachers[c][:,...], sorted_W_student[p], metric='cosine')
                sorted_cossim = sorted_cossim.at[c,p].set(sort_cossim)
                sort_SVD_sim = compute_similarity(W_teachers[c][:,...], sorted_W_student[p], metric='SVD')
                sorted_SVD_sim = sorted_SVD_sim.at[c,p].set(sort_SVD_sim)
    
        W_teachers_ = W_teachers[:cfg.num_contexts].reshape(cfg.num_contexts*cfg.output_size, cfg.input_size)
        W_student = sorted_W_student[...].reshape(cfg.num_paths*cfg.output_size, cfg.input_size)
        concat_cossim_ = compute_similarity(W_teachers_[:,...], W_student[:,...], metric='cosine')
        concat_cossim = concat_cossim.at[...].set(concat_cossim_)
        concat_SVD_ = compute_similarity(W_teachers_[:,...], W_student[:,...], metric='SVD')
        concat_SVD = concat_SVD.at[...].set(concat_SVD_) 

        return sorted_cossim, sorted_SVD_sim, concat_cossim, concat_SVD
    
    sorted_cossim = jnp.zeros((cfg.num_seeds, tape['t'][0].shape[0], cfg.num_contexts, cfg.num_paths))
    sorted_SVD_sim = jnp.zeros((cfg.num_seeds, tape['t'][0].shape[0], cfg.num_contexts, cfg.num_paths))
    concat_cossim = jnp.zeros((cfg.num_seeds, tape['t'][0].shape[0]))
    concat_SVD = jnp.zeros((cfg.num_seeds, tape['t'][0].shape[0]))

    # vmap over batch (0) and time (1) axes
    sort_cossim_v = jax.vmap(sort_cossim, in_axes=(0, 0, 0, 0, 0, 0), out_axes=(0, 0, 0, 0))
    sort_cossim_v = jax.vmap(sort_cossim_v, in_axes=(None, 1, 1, 1, 1, 1), out_axes=(1, 1, 1, 1)) 

    sorted_cossim, sorted_SVD_sim, concat_cossim, concat_SVD = sort_cossim_v(W_teachers, sorted_W_student, sorted_cossim, sorted_SVD_sim, concat_cossim, concat_SVD)     

    tape['sorted_cossim'] = sorted_cossim
    tape['sorted_SVD_sim'] = sorted_SVD_sim
    tape['concat_cossim'] = concat_cossim
    tape['concat_SVD'] = concat_SVD
    if cfg.control == 'deep_mono':
        tape['sorted_W2_student'] = sorted_W2_student
    elif cfg.control == 'N_diag_mono':
        tape['sorted_c_student'] = sorted_c_student
    Tape = type('Tape', (eqx.Module,), {k: None for k in tape.keys()} | {"__annotations__": {k: None for k, v in tape.items()}} | {"__getitem__": lambda self, k: getattr(self, k)})
    tape = Tape(**tape)

    return tape

def compute_concat_cossim(tape, cfg, W_teachers): 
    tape = asdict(tape)
    def sort_cossim(W_teachers, W_student, concat_cossim, concat_SVD):
        W_teachers_ = W_teachers[:cfg.num_contexts].reshape(cfg.num_contexts*cfg.output_size, cfg.input_size)
        W_student = W_student[...].reshape(cfg.num_paths*cfg.output_size, cfg.input_size)

        concat_cossim_ = compute_similarity(W_teachers_[:,...], W_student[:,...], metric='cosine')
        concat_cossim = concat_cossim.at[...].set(concat_cossim_)
        concat_SVD_ = compute_similarity(W_teachers_[:,...], W_student[:,...], metric='SVD')
        concat_SVD = concat_SVD.at[...].set(concat_SVD_) 

        return concat_cossim, concat_SVD
    
    concat_cossim = jnp.zeros((cfg.num_seeds, tape['t'][0].shape[0]))
    concat_SVD = jnp.zeros((cfg.num_seeds, tape['t'][0].shape[0]))

    # vmap over batch (0) and time (1) axes
    sort_cossim_v = jax.vmap(sort_cossim, in_axes=(0, 0, 0, 0), out_axes=(0, 0))
    sort_cossim_v = jax.vmap(sort_cossim_v, in_axes=(None, 1, 1, 1), out_axes=(1, 1)) 

    concat_cossim, concat_SVD = sort_cossim_v(W_teachers, tape['W1'], concat_cossim, concat_SVD)
        
    tape['concat_cossim'] = concat_cossim
    tape['concat_SVD'] = concat_SVD
    Tape = type('Tape', (eqx.Module,), {k: None for k in tape.keys()} | {"__annotations__": {k: None for k, v in tape.items()}} | {"__getitem__": lambda self, k: getattr(self, k)})
    tape = Tape(**tape)

    return tape

def run_config(cfg, return_teachers=False, return_cfg=False, params_init = None, sort=True):
    model = linear_model

    if not hasattr(cfg.W_teachers, 'shape') and (cfg.W_teachers == 'generate'):
        W_teachers = list()
        for s in range(cfg.num_seeds):
            orthogonal_teachers = generate_teachers(cfg.input_size, cfg.output_size, cfg.num_contexts, mode=cfg.teacher_mode, xx=cfg.teacher_xx, scale=cfg.teacher_scale, rotate=cfg.teacher_rotate)
            all_teachers = jnp.array(copy.deepcopy(orthogonal_teachers))
            if cfg.shared_teachers or cfg.shared_concat_teachers:
                for t in range(cfg.num_contexts):
                    if t == cfg.num_contexts-1 and cfg.num_contexts > 2:
                        if cfg.shared_concat_teachers:
                            shared_teacher = copy.deepcopy(orthogonal_teachers[t])
                            for row in range(shared_teacher.shape[0]):
                                if row % 2: # alternating teacher rows
                                    shared_teacher = shared_teacher.at[row].set(orthogonal_teachers[0][row])
                        else:
                            shared_teacher = cfg.mixing_factor*orthogonal_teachers[t] + (1-cfg.mixing_factor)*orthogonal_teachers[0]
                    elif t < cfg.num_contexts-1:
                        if cfg.shared_concat_teachers:
                            shared_teacher = copy.deepcopy(orthogonal_teachers[t])
                            for row in range(shared_teacher.shape[0]):
                                if row % 2: # alternating teacher rows
                                    shared_teacher = shared_teacher.at[row].set(orthogonal_teachers[t+1][row])
                        else:
                            shared_teacher = cfg.mixing_factor*orthogonal_teachers[t] + (1-cfg.mixing_factor)*orthogonal_teachers[t+1]
                    else:
                        shared_teacher = None
                    if shared_teacher is not None:
                        all_teachers = jnp.concatenate((all_teachers, shared_teacher.reshape(1, shared_teacher.shape[0], -1)), axis=0)
                if cfg.c_gt_curriculum == 'AB_BC_CD_DA__AC_BD__': 
                    if cfg.shared_concat_teachers:
                        shared_teacher1 = copy.deepcopy(orthogonal_teachers[0])
                        shared_teacher2 = copy.deepcopy(orthogonal_teachers[1])
                        for row in range(shared_teacher.shape[0]):
                            if row % 2: # alternating teacher rows
                                shared_teacher1 = shared_teacher1.at[row].set(orthogonal_teachers[2][row])
                                shared_teacher2 = shared_teacher2.at[row].set(orthogonal_teachers[3][row])
                    else:
                        shared_teacher1 = cfg.mixing_factor*orthogonal_teachers[0] + (1-cfg.mixing_factor)*orthogonal_teachers[2]
                        shared_teacher2 = cfg.mixing_factor*orthogonal_teachers[1] + (1-cfg.mixing_factor)*orthogonal_teachers[3]
                        all_teachers = jnp.concatenate((all_teachers, shared_teacher1.reshape(1, shared_teacher1.shape[0], -1), shared_teacher2.reshape(1, shared_teacher2.shape[0], -1)), axis=0)
            W_teachers.append(all_teachers)
        
        W_teachers = jnp.stack(jnp.array(W_teachers))
        cfg.num_shared_contexts = W_teachers.shape[1] - cfg.num_contexts

    elif hasattr(cfg.W_teachers, 'shape'):
        W_teachers = cfg.W_teachers
    else:
        raise ValueError
    
    # MAIN CALL: run the simulations over many seeds
    tape = get_tape_module(cfg)
    tape = simulate(model, tape, W_teachers, cfg, params_init=params_init)

    # do some checks
    for l in range(cfg.num_layers):
        max_grad_norm_W  = getattr(tape, f'grad_norm_W{l+1}').max()
        if cfg.W_tau is not None:
            if not max_grad_norm_W*cfg.W_tau**-1*cfg.dt <= 1e-1:
                logger.warning(f"grad norm W{l+1} x lr is too large at {max_grad_norm_W*cfg.W_tau**-1*cfg.dt}, not operating in Gradient Flow regime. Consider reducing dt={cfg.dt:.4f}. ")
        else:
            if not max_grad_norm_W*cfg.W1_tau**-1*cfg.dt <= 1e-1:
                logger.warning(f"grad norm W{l+1} x lr is too large at {max_grad_norm_W*cfg.W1_tau**-1*cfg.dt}, not operating in Gradient Flow regime. Consider reducing dt={cfg.dt:.4f}. ")

        max_grad_norm_c  = getattr(tape, f'grad_norm_c{l+1}').max()
        if not getattr(tape, f'grad_norm_c{l+1}').max()*cfg.c_tau**-1*cfg.dt <= 1e-1:
            logger.warning(f"grad norm c{l+1} x lr is too large at {max_grad_norm_c*cfg.c_tau**-1*cfg.dt}, not operating in Gradient Flow regime. Consider reducing dt={cfg.dt:4f}. ")

    tape = jax.tree_util.tree_map(lambda x: np.asarray(x), tape)
    if cfg.control == 'deep_mono' or cfg.control == 'N_diag_mono' or cfg.control=='2_diag_mono': 
        if cfg.c_gt_curriculum == 'AB_BC_CA__' or cfg.c_gt_curriculum == 'AB_CD__AD__':
            tape = SimpleNamespace(**asdict(tape))
        else:
            tape = compute_monolithic_alignment(tape, cfg, jnp.array(W_teachers))
        tape = compute_row_cossim(tape, cfg, jnp.array(W_teachers), cfg.teacher_mode)
    else:
        if sort:
            tape = sort_it(tape, cfg) 
            if cfg.log_aux:
                tape = compute_concat_cossim(tape, cfg, jnp.array(W_teachers))

    if return_teachers and return_cfg:
        tape = remove_teachers(tape)
        return tape, np.array(W_teachers), cfg
    elif return_teachers:
        return (tape, np.array(W_teachers))
    elif return_cfg:
        return tape, cfg
    return tape
