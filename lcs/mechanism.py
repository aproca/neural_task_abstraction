import matplotlib.pyplot as plt
from lcs.curricula import get_n_phasic_curriculum
from lcs.jax_utils import new_tape_type
from lcs.models import linear_model
plt.rcParams["animation.html"] = "jshtml"
import numpy as np
import jax
from lcs.joint_learning import loss_fn
jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
# set env var
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
from collections import namedtuple
from dataclasses import asdict
from lcs.joint_learning import get_tape_module, simulate
from lcs.configs import *
from lcs.joint_learning import get_tape_module
import logging

npa = np.asarray
logger = logging.getLogger('lcs')
logger.setLevel(logging.INFO)

toy_tape_fields = ['w11', 'w12', 'w21', 'w22', 'r', 'c1', 'c2']


def get_tape_1d(cfg):
    #ä 1D sim
    logger.info('Getting tape')
    Y_tgts = jnp.array([1, -1.]).reshape(2, 1,) # context, output
    Y_tgt = get_n_phasic_curriculum(g, cfg_toy_['block_duration'], *Y_tgts)
    cfg = Config(**dict(cfg_toy_,
                Y_tgt=Y_tgt, 
                output_size=1,
                ))
    
    params_init_full = dict(W1=jnp.array([.1, -.1]).reshape(2, 1, 1), c1=jnp.array([.5, .5]))
    tape_init = get_tape_module(cfg)

    tape_full = simulate(linear_model, tape=tape_init, W_teachers=Y_tgts.reshape(2, 1, 1), cfg=cfg, params_init=params_init_full)
    tape_full = jax.tree.map(lambda x: jnp.array(x).squeeze(0), tape_full)
    tape_full_dict = asdict(tape_full)


    tape_toy = dict(s1=tape_full['W1'][0, 0, 0], s2=tape_full['W1'][1, 0, 0], r=jnp.linalg.norm(tape_full['W1'], axis=-2).mean((-2, -1)), c_bar=jnp.mean(jnp.array([tape_full['c1'][0] - .5, -tape_full['c1'][1]  + .5]), axis=0))
    Tape = namedtuple('Tape', (tape_toy |  tape_full_dict).keys())
    tape = Tape(**(tape_toy |  tape_full_dict))
    logger.info('Got tape')
    return cfg, tape

# 2d simulation
def get_tape_2d(cfg, params_init_toy=None, full2toy=None, toy2full=None):

    tape_init = get_tape_module(cfg)
    params_init_full = toy2full(params_init_toy)
    tape_full = simulate(linear_model, tape=tape_init, W_teachers=cfg.W_teachers, cfg=cfg, params_init=params_init_full, record_grads=True)
    tape_full = jax.tree.map(lambda x: jnp.array(x).squeeze(0), tape_full)  # assert single seed
    tape_full_dict = asdict(tape_full)
    tape_toy = full2toy(dict(W1=tape_full_dict['W1'], c1=tape_full_dict['c1']))
    
    new_keys = list((tape_toy |  tape_full_dict).keys())
    Tape = new_tape_type(new_keys)
    
    tape = Tape(**(tape_toy |  tape_full_dict))

    return tape


loss_6d = lambda params, Y_tgt, cfg: loss_fn(params, Y_pred=linear_model(jnp.ones((cfg.batch_size, cfg.input_size)), params, cfg), Y=Y_tgt, cfg=cfg, reg_on=False)

def augment_tape(tape, W_teachers, sv=0):

    # just take first batch element
    W_students = tape.W1
    W_teachers = W_teachers if W_teachers.ndim == 4 else W_teachers[None, ...] # (pij -> bpij)
    W_students = W_students if W_students.ndim == 5 else W_students[None, ...] # (ptij -> btpij)
    
    U_s, S_s, VT_s = jnp.linalg.svd(W_students, full_matrices=False)  # leading axis is time
    U_t, S_t, VT_t = jnp.linalg.svd(W_teachers, full_matrices=False)

    # make a change of basis so that the teachers are [0, 1] and [1, 0]
    U_s, S_s = np.broadcast_arrays(U_s, S_s[..., None, :])
    W_s__U_t = np.einsum("bqia,btpia->bpqta", U_t, U_s*S_s) # p: teacher, q: student, t: time, a: singular vector
    if sv == -1:
        W_s__U_t = np.abs(W_s__U_t[..., :])  # only look at leading singular vector, consider others
    else:
        W_s__U_t = np.abs(W_s__U_t[..., sv])


    w11, w12, w21, w22 = W_s__U_t[:,0, 0], W_s__U_t[:,1, 0], W_s__U_t[:,0, 1], W_s__U_t[:,1, 1]
    w1 = jnp.array([w11, w12])
    w2 = jnp.array([w21, w22])
    tape_dict = asdict(tape)

    # ToyTape = namedtuple('ToyTape', ['w11', 'w12', 'w21', 'w22'] + list(asdict(tape).keys()))
    new_keys = ['w11', 'w12', 'w21', 'w22'] + list(tape_dict.keys())
    ToyTape = new_tape_type(new_keys)
    return ToyTape(**(dict(w11=w11, w12=w12, w21=w21, w22=w22) | tape_dict))


# parameter conversion

def toy2full_4d(toy):
    dw1 = toy['dw1']
    dw2 = toy['dw2']
    r = toy['r']
    c_bar = toy['c_bar']

    W1 = jnp.array([.5 + dw1, .5 - dw1])[:, None]  # ij
    W2 = jnp.array([.5 - dw2, .5 + dw2])[:, None]  # mind the sign for this definition

    W = r * jnp.stack([W1, W2], axis=0)  # pji
    W
    c = jnp.array([.5 + c_bar, .5 - c_bar])

    return dict(W1=W, c1=c)


def full2toy_4d(full):
    W = full['W1']
    c = full['c1']
    dw1 = W[..., 0, 0, 0] - W[..., 0, 1, 0] # tpij = tpi1
    dw2 = W[..., 1, 0, 0] - W[..., 1, 1, 0]
    r = jnp.linalg.norm(W, axis=-2).mean((-2, -1))  # norm over i, then average out p and j
    c_bar = jnp.mean(jnp.array([c[..., 0] - .5, -c[...,1]  + .5]), axis=0)
    return dict(dw1=dw1, dw2=dw2, r=r, c_bar=c_bar)

def full2toy_6d(full):
    W = full['W1']
    c = full['c1']
    w11 = W[..., 0, 0, 0]
    w12 = W[..., 0, 1, 0]
    w21 = W[..., 1, 0, 0]
    w22 = W[..., 1, 1, 0]
    r = jnp.linalg.norm(W, axis=-2).mean((-2, -1))  # norm over i, then average out p and j
    c1 = c[..., 0]
    c2 = c[..., 1]
    return dict(w11=w11, w12=w12, w21=w21, w22=w22, r=r, c1=c1, c2=c2)

def toy2full_6d(toy):
    w11 = toy['w11']
    w12 = toy['w12']
    w21 = toy['w21']
    w22 = toy['w22']
    c1 = toy['c1']
    c2 = toy['c2']

    W11 = w11
    W12 = w12
    W21 = w21
    W22 = w22

    W = jnp.stack([jnp.array([W11, W12])[:, None], jnp.array([W21, W22])[:, None]], axis=0)  # pji
    c = jnp.array([c1, c2])

    return dict(W1=W, c1=c)

def full2toy_8d(full):
    W = full['W1']
    c = full['c1']
    w11 = W[..., 0, 0, :].squeeze(-1)
    w12 = W[..., 0, 1, :].squeeze(-1)
    w21 = W[..., 1, 0, :].squeeze(-1)
    w22 = W[..., 1, 1, :].squeeze(-1)
    r = jnp.linalg.norm(W, axis=-2).mean((-2, -1))  # norm over i, then average out p and j
    c11 = c[..., 0, 0]
    c12 = c[..., 0, 1]
    c21 = c[..., 1, 0]
    c22 = c[..., 1, 1]
    return dict(w11=w11, w12=w12, w21=w21, w22=w22, r=r, c11=c11, c12=c12, c21=c21, c22=c22)

def toy2full_8d(toy):
    w11 = toy['w11']
    w12 = toy['w12']
    w21 = toy['w21']
    w22 = toy['w22']
    c11 = toy['c11']
    c12 = toy['c12']
    c21 = toy['c21']
    c22 = toy['c22']

    W11 = w11
    W12 = w12
    W21 = w21
    W22 = w22

    W = jnp.stack([jnp.array([W11, W12])[:, None], jnp.array([W21, W22])[:, None]], axis=0)  # pji
    c = jnp.array([[c11, c12], 
                   [c21, c22]])  # pi

    return dict(W1=W, c1=c)

def full2hyperb(full):
    """
    Hyperbolic parameterization from Saxe et al. 2013, Appendix A
    """
    W = full['W1']
    c = full['c1']
    
    w11 = W[..., 0, 0, 0]
    w12 = W[..., 0, 1, 0]
    w21 = W[..., 1, 0, 0]
    w22 = W[..., 1, 1, 0]

    c1 = c[..., 0]
    c2 = c[..., 1]

    gamma11 = c1**2 - w11**2
    theta11 = jnp.arctanh((c1*w11) / (c1**2 + w11**2))

    gamma12 = c1**2 - w12**2
    theta12 = jnp.arctanh((c1*w12) / (c1**2 + w12**2))

    gamma21 = c2**2 - w21**2
    theta21 = jnp.arctanh((c2*w21) / (c2**2 + w21**2))

    gamma22 = c2**2 - w22**2
    theta22 = jnp.arctanh((c2*w22) / (c2**2 + w22**2))

    return dict(gamma11=gamma11, theta11=theta11, gamma12=gamma12, theta12=theta12, gamma21=gamma21, theta21=theta21, gamma22=gamma22, theta22=theta22)

def hyperb2full(hyperb):
    ...

def get_hyperb_sol(ts, theta_f, theta_s, s, gamma):
    """
    Saxe 2013, Appendix A, eq. 26

    s and gamma specify the final value the dynamics converge to. 
    """
    
    tau = 1.
    F = lambda theta: jnp.log(((gamma**2 + s**2)**0.5 + gamma + s*jnp.tanh(theta/2)) / ((gamma**2 + s**2)**0.5 - gamma - s*jnp.tanh(theta/2)))
    t = (tau / (gamma**2 + s**2)) * (F(theta_f) - F(theta_s))

    return t