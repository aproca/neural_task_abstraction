# Define the learning rate and number of iterations
from dataclasses import InitVar, dataclass, field, fields
from typing import Callable
from jaxtyping import ArrayLike
import os
import equinox as eqx
import numpy as np
import jax.numpy as jnp
from dataclasses import dataclass
from simple_parsing import ArgumentParser
import logging
from lcs.curricula import get_n_phasic_curriculum
from lcs.utils import get_timestamp

logger = logging.getLogger(__name__)

# SHARED GLOBAL PARAMETERS
num_blocks = 20
Y_tgts = jnp.array([[1, 0], [0, 1]])  # list of axis-aligned basis vectors
g = 100
block_duration = 1.

cfg_full_ = dict(
    name="full",
    t_tot=num_blocks * block_duration,
    block_duration=block_duration,  # timescale of the context switch
    W_tau=block_duration * 1.3,
    c_tau=block_duration * .03,
    T_tape=-1,
    regularization_type=[
        ("nonnegative", 1),
        ("gating_manifold_L1", 10.0),
        ("L2_W", 0.0),
    ],
    regularization_strength=0.1,
    c_gt_curriculum="A_B__",
    teacher_mode='orthogonal',
)

# INFO: the syntax dict_child = dict(parent_dict, **child_dict) will return the parent_dict but will put the child_dict values in the parent_dict

cfg_baseline_ = dict(
    cfg_full_,  # inherit and update
    name="baseline",
    W1_tau=cfg_full_["W_tau"],
    W2_tau=cfg_full_["W_tau"],
    control="deep_mono",
    context_model=False,
    regularization_strength=0.0,  # we don't regularize the baseline model, as we don't have a modelling assumption that would imply gating on weights --> but see our experiments on the two-layer monolithic model
    num_layers=2,
)

cfg_forgetful_ = dict(
    cfg_full_,  # inherit and update
    name="forgetful",
    W_tau=cfg_full_["W_tau"],
    c_tau=cfg_full_["W_tau"],
    regularization_strength=0.0,
)

cfg_toy_ = dict(
    cfg_full_,  # inherit and update
    W_tau=block_duration * 1 / 2,
    c_tau=block_duration / 20,
    name="toy",
    output_size=2,
    num_seeds=1,
    input_size=1,
    use_X=False,
    metric="cosine,cols",  # we need to use cols here because W = [[1], [0]] has diverging norm row-wise
    W_teachers=Y_tgts.reshape(2, 2, 1),
)


""" END CONFIGS """

class Curriculum(eqx.Module):
    def __init__(self, Y_gt: str):
        self.Y_gt = Y_gt

    def __hash__(self, Y_gt: Callable[[float], np.ndarray], cfg):
        t = np.linspace(0, 100, 1000)
        y = Y_gt(t)
        return hash(y.mean()*y.std())  # redneck hashing
    
    def __call__(self, t):
        return self.Y_gt(t)

@dataclass
class Config:
    input_size: int = 20
    output_size: int = 10
    num_seeds: int = 10
    num_contexts: int = 2
    num_paths: int = 2
    batch_size: int = 200
    initialization_scale: float = 1e-2  # needs to be << 1 to prevent lazy learning=frustrated students
    turn_off_regularization: bool = False

    regularization_strength: float = 1e-2
    num_layers: int = 1
    hidden_size: int = 10
    context_model: bool = True
    name: str = ''
    regularization_type: str = 'nonnegative'

    shared_teachers: bool = False
    shared_concat_teachers: bool = False
    c_gt_curriculum: str = None
    Y_tgt: Callable = None
    W_teachers: np.array = field(default='generate', repr=False)
    use_X: bool = True  # determines whether to provide a constant X=1 input instead of using the random samples

    metric: str = 'cosine'
    teacher_mode: str = 'orthogonal'
    teacher_xx: float = 0.0
    teacher_scale: str = 'd_in'  # either 'unit' or 'd_in'
    teacher_rotate: bool = True  # whether to rotate the teacher-sVs out of axis alignment with an orthonormal matrix
        
    t_tot: float = None
    T_tot: int = None

    num_blocks: int = None
    block_duration: float = None

    # TODO: remove these options
    W_lr: float = None
    W1_lr: float = None
    W2_lr: float = None

    c_lr: float = None

    W_tau: float = None  # W_tau = 1 will lead to SVs=target SV being reached after 1 time unit
    W1_tau: float = None
    W2_tau: float = None

    c_tau: float = 2.5 ** -1  # c_tau = 1 will lead to c=1 being reached after 1 time unit
  
    dt: float = 1e-3  # simulation time resolution. This determines the number of samples and hence the information per time unit

    log_every: int = None
    T_tape: int = 1000
    dt_tape: float = None

    control: str = ''
    data_out_dir: str ='data_internal/joint_learning/'
    results_out_dir: str = 'results_internal/joint_learning/'
    num_shared_contexts: int = 0
    mixing_factor: float = 0.5
    W_regularization_strength: float = 0.

    log_aux: bool = True

    # MNIST-specific
    dataset_name: str = 'mnist'
    data_appendix: str = '_CNN_bottleneck10'
    permutation1: str = None
    permutation2: str = None
    data_folder: str = os.path.join('data', 'mnist'),
    results_folder: str = os.path.join('results', 'mnist', "%s_%s"%(str(None), str(None)))


    def __post_init__(self):
        # TODO remove this compatibility layer
        self.W_tau = self.W_lr ** -1 if self.W_lr is not None else self.W_tau
        self.W1_tau = self.W1_lr ** -1 if self.W1_lr is not None else self.W1_tau
        self.W2_tau = self.W2_lr ** -1 if self.W2_lr is not None else self.W2_tau
        self.c_tau = self.c_lr ** -1 if self.c_lr is not None else self.c_tau

        # allow to either specify num_blocks or T_tot

        ## num_blocks measures the number of blocks
        if self.t_tot is not None and self.block_duration is not None:
            self.num_blocks = int((self.t_tot // self.block_duration) + 1)
        ## t_tot measures the total "continuous" time
        elif self.num_blocks is not None and self.block_duration is not None:
            self.t_tot = self.num_blocks * self.block_duration
        elif self.num_blocks is not None and self.t_tot is not None:
            self.block_duration = self.t_tot / self.num_blocks
        else:
            raise Exception("Must specify either t_tot or num_blocks and block_duration")


        ## T_tot is the number of discrete time steps
        self.T_tot = int(self.t_tot // self.dt + 1)

        self.layer_sizes = [self.input_size] + [self.hidden_size]*(self.num_layers-1) + [self.output_size]

        if self.T_tape == -1:
            self.T_tape = self.T_tot

        if self.log_every is None:
            self.log_every = self.T_tot // self.T_tape

        ## T_tape is the number of discrete time steps to log
        if self.T_tape is None:
            self.T_tape = self.T_tot // self.log_every

        if self.T_tape > self.T_tot:
            self.T_tape = self.T_tot


        ## dt_tape is the time step of the tape
        self.dt_tape = self.t_tot / self.T_tape
        
        if self.dt_tape > self.block_duration:
            logger.warning("Tape duration is smaller than block duration, aliasing to be expected.")

        if self.output_size == 2 and ((type(self.W_teachers) == str) and (self.W_teachers == 'generate')):
            logger.warning("One of the students is likely very well aligned already in terms of cossim due to the small size of R2.")

        if self.control == '2_diag_mono' or self.control == 'N_diag_mono' or self.control == 'deep_mono':
            self.layer_sizes = [self.input_size] + [self.num_paths*self.hidden_size] + [self.output_size]
        if self.name is None:
            self.name = get_timestamp()

        if self.context_model and self.control != '':
            raise Exception("Cannot be both context model and control")
        
        if self.dt > 1e-3:
            logger.warning("Using a dt > 1e-3 introduces finite-size effects!")
        
        assert not (hasattr(self.c_gt_curriculum, '__call__') and  hasattr(self.Y_tgt, '__call__')), "Cannot have both c_gt_curriculum and Y_tgt callable"
        

    def __getitem__(self, k):
        return getattr(self, k)
    
    def __hash__(self):
        import json
        return hash(json.dumps({field.name: getattr(self, field.name) for field in fields(self) if field.init and not hasattr(getattr(self, field.name), 'shape')}, sort_keys=True))
    


# construct parser
parser = ArgumentParser(description='config')
parser.add_arguments(Config, "config")

# comptiblity with cli interface
def cfg_from_string(input_string):
    parser_result = parser.parse_args(input_string.split()) if input_string else parser.parse_args()
    cfg = parser_result.config
    return cfg

if __name__ == '__main__':

    test_cli_string = "--name=Ndiag --block_duration=50 --num_blocks=40 --c_gt_curriculum=A_B__AB__ --control=N_diag_mono --context_model=False --regularization_type=nonneg_lagrange --shared_concat_teachers=True"
    test_cli_string += " --W_lr=0.05 --c_lr=1.25" # TODO remove this
    # test_cli_string += " --W_tau=20 --c_tau=0.5"  # TODO adopt this
    print(parser.parse_args(test_cli_string.split()))