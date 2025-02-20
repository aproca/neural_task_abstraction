from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import json
from scipy import stats
import os
from datetime import datetime
from dataclasses import asdict
import logging
from lcs.curricula import get_n_phasic_curriculum
from lcs.svd_random_matrix_theory import get_expected_sv

logger = logging.getLogger(__name__)


# %% GENERAL FUNCTIONS

def allow_dot_notation(tape):
    for k, v in tape.items():
        setattr(tape, k, tape[k])
    return tape

def compute_mean_sem(data, axes):
    return np.mean(data, axis=axes), stats.sem(data, axis=axes)

def calc_specialization(tape, cfg):
    from lcs.joint_learning import sort_it

    tape = sort_it(tape, cfg)
    sim = np.abs(tape.cos_sim1)  # (B, T, p, p)

    idx_last_10_perct = np.arange(-((tape.t.shape[-1]) // 10), 0)
    sim = sim[:, idx_last_10_perct]

    diag_idx = np.eye(sim.shape[2], dtype=bool)

    align = sim[:, :, diag_idx]
    align = align.max((-1, -2))

    disalign = sim[:, :, ~diag_idx]
    disalign = disalign.min((-1, -2))

    return align, disalign

def _compute_similarity(W_teacher, W_student, metric=None, truncate=True):
    """
    vectorization enables calling syntax with ndim = 3 arrays of shape
    W_teacher=(M, K, I) OR  W_teacher=(K, I) and W_teacher2=(K, I)
    W_student=(P, L, J)

    rsa_matrix = similarity(W_teacher[:, None, ...], W_student[None, :, ...], metric="SVD")  # (M, P, ...)

    where 
    sim_matrix[m, p] = similarity(W_teacher[m], W_student[p], metric="SVD")

    or 
    sim_matrix[a, p] = projection of student p on joint-teachers-sv-space a

    We can also call
    rsa = similarity(W_teacher[None, ...], W_student, metric="SVD")  # (T, M, P)
    if the students have a leading time axis that the teachers do not have

    If W_teacher2 is provided, the teachers are used to calculate a *joint* SV space.
    """
    
    if "svd" in metric.lower():
        # place tensors on CPU to prevent Apple METAL problem
        cpu = jax.devices("cpu")[0]
        W_student = jax.device_put(W_student, cpu)
        W_teacher = jax.device_put(W_teacher, cpu)

        U_s, S_s, VT_s = jnp.linalg.svd(W_student, full_matrices=False)  # full matrices is important in order to truncate the SVD to the correct rank
        if 'joint' in metric.lower():
            W_teacher1, W_teacher2 = W_teacher
            U_t, S_t, VT_t = jnp.linalg.svd(W_teacher1 + W_teacher2, full_matrices=False)
        else:
            U_t, S_t, VT_t = jnp.linalg.svd(W_teacher, full_matrices=False)

        if truncate:
            n_SV_s = 2  # every context can only have one salient SV -> truncate
            n_SV_t = 2 if 'joint' in metric.lower() else 1
            U_s = U_s[:, :n_SV_s]
            U_t = U_t[:, :n_SV_t]
            VT_s = VT_s[:n_SV_s]
            VT_t = VT_t[:n_SV_t]
            S_s = S_s[:n_SV_s]
            S_t = S_t[:n_SV_t]


        # compute the cosine similarity between the singular vectors
        # a is the sv index, i is the component of the singular vector
        # cosine similarity between the singular vectors, weighted by their importance in the teacher
        if metric.lower() == "svd":
            metric = "svd,u"

        K_V = jnp.abs(jnp.einsum("ai,bi->ab", VT_t, VT_s))
        K_U = jnp.abs(jnp.einsum("ia,ib->ab", U_t, U_s))

        if ",ab" in metric.lower(): 
            if ',v' in metric.lower():
                similarity = jnp.einsum('aj,bj->ab', VT_t, VT_s)
            elif ',u' in metric.lower():
                similarity = jnp.einsum('ia,ib->ab', U_t, U_s)
            elif ',sv' in metric.lower():
                similarity = jnp.einsum('a,b->ab', S_t, S_s) / S_t[:, None]  # (a, b)
            else:
                raise ValueError

            # only take the diagonal, legit if the SVs are aligned
            # similarity = jnp.diag(similarity)  # (a,)    
        elif ',v' in metric.lower():
            similarity = K_V.max(-1)  if ',a' in metric.lower() else K_V.max()  # maximum student alignment to teacher SV a
        
        elif ",u" in metric.lower():
            similarity = K_U.max(-1)  if ',a' in metric.lower() else K_U.max()
        
        elif ',sv' in metric.lower():
            # assume that singular vectors are already aliged, and get the projections
            # we normalize by the teacher SVs
            S_t_abs = jnp.abs(S_t)
            S_s_abs = jnp.abs(S_s)

            idx_sort = jnp.argsort(K_U[0] if 'k_u' in metric.lower() else K_V[0])[::-1]  # get indcs to the closest teacher (the teacher only has one SV)
            S_s_abs = S_s_abs[idx_sort]

            similarity = S_s_abs / S_t_abs  # (a)
            similarity = similarity if ',a' in metric.lower() else similarity[0]  # entry corresponding to highest aligned pair
        
        elif ",sandwich" in metric.lower():
            similarity = jnp.einsum("ia,ij,aj->a", U_t, W_student, VT_t)  # importantly, we here sum over all right SVs (b) because the student cannot know which X direction is pertinent
            
        
        else:
            raise NotImplementedError("Only SVD similarity is implemented")

    elif "cos" in metric.lower(): 
        if ',' not in metric:
            # default value
            metric = metric + ',rows'

        if (',rows' in metric.lower()) or (',v' in metric.lower()):
            norm_student = jnp.linalg.norm(W_student, axis=-1, keepdims=True)
            norm_teacher = jnp.linalg.norm(W_teacher, axis=-1, keepdims=True)
            W_student = W_student / norm_student
            W_teacher = W_teacher / norm_teacher
            similarity = jnp.einsum("ij,ij->i", W_teacher, W_student)
        elif (',cols' in metric.lower()) or (',u' in metric.lower()):
            norm_student = jnp.linalg.norm(W_student, axis=-2, keepdims=True)
            norm_teacher = jnp.linalg.norm(W_teacher, axis=-2, keepdims=True)
            W_student = W_student / norm_student
            W_teacher = W_teacher / norm_teacher
            similarity = jnp.einsum("ij,ij->i", W_teacher, W_student)
        else:
            raise ValueError
        
        similarity = similarity.mean(axis=-1)

    elif "contract_both" in metric.lower():
        similarity = jnp.einsum("ij,ij", W_teacher, W_student)
    else:
        raise NotImplementedError("Unknown metric")
    
    return similarity if similarity.ndim > 0 else similarity[None]  # (A,)

def compute_similarity(W_teacher, W_student, metric=None, truncate=True):
    if 'ab' in metric.lower():
        vectorize = partial(jnp.vectorize, signature="(k,i),(l,j)->(a,b)", excluded=(2, "metric", "truncate"))
        _compute_similarity_v = vectorize(_compute_similarity)
        sim = _compute_similarity_v(W_teacher, W_student, metric=metric, truncate=truncate)
        return sim

    elif 'joint' not in metric.lower():
        # Squeeze the SV axis if present!
        vectorize = partial(jnp.vectorize, signature="(k,i),(l,j)->(a)", excluded=(2, "metric", "truncate"))
        _compute_similarity_v = vectorize(_compute_similarity)
        sim = _compute_similarity_v(W_teacher, W_student, metric=metric, W_teacher2=None)
        return sim.squeeze(-1) if sim.shape[-1] == 1 else sim

    else:
        # bring SV axis to front to replace M axis
        vectorize = partial(jnp.vectorize, signature="(p,k,i),(l,j)->(a)", excluded=(2, "metric", "truncate"))
        _compute_similarity_v = vectorize(_compute_similarity)

        # revert any broadcasting that had been done on the teachers, as they are now being treated jointly
        W_teacher = W_teacher.squeeze(-3) if W_teacher.shape[-3] == 1 else W_teacher

        sim = _compute_similarity_v(W_teacher, W_student, metric=metric, truncate=truncate)
        return sim.swapaxes(-2, -1)  # (..., P, A) -> (A==M, ...)

def test_compute_similarity():
    for metric in ["SVD", "cosine"]:
        # A & B are i x j: default case that does not need vectorization
        W_student, W_teacher = np.random.randn(3, 4), np.random.randn(3, 4)
        sim = compute_similarity(W_teacher, W_student, metric=metric)
        assert jnp.allclose(compute_similarity(W_student, W_student, metric=metric), 1.)
        assert sim.shape == ()

        # # A & B are p x i x j: first dimension is context-> introduce a broadcasting dimension with None
        W_student, W_teacher = np.random.randn(2, 3, 4), np.random.randn(2, 3, 4)
        sim = compute_similarity(W_teacher[:, None, ...], W_student[None, :, ...], metric=metric)
        assert sim.shape == (2, 2)

        # # A & B are s x i x j: first dimension is seed -> DON'T introduce a broadcasting dimension with None
        W_student, W_teacher = np.random.randn(2, 3, 4), np.random.randn(2, 3, 4)
        sim = compute_similarity(W_teacher[:, ...], W_student[:, ...], metric=metric)
        assert jnp.allclose(compute_similarity(W_student[:, ...], W_student[:, ...], metric=metric), 1.)
        assert sim.shape == (2,)

        # # A & B are j,: -> introduce a broadcasting dimension with None, THIS IS NOT THE OUTER PRODUCT THOUGH as above but just reshape(1, -1)
        W_student, W_teacher = np.random.randn(4), np.random.randn(4)
        sim = compute_similarity(W_teacher[None, ...], W_student[None, ...], metric=metric)
        assert jnp.allclose(compute_similarity(W_student[None, ...], W_student[None, ...], metric=metric), 1.)
        assert sim.shape == ()

    W_student, W_teacher = np.random.randn(3, 4), np.random.randn(3, 4)
    sim_1 = compute_similarity(W_teacher, W_student,  metric="Svd")
    sim_2 = compute_similarity(W_teacher, W_student, metric="cosine")
    assert not jnp.allclose(sim_1, sim_2)

def load_arg_parser(data_folder):
    parser = arg_parser.get_parser()  
    args = parser.parse_args('')
    with open(os.path.join(data_folder, 'args.json'), 'r') as f:
        args.__dict__ = json.load(f)
    return args

def append_to_dict_list(dict, key, value):
    if key not in dict:
        dict[key] = []
    dict[key].append(value)
    return dict

def combine_tuples(tuples): 
    combined = []
    [combined.extend(list(x)) for x in tuples]

    return tuple(combined)

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def make_timed_results_folder(base_folder = None, addition=''):
    results_folder = '%s%s' %(datetime.now().strftime('%Y%m%d%H%M%S'), addition)
    if base_folder is not None:
        results_folder = os.path.join(base_folder, results_folder)
    
    os.makedirs(results_folder, exist_ok=True)
    return results_folder

def flatten(l):
    return [item for sublist in l for item in sublist]

def get_timestamp():
    return datetime.now().strftime('%Y%m%d%H%M%S')

def configure_arg_parser(args):
    if args.t_tot is not None:
        args.num_blocks = int((args.t_tot // args.block_duration) + 1)
    else:
        args.t_tot = args.num_blocks * args.block_duration
    args.T_tot = int(args.t_tot // args.dt + 1)
    if args.T_tape == -1:
        args.T_tape = args.T_tot
    if args.log_every is None:
        args.log_every = args.T_tot // args.T_tape
    if args.T_tape is None:
        args.T_tape = args.T_tot // args.log_every
    assert args.T_tot >= args.T_tape, "Tape must be smaller than total time"
    args.dt_tape = args.t_tot / args.T_tape
    if not args.dt_tape > args.block_duration:
        logger.warning("Tape duration is smaller than block duration, aliasing to be expected.")

    return args

def get_2D_proj(tape, cfg, W_teachers=None, compute_sim=None, metric=None, end_of_block_only=False, tr=None, where_fit=None):
    """
    Returns the students in teacher coordinates, i.e. a matrix (M, P, T, A) = (paths, contexts, times, singular values). Svs are sorted
    """
    if tr is None:
        tr = lambda x: x
    tape = jax.tree.map(lambda x: x[0], tape) if tape.t.ndim == 2 else tape

    # just take first batch element
    W_students = tape.W1
    W_teachers = tape.W_teachers if W_teachers is None else W_teachers
    if W_teachers.ndim == 4:
        W_teachers = W_teachers[0]

    # add a path axis if not present
    W_students = W_students if W_students.ndim == 4 else W_students[:, None, :, :] # (tij -> tpij)
    
    
    if compute_sim is None:
        compute_sim = partial(compute_similarity, metric=metric)
    sim = compute_sim(W_teachers[None, :,    None], # (T, M, 1, ...)
                      W_students[:,    None, :   ], # (T, 1, P, ...)
    )  # (T, M, P, ...) = (time, multi-context, paths, svs students)

    if end_of_block_only:
        t_ante_switch = np.where(np.abs(np.diff(tape.c_gt1) / cfg.dt_tape) > 0.2)[0] - 1
        assert len(t_ante_switch) > 0
        t_ante_switch = t_ante_switch[t_ante_switch >= 0] 
        sim = sim[t_ante_switch]

    return sim[..., None]  # (T, M, P, A,)

def run_cfg_pair(cfg_toy_, cfg_full_, w1, w2, run_full=True, mirror=False, average=False, params_init_full=None):
    from lcs.configs import Config, Y_tgts
    from lcs.mechanism import get_tape_2d, new_tape_type, augment_tape, full2toy_6d, toy2full_6d
    from lcs.svd_random_matrix_theory import get_expected_sv_mc
    from lcs.joint_learning import run_config
    from lcs.joint_learning import sort_it

    block_duration = cfg_toy_['block_duration']
    w11, w12 = w1
    w21, w22 = w2
    tapes_2d = []
    tapes_full = []
    cfgs_2d = []
    cfgs_full = []
    for c_start in [0, 1]:

        g = 1000
        Y_tgt, c_gt_curriculum = get_n_phasic_curriculum(g, block_duration, *(Y_tgts[::-1] if c_start == 1 else Y_tgts), return_c=True)

        cfg_toy = Config(**dict(cfg_toy_,
                    Y_tgt=Y_tgt,
                    W_teachers=Y_tgts.reshape(2, 2, 1),
                    ))

        tape_2d = get_tape_2d(
            cfg_toy,
            dict(
                w11=w11,
                w12=w12,
                w21=w21,
                w22=w22,
                c1=0.5,
                c2=0.5,
            ),
            full2toy=full2toy_6d,
            toy2full=toy2full_6d,
        )

        # undo squeezing
        tape_2d = jax.tree.map(lambda x: x[None, ...], tape_2d) if tape_2d.t.ndim == 1 else tape_2d
        tapes_2d.append(tape_2d)
        cfgs_2d.append(cfg_toy)

        if run_full:

            _cfg_full_ = Config(**dict(cfg_full_,  # just to retrieve the output_size and input_size
                        c_gt_curriculum=c_gt_curriculum,
                        ))

            output_size, input_size = _cfg_full_.output_size, _cfg_full_.input_size
            sv_expc = get_expected_sv(output_size, input_size, 
                                      sigma=1. # teacher SV scale
                                      )
            print(f"expected_sv: {sv_expc}")
            initialization_scale = np.linalg.norm(w1) / sv_expc  # we divide out the expected SV of the initialized students so that it becomes norm(w1)
            print(f"initialization_scale: {initialization_scale}")

            cfg_full = Config(**dict(cfg_full_,
                        # c_gt_curriculum=c_gt_curriculum,
                        initialization_scale=initialization_scale,
                        ))

            tape_full = run_config(cfg_full, params_init=params_init_full)
            tape_full = augment_tape(tape_full, tape_full.W_teachers, sv=None)
            tape_full = jax.tree.map(lambda x: x.mean(0), tape_full) if average else tape_full
            tapes_full.append(tape_full)
            cfgs_full.append(cfg_full)
        else:
            tape_full = None
            cfg_full = None

    # need this patch because the created simulations are of different type because they happen with different configs...
    Tape = new_tape_type(asdict(tape_2d).keys())
    tapes_2d = [Tape(**asdict(t)) for t in tapes_2d]
    Tape = new_tape_type(asdict(tape_full).keys())
    tapes_full = [Tape(**asdict(t)) for t in tapes_full if t is not None]

    if mirror:
        tape_2d = jax.tree.map(lambda x, y: (x + y) / 2, tapes_2d[0], tapes_2d[1])
        tape_full = jax.tree.map(lambda x, y: (x + y) / 2, tapes_full[0], tapes_full[1])
        
    else:
        tape_2d = tapes_2d[0]
        tape_full = tapes_full[0]

    cfg_full = cfgs_full[0]
    cfg_toy = cfgs_2d[0]

    return tape_2d, cfg_toy, tape_full, cfg_full

def rdm_bunch_of_xx(xx, N, K=1, seed=None, method="gaussian"):
    """
    Generate random pairs of vectors in high-dimensional space.

    Args:
        xx (ndarray): Array of angles in radians.
        N (int): Number of dimensions for each vector.
        K (int, optional): Number of pairs to generate for each angle. Defaults to 1.
        seed (int, optional): Seed value for the random number generator. Defaults to None.

    Returns:
        ndarray, ndarray: Arrays of shape (len(xx), K, N) representing the generated vectors.

    Raises:
        AssertionError: If the generated vectors do not satisfy the given angle.

    """
    rng = np.random.default_rng(seed)
    x_base = rng.normal(size=(N))
    x_base /= np.linalg.norm(x_base)

    

    if method == "gaussian":
        std = ((1-xx) / xx)**.5
        X_add = rng.normal(size=(K, N)) * (1/N)**.5
        # superpose
        X = x_base + X_add*std

        # (x+xi).(x+xi') / (x+xi^2) = 1/(1 + sigma^2) != xx <-> 

    # compute average overlap
    X /= np.linalg.norm(X, axis=-1, keepdims=True)
    avg_overlaps = np.einsum("ki,li->kl", X, X) / (np.linalg.norm(X, axis=-1)[:, None] * np.linalg.norm(X, axis=-1)[None, :])
    diag_idx = np.eye(K, dtype=bool)
    avg_overlaps = avg_overlaps[~diag_idx]

    print(f"Average overlap: {avg_overlaps.mean()}, std: {avg_overlaps.std()}")

    return X

def get_adapt(cossim, lb=1):
    cossim = cossim[-lb:]
    diag = np.eye(2).astype(bool)

    # off_diag = var[:, ~diag]
    # diag = var[:, diag]

    # return np.abs(np.abs(diag).mean() - np.abs(off_diag).mean())

    # low_pass_filter cossim
    filter_size = len(cossim) // 10
    convolve = partial(jax.numpy.convolve, v=np.ones(filter_size) / filter_size, mode='same')
    cossim = jax.numpy.array(cossim)
    convolve = jax.vmap(convolve, in_axes=(1))
    convolve = jax.vmap(convolve, in_axes=(2))
    cossim = convolve(cossim)

    return (cossim[diag].mean() - cossim[~diag].mean())


if __name__ == "__main__":
    X = rdm_bunch_of_xx(.9, N=100, K=20)