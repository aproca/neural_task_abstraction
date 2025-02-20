""" Curriculum definitions """

import jax
import jax.numpy as jnp

def get_cos(g, t_ctx, t_period):
    """
    g: hardness of switch
    t_ctx: timescale of the context switch
    """
    clip_sin = lambda x: jnp.where(jnp.abs(x) < 1., jnp.sin(x*jnp.pi/2), 1.)*jnp.sign(x)
    gain = lambda y: jnp.sign(y) * jnp.abs(jnp.sin(y*jnp.pi/2))**(1/g)

    def rect_cos(x):
        return jnp.where(jnp.abs(x) < jnp.pi, jnp.cos(x), -1.)


    def squashed_cos(x):
        # original width is pi and we want it to be t_ctx
        return (rect_cos(x * jnp.pi / t_ctx))
    
    def gained_shifted_cos(x):
        return (gain(squashed_cos(x))  + 1.) / 2

    def tiled_cos(x):
        x_eval = jnp.abs(jnp.fmod(x, t_period))
        x_eval = jnp.minimum(x_eval, t_period - x_eval)
        return gained_shifted_cos(x_eval)
    
    return tiled_cos  # in [0, 1]

def get_n_phasic_curriculum(g=100, t_ctx=10, *Y_tgts, return_c=False, phase_shift=1/4):
    """
    g: hardness of switch
    t_ctx: timescale of the context switch
    """
    Y_tgts = jnp.array(Y_tgts)

    N_ctx = len(Y_tgts)
    t_period = t_ctx*N_ctx

    tiled_cos = get_cos(g, t_ctx, t_period)

    def w_t_overlaps(t):
        eps = 1 - 1e-4  # We need to make sure that at least one context is always on, so make them overlap very slightly
        y = jnp.array([(tiled_cos((t - i * (t_period/N_ctx) - t_period*phase_shift*eps))) for i in range(N_ctx)]).T
        return y

    def Y_tgt(t):
        Y = jnp.sum(w_t_overlaps(t)[..., None, :] * Y_tgts.T, axis=-1)
        return Y
    
    return Y_tgt if not return_c else (Y_tgt, w_t_overlaps)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    Y_tgts = jnp.eye(2)[:2]
    g = 1000

    largs = dict(marker='o', alpha=.5)

    Y_tgt, c_vec = get_n_phasic_curriculum(g, 1, *Y_tgts, return_c=True)

    Y_tgt = jax.jit(Y_tgt)
    c_vec = jax.jit(c_vec)

    t = jnp.linspace(-10, 10, 1001)  # there are some points where the same value is put out
    y = Y_tgt(t)
    ax1.plot(t, y, **largs)

    for i in range(len(Y_tgts)):
        ax2.plot(t, c_vec(t)[...,i], label=f'c_{i}', **largs)
    ax2.legend()
    plt.show()