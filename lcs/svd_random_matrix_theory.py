from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ['ENABLE_PJRT_COMPATIBILITY'] = '1'

@jax.jit
@partial(jnp.vectorize, excluded=(1,2))
def marchpast(l, g, sigma=1.):
    g = jnp.clip(g, 0.0, None)  # numerical stability
    "Marchenko-Pastur distribution"
    def m0(a):
        return jnp.clip(a, 0.0, None)
    gplus=(1+g**0.5)**2 * sigma**2
    gminus=(1-g**0.5)**2 * sigma**2
    return jnp.sqrt( m0(gplus - l) * m0(l- gminus)) / ( 2*jnp.pi*g*l*sigma**2)

# @jax.jit
@partial(jnp.vectorize, excluded=(1,2))
def marchpast_cdf(l, g, sigma=1.):
    g = jnp.clip(g, 0.0, None)  # numerical stability
    gplus = (1 + g**0.5)**2 * sigma**2
    gminus = (1 - g**0.5)**2 * sigma**2
    
    def integrand(x):
        return marchpast(x, g, sigma)
    
    def compute_cdf(l):
        upper = jax.lax.min(l, gplus)
        lower = gminus
        x_eval = jnp.linspace(lower, upper, 1000)
        return jax.scipy.integrate.trapezoid(integrand(x_eval), x_eval)
    
    val = jax.lax.cond(
        l <= gminus,
        lambda _: 0.0,
        lambda _: jax.lax.cond(
            l > gplus,
            lambda _: 1.0,
            compute_cdf,
            l
        ),
        l
    )

    return val

def get_expected_sv(output_size, input_size, sigma):
    return sigma**2

def get_expected_sv_mc(output_size, input_size, sigma):
    assert output_size <= input_size

    # compute expected singular value by sampling
    g = output_size/input_size
    gplus=(1+g**0.5)**2
    gminus=(1-g**0.5)**2
    lmbds_cand = jnp.linspace(gminus, gplus, 1000)
    svs = []
    for i in range(100):
        r = np.random.uniform(0, 1)
        i_sol = jnp.argmin(jnp.abs(marchpast_cdf(lmbds_cand, g) - r))
        lmbd = lmbds_cand[i_sol]
        sv = jnp.sqrt(lmbd)
        svs.append(sv)
    return np.mean(svs)

def sample_sv(m, n):
    """
    Sample singular values of a random matrix
    """
    X = np.random.randn(m, n) / np.sqrt(n)
    sv = np.linalg.svd(X, compute_uv=False)
    return sv

def empirical_marchenko_pastur(m, n, num_samples=1000):
    """
    Generate empirical Marchenko-Pastur distribution
    m: number of rows
    n: number of columns
    num_samples: number of matrices to generate
    """
    lmbds = []
    for _ in range(num_samples):
        X = np.random.randn(m, n) / np.sqrt(n)
        cov = X @ X.T
        lmbds_ = np.linalg.eigvalsh(cov)
        lmbds.extend(lmbds_)
    return np.array(lmbds)

def empirical_delta_s(m, n, num_samples=1000):
    """
    Calculate empirical Δs (difference in singular values of two independently sampled matrices)
    """
    delta_s_values = []
    for _ in range(num_samples):
        X1 = np.random.randn(m, n) / np.sqrt(n)
        X2 = np.random.randn(m, n) / np.sqrt(n)
        s1_ = np.linalg.svd(X1, compute_uv=False)
        s2_ = np.linalg.svd(X2, compute_uv=False)
        all_delta_s = np.abs(s1_[:, None] - s2_[None, :])
        all_delta_s = all_delta_s[~np.eye(all_delta_s.shape[0], dtype=bool)].flatten()
        delta_s = np.mean(all_delta_s)
        delta_s_values.append(delta_s)
    return np.array(delta_s_values)

def get_s0_over_s1(m, n, num_samples=1000):
    s0_over_s1 = []
    for _ in range(num_samples):
        X = np.random.randn(m, n) / np.sqrt(n)
        s = np.linalg.svd(X, compute_uv=False)
        
        s0_over_s1.append(s[0]/s[1])
    return np.array(s0_over_s1).mean()

def monte_carlo_delta_s(m, n, num_samples=1000):
    """
    Calculate Monte Carlo Δs (difference in singular values of two independently sampled matrices)
    """
    delta_s_values = []
    cdf = marchpast_cdf
    lmbds_range = jnp.linspace(0, 4, 1000)

    for _ in range(num_samples):
        r1 = np.random.uniform(0., 1.)
        lmbd1 = lmbds_range[jnp.argmin(jnp.abs(cdf(lmbds_range, m/n) - r1))]
        r2 = np.random.uniform(0., 1.)
        lmbd2 = lmbds_range[jnp.argmin(jnp.abs(cdf(lmbds_range, m/n) - r2))]

        # print(f"r1: {r1}, r2: {r2}")
        # print(f"sv1: {sv1}, sv2: {sv2}")

        delta_s = np.abs(lmbd2**.5 - lmbd1**.5)
        delta_s_values.append(delta_s)
    return np.array(delta_s_values)

if __name__ == '__main__':
    s = 1
    m, n = 10*s, 10*s
    g = m/n
    sigma = 1.

    # Calculate expected eigenvalue
    expected_sv = sigma**2
    expected_lmbd = expected_sv**2

    print(f"Expected eigenvalue: {expected_lmbd}")

    # Calculate empirical and Monte Carlo Δs
    empirical_delta_s_values = empirical_delta_s(m, n, )
    monte_carlo_delta_s_values = monte_carlo_delta_s(m, n,)

    # Plot distributions
    l = jnp.linspace(0.0, 4, 1000)
    plt.figure(figsize=(18, 6))

    # Theoretical distribution
    plt.subplot(1, 3, 1)
    plt.title(f"Theoretical Marchenko-Pastur distribution\ng={g}, sigma={sigma}")
    plt.plot(l, marchpast(l, g, sigma), label='PDF')
    plt.plot(l, marchpast_cdf(l, g, sigma), label='CDF')
    plt.axvline(x=expected_lmbd, color='r', linestyle='--', label='Expected eigenvalue, thy')
    plt.xlabel(r'Eigenvalue $\lambda$')
    plt.legend()

    # Empirical distribution
    plt.subplot(1, 3, 2)
    empirical_eigenvalues = empirical_marchenko_pastur(m, n,)
    plt.title(f"Empirical Marchenko-Pastur distribution\ng={g}, sigma={sigma}")
    plt.hist(empirical_eigenvalues, bins=50, density=True, alpha=0.7)
    plt.plot(l, marchpast(l, g, sigma), 'r-', label='Theoretical PDF')
    plt.axvline(x=np.mean(empirical_eigenvalues), color='r', linestyle='--', label='Expected eigenvalue, sim')
    plt.xlabel(r'Eigenvalue $\lambda$')
    plt.legend()

    # Δs distributions
    plt.subplot(1, 3, 3)
    plt.title(f"Δs distributions\ng={g}, sigma={sigma}")
    plt.hist(empirical_delta_s_values, bins=50, density=True, alpha=0.7, label='Empirical Δs')
    plt.hist(monte_carlo_delta_s_values, bins=50, density=True, alpha=0.7, label='Monte Carlo Δs')
    plt.xlabel(r'SINGULAR value $\Delta s$')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Print statistics
    print(f"Empirical Δs mean: {np.mean(empirical_delta_s_values)}")
    print(f"Empirical Δs std: {np.std(empirical_delta_s_values)}")
    print(f"Monte Carlo Δs mean: {np.mean(monte_carlo_delta_s_values)}")
    print(f"Monte Carlo Δs std: {np.std(monte_carlo_delta_s_values)}")