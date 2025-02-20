import jax.numpy as jnp

def model(X, params):
    """
    W = (p, i, j)  # context, output, input
    c = (p,)    # context
    ... batch dimension
    """
    c, W = params['c'], params['W']
    return jnp.einsum("p,pij,...j->...i", c, W, X)

# alias
default_model = model
def model_mono(X, params):
    """
    W = (i, j)   # context, output, input
    c = (i,)    # context
    ... batch dimension
    """
    W = params['W']
    return jnp.einsum("ij,...j->...i", W, X)

def linear_model(X, params, cfg):
    """"
    p: # paths
    i: # out
    j: # in
    b: batch size
    """
    hidden = X
    if cfg.context_model:
        return jnp.einsum("p,pij,bj->bi", params['c1'], params['W1'], hidden)
    elif cfg.control == '2_diag_mono':
        # creates long W2 (out, 2*in) with c_scalars on the diagonals
        # JB: this should be equivalent to the original LCS implementation
        W2 = jnp.zeros((cfg.output_size, cfg.num_paths*cfg.hidden_size)) ## should this be output size or hidden size ?
        for p in range(cfg.num_paths):
            for i in range(min(cfg.output_size, cfg.hidden_size)):
                W2 = W2.at[i,i+p*cfg.hidden_size].set(params['c1'][p])
        hidden = jnp.einsum("ij,jk,bk->bi", W2, params['W1'], hidden)
    elif cfg.control == 'N_diag_mono':
        # creates long W2 (out, 2*in) with c_vectors on the diagonals
        W2 = jnp.zeros((cfg.output_size, cfg.num_paths*cfg.hidden_size))
        c_idx = 0
        for p in range(cfg.num_paths):
            for i in range(min(cfg.output_size, cfg.hidden_size)):
                W2 = W2.at[i,i+p*cfg.hidden_size].set(params['c1'][c_idx])
                c_idx+=1
        hidden = jnp.einsum("ij,jk,bk->bi", W2, params['W1'], hidden)
    elif cfg.control == 'deep_mono':
        for l in range(cfg.num_layers):
            hidden = jnp.einsum('ij,bj->bi', params['W' + str(l+1)], hidden)
    elif cfg.control == 'c_hadamard':
        return jnp.einsum('pi,pij,bj->bi', params['c1'], params['W1'], X)
    else:
        raise ValueError(f"Control {cfg.control} not recognized")
    return hidden