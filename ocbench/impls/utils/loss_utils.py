import jax
import jax.numpy as jnp
import jax.scipy.special


HL_GAUSS_MIN_VALUE = 0.0
HL_GAUSS_MAX_VALUE = 1.0
HL_GAUSS_NUM_BINS = 101
HL_GAUSS_SIGMA_RATIO = 0.75


def sample_beta_times(rng, shape, betas):
    """Sample flow times from a Beta distribution."""
    beta_a, beta_b = betas
    if beta_a == 1.0:
        # Use inverse CDF for Beta(1, b), which is much faster than jax.random.beta.
        t_uniform = jax.random.uniform(rng, shape)
        return 1 - (1 - t_uniform) ** (1 / beta_b)
    return jax.random.beta(rng, beta_a, beta_b, shape)


def hl_gauss_bin_centers(num_bins=HL_GAUSS_NUM_BINS):
    """Return HL-Gauss bin centers over [0, 1]."""
    edges = jnp.linspace(HL_GAUSS_MIN_VALUE, HL_GAUSS_MAX_VALUE, num_bins + 1, dtype=jnp.float32)
    return (edges[:-1] + edges[1:]) / 2


def hl_gauss_targets(targets, num_bins=HL_GAUSS_NUM_BINS):
    """Convert scalar targets to Gaussian-smoothed histogram targets."""
    edges = jnp.linspace(HL_GAUSS_MIN_VALUE, HL_GAUSS_MAX_VALUE, num_bins + 1, dtype=jnp.float32)
    bin_width = (HL_GAUSS_MAX_VALUE - HL_GAUSS_MIN_VALUE) / num_bins
    sigma = HL_GAUSS_SIGMA_RATIO * bin_width

    targets = jnp.clip(targets, HL_GAUSS_MIN_VALUE, HL_GAUSS_MAX_VALUE)[..., None]
    cdf_evals = jax.scipy.special.erf((edges - targets) / (jnp.sqrt(2.0) * sigma))
    z = cdf_evals[..., -1:] - cdf_evals[..., :1]
    probs = cdf_evals[..., 1:] - cdf_evals[..., :-1]
    return probs / jnp.maximum(z, 1e-8)


def hl_gauss_from_probs(probs):
    """Convert categorical probabilities to scalar values."""
    centers = hl_gauss_bin_centers(probs.shape[-1])
    return jnp.sum(probs * centers, axis=-1)


def hl_gauss_from_logits(logits):
    """Convert categorical logits to scalar values."""
    return hl_gauss_from_probs(jax.nn.softmax(logits, axis=-1))


def hl_gauss_cross_entropy(logits, targets):
    """Compute cross entropy from HL-Gauss targets to predicted logits."""
    target_probs = hl_gauss_targets(targets, num_bins=logits.shape[-1])
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.sum(target_probs * log_probs, axis=-1)
