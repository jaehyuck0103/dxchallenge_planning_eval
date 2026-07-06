"""JAX Savitzky-Golay filter, copied from V-Max (vmax/simulator/metrics/utils.py)."""

import jax
import jax.numpy as jnp
from jax.numpy.linalg import lstsq


def savgol_coeffs_jax(window_length, polyorder, deriv=0, delta=1.0, pos=None):
    """Compute Savitzky-Golay filter coefficients using JAX.

    Args:
        window_length: Window length for the filter.
        polyorder: Polynomial order to approximate the data.
        deriv: Order of derivative to compute.
        delta: Spacing of the samples.
        pos: Position in the window to evaluate the derivative.

    Returns:
        The computed filter coefficients.

    """
    if polyorder >= window_length:
        raise ValueError("polyorder must be less than window_length.")

    halflen, rem = divmod(window_length, 2)
    if pos is None:
        pos = halflen if rem == 1 else halflen - 0.5

    if not (0 <= pos < window_length):
        raise ValueError("pos must be nonnegative and less than window_length.")

    x = jnp.arange(-pos, window_length - pos, dtype=jnp.float32)
    x = x[::-1]  # Reverse for convolution

    order = jnp.arange(polyorder + 1).reshape(-1, 1)
    a = x**order

    y = jnp.zeros(polyorder + 1)
    y = y.at[deriv].set(jax.scipy.special.factorial(deriv) / (delta**deriv))

    coeffs, _, _, _ = lstsq(a, y)

    return coeffs


def savgol_filter_jax(x, window_length, polyorder, deriv=0, delta=1.0, mode="interp"):
    """Apply a Savitzky-Golay filter to the input array using JAX.

    Args:
        x: Input data sequence.
        window_length: Window length for the filter.
        polyorder: Polynomial order for the filter.
        deriv: Order of derivative to compute.
        delta: Spacing between samples.
        mode: Padding mode, where "interp" is implemented.

    Returns:
        The filtered array.

    """
    coeffs = savgol_coeffs_jax(window_length, polyorder, deriv=deriv, delta=delta)

    # Handle padding for the edges
    if mode == "interp":
        pad_width = window_length // 2
        x_padded = jnp.pad(x, pad_width, mode="reflect")
        y = jnp.convolve(x_padded, coeffs, mode="valid")
    else:
        raise ValueError("Currently, only mode='interp' is supported in JAX implementation.")

    # Ignore the edges for now
    y = jnp.where(jnp.arange(len(y)) < window_length // 2, y[window_length // 2], y)
    y = jnp.where(
        jnp.arange(len(y)) >= len(y) - window_length // 2, y[len(y) - window_length // 2 - 1], y
    )
    return y
