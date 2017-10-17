from pymc3.diagnostics import (
    effective_n as pymc3_ess, gelman_rubin as pymc3_gelman_rubin
)
from pysgmcmc.diagnostics.sample_chains import pymc3_multitrace


def _pymc3_diagnostic(get_sampler, pymc3_diagnostic_fun, n_chains=2,
                      samples_per_chain=100):
    """
    Compute pymc3 diagnostic defined by callable `pymc3_diagnostic_fun`
    on sampler returned by callable `get_sampler`. To compute the diagnostic,
    extract `n_chains` chains with `samples_per_chain` samples each.

    Parameters
    ----------
    get_sampler : callable
        Callable that takes a `tensorflow.Session` as input and returns a
        (possibly already "burnt-in")
        `pysgmcmc.sampling.MCMCSampler` subclass instance.

    pymc3_diagnostic_fun : callable
        Callable that takes a `pymc3.MultiTrace` object as input and
        returns some diagnostic value for the chains in that `pymc3.MultiTrace`.

    n_chains : int, optional
        Number of individual chains/traces to extract.
        Defaults to `2`.

    samples_per_chain : int, optional
        Number of samples each individual chain should contain.
        Defaults to `100`.

    Returns
    ----------
    diagnostic_output
        Diagnostic result computed by pymc3.

    """

    multitrace = pymc3_multitrace(
        get_sampler, n_chains=n_chains, samples_per_chain=samples_per_chain
    )
    # delegate work of computing diagnostic to pymc3
    return pymc3_diagnostic_fun(multitrace)


def effective_sample_sizes(get_sampler, n_chains=2, samples_per_chain=100):
    """
    Calculate ess metric for a sampler returned by callable `get_sampler`.
    To do so, extract `n_chains` traces with `samples_per_chain` samples each.

    Parameters
    ----------
    get_sampler : callable
        Callable that takes a `tensorflow.Session` as input and returns a
        (possibly already "burnt-in")
        `pysgmcmc.sampling.MCMCSampler` subclass instance.

    n_chains : int, optional
        Number of individual chains/traces to extract.
        Defaults to `2`.

    samples_per_chain : int, optional
        Number of samples each individual chain should contain.
        Defaults to `100`.

    Returns
    ----------
    ess : dict
        Dictionary that maps the variable name of each target parameter of a
        sampler obtained by callable `get_sampler` to the corresponding
        effective sample size of (each dimension of) this variable.

    Notes
    ----------
    The diagnostic is computed as:

    .. math:: \\hat{n}_{eff} = \\frac{mn}{1 + 2 \\sum_{t=1}^T  \\hat{\\rho}_t}

    where :math:`\\hat{\\rho}_t` is the estimated autocorrelation at lag t, and T
    is the first odd positive integer for which the sum
    :math:`\\hat{\\rho}_{T+1} + \\hat{\\rho}_{T+1}` is negative.

    References
    ----------
    Gelman et al. (2014)

    Examples
    ----------
    Simple (arbitrary) example to showcase usage:

    >>> import tensorflow as tf
    >>> from pysgmcmc.samplers.sghmc import SGHMCSampler
    >>> params = [tf.Variable([1.0, 2.0], name="x", dtype=tf.float64)]
    >>> cost_fun = lambda params: tf.reduce_sum(params)  # dummy cost functions
    >>> get_sampler = lambda session: SGHMCSampler(
    ...    params=params, cost_fun=cost_fun, session=session
    ... )
    >>> ess_vals = effective_sample_sizes(get_sampler=get_sampler)
    >>> type(ess_vals)
    <class 'dict'>
    >>> list(ess_vals.keys())[0].startswith("x")  # necessary due to tensorflow variable enumeration
    True
    >>> len(ess_vals[params[0].name])  # x:0 has two dimensions, we have one ess value for each
    2
    """

    return _pymc3_diagnostic(
        pymc3_diagnostic_fun=pymc3_ess,
        get_sampler=get_sampler,
        n_chains=n_chains,
        samples_per_chain=samples_per_chain
    )


def gelman_rubin(get_sampler, n_chains=2, samples_per_chain=100):
    r"""
    Calculate gelman_rubin metric for a sampler returned by callable `get_sampler`.
    To do so, extract `n_chains` traces with `samples_per_chain` samples each.

    The Gelman-Rubin diagnostic tests for lack of convergence by comparing
    the variance between multiple chains to the variance within each chain.
    If convergence has been achieved, the between-chain and within-chain
    variances should be identical. To be most effective in detecting evidence
    for nonconvergence, each chain should have been initialized to starting
    values that are dispersed relative to the target distribution.

    Parameters
    ----------
    get_sampler : callable
        Callable that takes a `tensorflow.Session` as input and returns a
        (possibly already "burnt-in")
        `pysgmcmc.sampling.MCMCSampler` subclass instance.

    n_chains : int, optional
        Number of individual chains/traces to extract.
        Defaults to `2`.

    samples_per_chain : int, optional
        Number of samples each individual chain should contain.
        Defaults to `100`.

    Returns
    ----------
    gelman_rubin : dict
      Dictionary of the potential scale reduction factors, :math:`\\hat{R}`.

    Notes
    ----------

    The diagnostic is computed by:

      .. math:: \\hat{R} = \\frac{\\hat{V}}{W}

    where :math:`W` is the within-chain variance and :math:`\\hat{V}` is
    the posterior variance estimate for the pooled traces. This is the
    potential scale reduction factor, which converges to unity when each
    of the traces is a sample from the target posterior. Values greater
    than one indicate that one or more chains have not yet converged.

    References
    ----------
    Brooks and Gelman (1998)
    Gelman and Rubin (1992)

    Examples
    ----------
    Simple (arbitrary) example to showcase usage:

    >>> import tensorflow as tf
    >>> from pysgmcmc.samplers.sghmc import SGHMCSampler
    >>> params = [tf.Variable([1.0, 2.0], name="x", dtype=tf.float64)]
    >>> cost_fun = lambda params: tf.reduce_sum(params)
    >>> get_sampler = lambda session: SGHMCSampler(params=params, cost_fun=cost_fun, session=session)
    >>> factors = gelman_rubin(get_sampler=get_sampler)
    >>> type(factors)
    <class 'dict'>
    >>> list(factors.keys())[0].startswith("x")  # necessary due to tensorflow variable enumeration
    True
    >>> len(factors[params[0].name])  # x:0 has two dimensions, we have one factor for each
    2
    """
    return _pymc3_diagnostic(
        pymc3_diagnostic_fun=pymc3_gelman_rubin,
        get_sampler=get_sampler,
        n_chains=n_chains,
        samples_per_chain=samples_per_chain
    )
