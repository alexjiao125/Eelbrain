# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from itertools import repeat
from math import ceil
import random

import numpy as np

from .._data_obj import NDVar, Var, NestedEffect
from .._utils import intervals


_YIELD_ORIGINAL = 0
# for testing purposes, yield original order instead of permutations


def _resample_params(N, samples):
    """Decide whether to do permutations or random resampling

    Parameters
    ----------
    N : int
        Number of observations.
    samples : int
        ``samples`` parameter (number of resampling iterations, or < 0 to
        sample all permutations).

    Returns
    -------
    actual_n_samples : int
        Adapted number of resamplings that will be done.
    samples_param : int
        Samples parameter for the resample function (-1 to do all permutations,
        otherwise same as n_samples).
    """
    n_perm = 2 ** N
    if n_perm - 1 <= samples:
        samples = -1

    if samples < 0:
        n_samples = n_perm - 1
    else:
        n_samples = samples

    return n_samples, samples


def permute_order(n, samples=10000, replacement=False, unit=None, seed=0):
    """Generator function to create indices to shuffle n items

    Parameters
    ----------
    n : int
        Number of cases.
    samples : int
        Number of samples to yield. If < 0, all possible permutations are
        performed.
    replacement : bool
        whether random samples should be drawn with replacement or without.
    unit : categorial
        Factor specifying unit of measurement (e.g. subject). If unit is
        specified, resampling proceeds by first resampling the categories of
        unit (with or without replacement) and then shuffling the values
        within units (no replacement).
    seed : None | int
        Seed the random state of the relevant randomization module
        (:mod:`random` or :mod:`numpy.random`) to make replication possible.
        None to skip seeding (default 0).

    Returns
    -------
    Iterator over index.
    """
    n = int(n)
    samples = int(samples)
    if samples < 0:
        err = "Complete permutation for resampling through reordering"
        raise NotImplementedError(err)

    if _YIELD_ORIGINAL:
        original = np.arange(n)
        for _ in range(samples):
            yield original
        return

    if seed is not None:
        np.random.seed(seed)

    if unit is None:
        if replacement:
            for _ in range(samples):
                yield np.random.randint(n, n)
        else:
            index = np.arange(n)
            for _ in range(samples):
                np.random.shuffle(index)
                yield index
    else:
        if replacement:
            raise NotImplementedError("Replacement and units")
        idx_orig = np.arange(n)
        idx_perm = np.empty_like(idx_orig)
        unit_idxs = [np.flatnonzero(unit == cell) for cell in unit.cells]
        if isinstance(unit, NestedEffect):
            dst_idxs_iter = ((unit_idxs[i] for i in order)
                             for order in permute_order(len(unit_idxs), samples, seed=None))
        else:
            dst_idxs_iter = repeat(unit_idxs, samples)

        for dst_idxs in dst_idxs_iter:
            for src, dst in zip(unit_idxs, dst_idxs):
                v = idx_orig[src]
                np.random.shuffle(v)
                idx_perm[dst] = v
            yield idx_perm


def permute_sign_flip(n, samples=10000, seed=0, out=None):
    """Iterate over indices for ``samples`` permutations of the data

    Parameters
    ----------
    n : int
        Number of cases.
    samples : int
        Number of samples to yield. If < 0, all possible permutations are
        performed.
    seed : None | int
        Seed the random state of the :mod:`random` module to make replication 
        possible. None to skip seeding (default 0).
    out : array of int8  (n,)
        Buffer for the ``sign`` variable that is yielded in each iteration.

    Yields
    ------
    sign : array of int8  (n,)
        Sign for each case (``1`` or ``-1``; ``sign`` is the same array object 
        but its content modified in every iteration).
    """
    n = int(n)
    if seed is not None:
        random.seed(seed)

    if out is None:
        out = np.empty(n, np.int8)
    else:
        assert out.shape == (n,)

    if n > 62:  # Python 2 limit for xrange
        if samples < 0:
            raise NotImplementedError("All possibilities for more than 62 cases")
        n_groups = ceil(n / 62.)
        group_size = int(ceil(n / n_groups))
        out_parts = list(range(0, n, group_size)) + [n]
        for _ in zip(*(permute_sign_flip(stop - start, samples, None, out[start: stop])
                       for start, stop in intervals(out_parts))):
            yield out
        return

    # determine possible number of permutations
    n_perm_possible = 2 ** n
    if samples < 0:
        # do all permutations
        sample_sequences = range(1, n_perm_possible)
    else:
        # random resampling
        sample_sequences = random.sample(range(1, n_perm_possible), samples)

    for seq in sample_sequences:
        out.fill(1)
        for i in (i for i, s in enumerate(bin(seq)[-1:1:-1]) if s == '1'):
            out[i] = -1
        yield out


def resample(y, samples=10000, replacement=False, unit=None, seed=0):
    """
    Generator function to resample a dependent variable (y) multiple times

    Parameters
    ----------
    y : Var | NDVar
        Variable which is to be resampled.
    samples : int
        Number of samples to yield. If < 0, all possible permutations are
        performed.
    replacement : bool
        whether random samples should be drawn with replacement or without.
    unit : categorial
        Factor specifying unit of measurement (e.g. subject). If unit is
        specified, resampling proceeds by first resampling the categories of
        unit (with or without replacement) and then shuffling the values
        within units (no replacement).
    seed : None | int
        Seed the random state of the relevant randomization module
        (:mod:`random` or :mod:`numpy.random`) to make replication possible.
        None to skip seeding (default 0).

    Returns
    -------
    Iterator over Y_resampled. One copy of ``y`` is made, and this copy is
    yielded in each iteration with shuffled data.
    """
    if isinstance(y, Var):
        pass
    elif isinstance(y, NDVar):
        if not y.has_case:
            raise ValueError("Need NDVar with cases")
    else:
        raise TypeError("Need Var or NDVar")

    out = y.copy('{name}_resampled')

    for index in permute_order(len(out), samples, replacement, unit, seed):
        out.x[index] = y.x
        yield out
