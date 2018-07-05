# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"Utilities for testing"
from contextlib import ContextDecorator
from distutils.version import LooseVersion
from functools import reduce, wraps
from importlib import import_module
from importlib.util import spec_from_file_location, module_from_spec
import os
from operator import mul
import shutil
import tempfile

from nose.plugins.skip import SkipTest
from nose.tools import assert_equal, eq_
import numpy as np
from numpy.testing import assert_array_equal

import eelbrain._wxgui
from .._data_obj import Dataset, NDVar, Var, Factor, isdatalist, isdatacontainer, isuv


class TempDir(str):
    "After MNE-Python mne.utils"
    def __new__(cls):
        return str.__new__(cls, tempfile.mkdtemp())

    def __del__(self):
        shutil.rmtree(self, ignore_errors=True)


def assert_dataset_equal(ds1, ds2, msg="Datasets unequal", decimal=None):
    """
    Raise an assertion if two Datasets are not equal up to desired precision.

    Parameters
    ----------
    ds1, ds2 : Dataset
        Datasets to compare.
    msg : str
        Prefix of the error message to be printed in case of failure.
    decimal : None | int
        Desired precision (default is exact match).
    """
    assert_equal(ds1.keys(), ds2.keys(), "%s: different keys (%s vs %s)" %
                 (msg, ds1.keys(), ds2.keys()))
    for k in ds1.keys():
        assert_dataobj_equal(ds1[k], ds2[k], msg=msg, decimal=decimal)
    assert_equal(ds1.info.keys(), ds2.info.keys(), "%s: keys in info" % msg)


def assert_dataobj_equal(d1, d2, msg="Data-objects unequal", decimal=None):
    """Assert that two data-objects are equal up to desired precision.

    Parameters
    ----------
    d1, d2 : data-objects
        Data-objects to compare.
    msg : str
        Prefix of the error message to be printed in case of failure.
    decimal : None | int
        Desired precision (default is exact match).
    """
    if not isdatacontainer(d1):
        raise TypeError("d1 is not a data-object but %s" % repr(d1))
    elif not isdatacontainer(d2):
        raise TypeError("d2 is not a data-object but %s" % repr(d2))
    else:
        eq_(type(d1), type(d2))
    msg += ":"
    assert_equal(d1.name, d2.name, "%s unequal names (%r vs %r"
                 ")" % (msg, d1.name, d2.name))
    msg += ' Two %ss named %r have' % (d1.__class__.__name__, d1.name)
    len1 = len(d1)
    len2 = len(d2)
    assert_equal(len1, len2, "%s unequal length: %i/%i" % (msg, len1, len2))
    if isuv(d1):
        if isinstance(d1, Var):
            is_equal = np.allclose(d1.x, d2.x, equal_nan=True, rtol=0,
                                   atol=10**-decimal if decimal else 0)
        else:
            is_equal = d1 == d2
        if not np.all(is_equal):
            ds = Dataset()
            ds['value'] = d1
            ds['target'] = d2
            ds['unequal'] = Factor(is_equal, labels={True: '', False: 'x'})
            if isinstance(d1, Var):
                ds['difference'] = d1 - d2
            raise AssertionError(f'{msg} unequal values:\n\n{ds}')
    elif isinstance(d1, NDVar):
        if d1.shape != d2.shape:
            raise AssertionError(
                f"NDVars have different shape:\n  {d1.name}: {d1.shape}\n"
                f"  {d2.name}: {d2.shape}")
        if decimal:
            is_different = np.max(np.abs(d1.x - d2.x)) >= 10**-decimal
        else:
            is_different = np.any(d1.x != d2.x)

        if is_different:
            n = reduce(mul, d1.x.shape)
            n_different = (d1.x != d2.x).sum()
            mean_diff = np.abs(d1.x - d2.x).sum() / n_different
            raise AssertionError("%s unequal values. Difference in %i of %i "
                                 "values, average difference=%s." %
                                 (msg, n_different, n, mean_diff))
    elif isdatalist(d1):
        for i in range(len(d1)):
            assert_equal(d1[i], d2[i], "%s unequal values" % msg)


def assert_source_space_equal(src1, src2, msg="SourceSpace Dimension objects "
                              "unequal"):
    """Assert that two SourceSpace objects are identical

    Parameters
    ----------
    src1, src2 : SourceSpace objects
        SourceSpace objects to compare.
    msg : str
        Prefix of the error message to be printed in case of failure.
    """
    msg = "%s:" % msg
    assert_array_equal(src1.vertices[0], src2.vertices[0], "%s unequal lh vertices "
                       "(%r vs %r)" % (msg, src1.vertices[0], src2.vertices[0]))
    assert_array_equal(src1.vertices[1], src2.vertices[1], "%s unequal rh vertices "
                       "(%r vs %r)" % (msg, src1.vertices[1], src2.vertices[1]))
    assert_equal(src1.subject, src2.subject, "%s unequal subject (%r vs %r"
                 ")" % (msg, src1.subject, src2.subject))
    assert_equal(src1.src, src2.src, "%s unequal names (%r vs %r"
                 ")" % (msg, src1.src, src2.src))
    assert_equal(src1.subjects_dir, src2.subjects_dir, "%s unequal names (%r "
                 "vs %r)" % (msg, src1.subjects_dir, src2.subjects_dir))


class GUITestContext(ContextDecorator):
    modules = (
        eelbrain._wxgui.select_epochs,
        eelbrain._wxgui.select_components,
        eelbrain._wxgui.history,
    )

    def __init__(self):
        self._i = 0

    def __enter__(self):
        self._i += 1
        if self._i == 1:
            for mod in self.modules:
                mod.TEST_MODE = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._i -= 1
        if self._i == 0:
            for mod in self.modules:
                mod.TEST_MODE = False


gui_test = GUITestContext()


def import_attr(path, attr):
    spec = spec_from_file_location('module', path)
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, attr)


def requires_mne_sample_data(function):
    import mne

    if mne.datasets.sample.data_path(download=False):
        @wraps(function)
        def decorator(*args, **kwargs):
            return function(*args, **kwargs)
    else:
        @wraps(function)
        def decorator(*args, **kwargs):
            raise SkipTest('Skipped %s, requires mne sample data' % function.__name__)
    return decorator


def requires_module(name, version):
    mod = import_module(name)

    def wrapper(function):
        if LooseVersion(mod.__version__) >= LooseVersion(version):
            @wraps(function)
            def decorator(*args, **kwargs):
                return function(*args, **kwargs)
        else:
            @wraps(function)
            def decorator(*args, **kwargs):
                raise SkipTest('Skipped %s, requires %s %s, found mne %s' %
                               (function.__name__, name, version,
                                mod.__version__))
        return decorator
    return wrapper


def requires_r_ez(function):
    from .r_bridge import r, r_warning_filter

    with r_warning_filter:
        success = r('require(ez)')[0]

    if success:
        @wraps(function)
        def decorator(*args, **kwargs):
            return function(*args, **kwargs)
    else:
        @wraps(function)
        def decorator(*args, **kwargs):
            raise SkipTest('Skipped %s, requires r-ez' % function.__name__)
    return decorator


def skip_on_windows(function):
    @wraps(function)
    def decorator(*args, **kwargs):
        if os.name == 'nt':
            raise SkipTest('Skipped %s on Windows' % function.__name__)
        else:
            return function(*args, **kwargs)
    return decorator


def file_path(name):
    "Path to test data file in the test_data directory"
    path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..',
                                         'test_data', name))
    if os.path.exists(path):
        return path
    else:
        raise IOError("Testing file does not exist. Test can only be executed "
                      "from source repository.")
