# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from inspect import getargspec
import re

from .. import testnd
from .._exceptions import DefinitionError
from .definitions import Definition
from .vardef import GroupVar


__test__ = False
TAIL_REPR = {0: '=', 1: '>', -1: '<'}


def assemble_tests(test_dict):
    "Interpret dict with test definitions"
    out = {}
    for key, params in test_dict.items():
        if isinstance(params, Test):
            out[key] = params
            continue
        elif not isinstance(params, dict):
            raise TypeError("Invalid object for test definitions %s: %r" %
                            (key, params))
        params = params.copy()
        if 'stage 1' in params:
            params['stage_1'] = params.pop('stage 1')
        kind = params.pop('kind')
        if kind in TEST_CLASSES:
            out[key] = TEST_CLASSES[kind](**params)
        else:
            raise DefinitionError("Unknown test kind in test definition %s: "
                                  "%r" % (key, kind))
    return out


def tail_arg(tail):
    try:
        if tail == 0:
            return 0
        elif tail > 0:
            return 1
        else:
            return -1
    except Exception:
        raise TypeError("tail=%r; needs to be 0, -1 or 1" % (tail,))


class Test(Definition):
    "Baseclass for any test"
    kind = None
    DICT_ATTRS = ('kind', 'model', 'vars')

    def __init__(self, desc, model, vars=None):
        self.desc = desc
        self.model = model
        self.vars = vars

        if model is None:  # no averaging
            self._between = None
            self._within_model = None
            self._within_model_items = None
        else:
            model_elements = list(map(str.strip, model.split('%')))
            if 'group' in model_elements:
                self._between = model_elements.index('group')
                del model_elements[self._between]
            else:
                self._between = None
            self._within_model_items = model_elements
            self._within_model = '%'.join(model_elements)


class EvokedTest(Test):
    "Group level test applied to subject averages"
    def __init__(self, desc, model, cat=None, vars=None):
        Test.__init__(self, desc, model, vars)
        self.cat = cat
        if cat is not None:
            if self._within_model is None or len(self._within_model_items) == 0:
                cat = None
            elif self._between is not None:
                # remove between factor from cat
                cat = [[c for i, c in enumerate(cat) if i != self._between]
                       for cat in self.cat]
        self._within_cat = cat

    def make(self, y, ds, force_permutation, kwargs):
        raise NotImplementedError


class TTestOneSample(EvokedTest):
    kind = 'ttest_1samp'
    DICT_ATTRS = Test.DICT_ATTRS + ('tail',)

    def __init__(self, tail=0):
        desc = "%s 0" % TAIL_REPR[tail]
        EvokedTest.__init__(self, desc, '')
        self.tail = tail

    def make(self, y, ds, force_permutation, kwargs):
        return testnd.ttest_1samp(
            y, match='subject', ds=ds, tail=self.tail,
            force_permutation=force_permutation, **kwargs)


class TTest(EvokedTest):
    DICT_ATTRS = Test.DICT_ATTRS + ('c1', 'c0', 'tail')

    def __init__(self, model, c1, c0, tail, vars=None):
        tail = tail_arg(tail)
        desc = '%s %s %s' % (c1, TAIL_REPR[tail], c0)
        EvokedTest.__init__(self, desc, model, (c1, c0), vars)
        self.c1 = c1
        self.c0 = c0
        self.tail = tail


class TTestInd(TTest):
    "Independent measures t-test"
    kind = 'ttest_ind'

    def __init__(self, model, c1, c0, tail=0):
        if model != 'group':
            raise DefinitionError("model=%r; TTestInd always needs model="
                                  "'group'" % (model,))
        TTest.__init__(self, model, c1, c0, tail, (('group', GroupVar((c1, c0))),))

    def make(self, y, ds, force_permutation, kwargs):
        return testnd.ttest_ind(
            y, self.model, self.c1, self.c0, 'subject', ds=ds, tail=self.tail,
            force_permutation=force_permutation, **kwargs)


class TTestRel(TTest):
    """Related measures t-test

    Notes
    -----
    For a t-test between two epochs, use an :class:`EpochCollection` epoch
    and ``model='epoch'``.
    """
    kind = 'ttest_rel'

    def __init__(self, model, c1, c0, tail=0):
        TTest.__init__(self, model, c1, c0, tail)
        assert self._between is None

    def make(self, y, ds, force_permutation, kwargs):
        return testnd.ttest_rel(
            y, self.model, self.c1, self.c0, 'subject', ds=ds, tail=self.tail,
            force_permutation=force_permutation, **kwargs)


class TContrastRel(EvokedTest):
    "T-contrast"
    kind = 't_contrast_rel'
    DICT_ATTRS = Test.DICT_ATTRS + ('contrast', 'tail')

    def __init__(self, model, contrast, tail=0):
        tail = tail_arg(tail)
        EvokedTest.__init__(contrast, model)
        self.contrast = contrast
        self.tail = tail

    def make(self, y, ds, force_permutation, kwargs):
        return testnd.t_contrast_rel(
            y, self.model, self.contrast, 'subject', ds=ds, tail=self.tail,
            force_permutation=force_permutation, **kwargs)


class ANOVA(EvokedTest):
    """ANOVA test

    Parameters
    ----------
    x : str
        ANOVA model specification (see :func:`test.anova`).
    model : str
        Model for grouping trials before averaging (does not need to be
        specified unless it should include variables not in ``x``).
    vars : tuple | dict
        Variables to add dynamically.
    """
    kind = 'anova'
    DICT_ATTRS = Test.DICT_ATTRS + ('x',)

    def __init__(self, x, model=None, vars=None):
        x = ''.join(x.split())
        if model is None:
            items = sorted(i.strip() for i in x.split('*'))
            within_items = (i for i in items if not re.match('^subject(\(\w+\))$', i))
            model = '%'.join(within_items)
        EvokedTest.__init__(self, x, model, vars=vars)
        if self._between is not None:
            raise NotImplementedError("Between-subject ANOVA")
        self.x = x

    def make(self, y, ds, force_permutation, kwargs):
        return testnd.anova(y, self.x, ds=ds, force_permutation=force_permutation, **kwargs)


class TwoStageTest(Test):
    "Two-stage test on epoched or evoked data"
    kind = 'two-stage'
    DICT_ATTRS = Test.DICT_ATTRS + ('stage_1',)

    def __init__(self, stage_1, vars=None, model=None):
        Test.__init__(self, stage_1, model, vars)
        self.stage_1 = stage_1

    def make_stage_1(self, y, ds, subject):
        """Assumes that model has already been applied"""
        return testnd.LM(y, self.stage_1, ds, subject=subject)

    def make_stage_2(self, lms, kwargs):
        lm = testnd.LMGroup(lms)
        lm.compute_column_ttests(**kwargs)
        return lm


TEST_CLASSES = {
    'anova': ANOVA,
    'ttest_1samp': TTestOneSample,
    'ttest_rel': TTestRel,
    'ttest_ind': TTestInd,
    't_contrast_rel': TContrastRel,
    'two-stage': TwoStageTest,
}
AGGREGATE_FUNCTIONS = ('mean', 'rms')
DATA_RE = re.compile("(source|sensor)(?:\.(%s))?$" % '|'.join(AGGREGATE_FUNCTIONS))


class TestDims(object):
    """Data shape for test

    Paremeters
    ----------
    string : str
        String describing data.
    time : bool
        Whether the base data contains a time axis.
    """
    source = None
    sensor = None

    def __init__(self, string, time=True):
        self.time = time
        substrings = string.split()
        for substring in substrings:
            m = DATA_RE.match(substring)
            if m is None:
                raise ValueError("Invalid test dimension description: %r" %
                                 (string,))
            dim, aggregate = m.groups()
            setattr(self, dim, aggregate or True)
        if sum(map(bool, (self.source, self.sensor))) != 1:
            raise ValueError("Invalid test dimension description: %r. Need "
                             "exactly one of 'sensor' or 'source'" % (string,))
        self.string = ' '.join(substrings)

        dims = []
        if self.source is True:
            dims.append('source')
        elif self.sensor is True:
            dims.append('sensor')
        if self.time is True:
            dims.append('time')
        self.dims = tuple(dims)

        # whether parc is used from subjects or from common-brain
        if self.source is True:
            self.parc_level = 'common'
        elif self.source:
            self.parc_level = 'individual'
        else:
            self.parc_level = None

    @classmethod
    def coerce(cls, obj, time=True):
        if isinstance(obj, cls):
            if bool(obj.time) == time:
                return obj
            else:
                return cls(obj.string, time)
        else:
            return cls(obj, time)

    def __repr__(self):
        return "TestDims(%r)" % (self.string,)

    def __eq__(self, other):
        if not isinstance(other, TestDims):
            return False
        return self.string == other.string and self.time == other.time


class ROITestResult(object):
    """Store samples as attribute"""

    def __init__(self, subjects, samples, n_trials_ds, merged_dist, res):
        self.subjects = subjects
        self.samples = samples
        self.n_trials_ds = n_trials_ds
        self.merged_dist = merged_dist
        self.res = res

    def __getstate__(self):
        return {attr: getattr(self, attr) for attr in
                getargspec(self.__init__).args[1:]}

    def __setstate__(self, state):
        self.__init__(**state)
