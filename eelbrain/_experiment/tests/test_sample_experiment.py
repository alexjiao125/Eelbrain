# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Test MneExperiment using mne-python sample data"""
from pathlib import Path
from os.path import join

from nose.tools import eq_, assert_raises
import numpy as np

from eelbrain import *
from eelbrain._exceptions import DefinitionError
from eelbrain._utils.testing import (
    TempDir, assert_dataobj_equal, import_attr, requires_mne_sample_data,
)


sample_path = Path(__file__).parents[3] / 'examples/experiment'


@requires_mne_sample_data
def test_sample():
    set_log_level('warning', 'mne')
    # import from file:  http://stackoverflow.com/a/67692/166700
    SampleExperiment = import_attr(sample_path / 'sample_experiment.py', 'SampleExperiment')
    tempdir = TempDir()
    datasets.setup_samples_experiment(tempdir, 3, 2)

    root = join(tempdir, 'SampleExperiment')
    e = SampleExperiment(root)

    eq_(e.get('subject'), 'R0000')
    eq_(e.get('subject', subject='R0002'), 'R0002')

    # events
    e.set('R0001', rej='')
    ds = e.load_selected_events(epoch='target')
    assert ds.n_cases == 39
    ds = e.load_selected_events(epoch='auditory')
    assert ds.n_cases == 20
    ds = e.load_selected_events(epoch='av')
    assert ds.n_cases == 39
    
    # evoked cache invalidated by change in bads
    e.set('R0001', rej='', epoch='target')
    ds = e.load_evoked()
    eq_(ds[0, 'evoked'].info['bads'], [])
    e.make_bad_channels(['MEG 0331'])
    ds = e.load_evoked()
    eq_(ds[0, 'evoked'].info['bads'], ['MEG 0331'])

    e.set(rej='man', model='modality')
    sds = []
    for _ in e:
        e.make_rej(auto=2.5e-12)
        sds.append(e.load_evoked())

    ds = e.load_evoked('all')
    assert_dataobj_equal(combine(sds), ds)

    # test with data parameter
    megs = [e.load_evoked(cat='auditory')['meg'] for _ in e]
    res = e.load_test('a>v', 0.05, 0.2, 0.05, samples=100, data='sensor.rms',
                      sns_baseline=False, make=True)
    meg_rms = combine(meg.rms('sensor') for meg in megs).mean('case', name='auditory')
    assert_dataobj_equal(res.c1_mean, meg_rms, decimal=21)
    res = e.load_test('a>v', 0.05, 0.2, 0.05, samples=100, data='sensor.mean',
                      sns_baseline=False, make=True)
    meg_mean = combine(meg.mean('sensor') for meg in megs).mean('case', name='auditory')
    assert_dataobj_equal(res.c1_mean, meg_mean, decimal=21)

    # e._report_subject_info() broke with non-alphabetic subject order
    subjects = e.get_field_values('subject')
    ds = Dataset()
    ds['subject'] = Factor(reversed(subjects))
    ds['n'] = Var(range(3))
    s_table = e._report_subject_info(ds, '')

    # test multiple epochs with same time stamp
    class Experiment(SampleExperiment):
        epochs = SampleExperiment.epochs.copy()
    Experiment.epochs['v1'] = {'base': 'visual', 'vars': {'shift': 'Var([0.0], repeat=len(side))'}}
    Experiment.epochs['v2'] = {'base': 'visual', 'vars': {'shift': 'Var([0.1], repeat=len(side))'}}
    Experiment.epochs['vc'] = {'sub_epochs': ('v1', 'v2'), 'post_baseline_trigger_shift': 'shift', 'post_baseline_trigger_shift_max': 0.1, 'post_baseline_trigger_shift_min': 0.0}
    e = Experiment(root)
    ds = e.load_epochs(baseline=True, epoch='vc')
    v1 = ds.sub("epoch=='v1'", 'meg').sub(time=(0, 0.199))
    v2 = ds.sub("epoch=='v2'", 'meg').sub(time=(-0.1, 0.099))
    assert_dataobj_equal(v1, v2, decimal=20)

    # duplicate subject
    class BadExperiment(SampleExperiment):
        groups = {'group': ('R0001', 'R0002', 'R0002')}
    assert_raises(DefinitionError, BadExperiment, root)

    # non-existing subject
    class BadExperiment(SampleExperiment):
        groups = {'group': ('R0001', 'R0003', 'R0002')}
    assert_raises(DefinitionError, BadExperiment, root)

    # unsorted subjects
    class Experiment(SampleExperiment):
        groups = {'group': ('R0002', 'R0000', 'R0001')}
    e = Experiment(root)
    eq_([s for s in e], ['R0000', 'R0001', 'R0002'])

    # changes
    class Changed(SampleExperiment):
        variables = {
            'event': {(1, 2, 3, 4): 'target', 5: 'smiley', 32: 'button'},
            'side': {(1, 3): 'left', (2, 4): 'right_changed'},
            'modality': {(1, 2): 'auditory', (3, 4): 'visual'}
        }
        tests = {
            'twostage': {
                'kind': 'two-stage',
                'stage 1': 'side_left + modality_a',
                'vars': {
                    'side_left': "side == 'left'",
                    'modality_a': "modality == 'auditory'",
                }
            },
            'novars': {
                'kind': 'two-stage',
                'stage 1': 'side + modality'
            },
        }
    e = Changed(root)

    # changed variable, while a test with model=None is not changed
    class Changed(Changed):
        variables = {
            'side': {(1, 3): 'left', (2, 4): 'right_changed'},
            'modality': {(1, 2): 'auditory', (3, 4): 'visual_changed'}
        }
    e = Changed(root)

    # changed variable, unchanged test with vardef=None
    class Changed(Changed):
        variables = {
            'side': {(1, 3): 'left', (2, 4): 'right_changed'},
            'modality': {(1, 2): 'auditory', (3, 4): 'visual_changed'}
        }
    e = Changed(root)

    # ICA
    # ---
    e = SampleExperiment(root)
    ica_path = e.make_ica(raw='ica')
    e.set(raw='ica1-40', model='')
    e.make_rej(auto=2e-12, overwrite=True)
    ds1 = e.load_evoked(raw='ica1-40')
    ica = e.load_ica(raw='ica')
    ica.exclude = [0, 1, 2]
    ica.save(ica_path)
    ds2 = e.load_evoked(raw='ica1-40')
    assert not np.allclose(ds1['meg'].x, ds2['meg'].x, atol=1e-20), "ICA change ignored"


@requires_mne_sample_data
def test_samples_sesssions():
    set_log_level('warning', 'mne')
    SampleExperiment = import_attr(sample_path / 'sample_experiment_sessions.py', 'SampleExperiment')
    tempdir = TempDir()
    datasets.setup_samples_experiment(tempdir, 2, 1, 2)

    root = join(tempdir, 'SampleExperiment')
    e = SampleExperiment(root)
    # bad channels
    e.make_bad_channels('0111')
    eq_(e.load_bad_channels(), ['MEG 0111'])
    eq_(e.load_bad_channels(session='sample2'), [])
    e.show_bad_channels()
    e.merge_bad_channels()
    eq_(e.load_bad_channels(session='sample2'), ['MEG 0111'])
    e.show_bad_channels()

    # rejection
    for _ in e:
        for epoch in ('target1', 'target2'):
            e.set(epoch=epoch)
            e.make_rej(auto=2e-12)

    ds = e.load_evoked('R0000', epoch='target2')
    e.set(session='sample1')
    ds2 = e.load_evoked('R0000')
    assert_dataobj_equal(ds2, ds)

    # super-epoch
    ds1 = e.load_epochs(epoch='target1')
    ds2 = e.load_epochs(epoch='target2')
    ds_super = e.load_epochs(epoch='super')
    assert_dataobj_equal(ds_super['meg'], combine((ds1['meg'], ds2['meg'])))
