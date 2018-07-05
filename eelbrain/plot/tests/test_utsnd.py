'''
Created on Jul 11, 2013

@author: christian
'''
import numpy as np
from nose.tools import eq_

from eelbrain import datasets, plot, testnd
from eelbrain.plot._figure import Figure
from eelbrain.plot._utsnd import _ax_bfly_epoch
from eelbrain._utils.testing import requires_mne_sample_data


def test_plot_butterfly():
    "Test plot.Butterfly"
    ds = datasets.get_uts(utsnd=True)
    p = plot.Butterfly('utsnd', ds=ds, show=False)
    p.close()
    p = plot.Butterfly('utsnd', 'A%B', ds=ds, show=False)
    p.close()

    # other y-dim
    stc = datasets.get_mne_stc(True)
    p = plot.Butterfly(stc, show=False)
    p.close()

    # _ax_bfly_epoch (used in GUI, not part of a figure)
    fig = Figure(1, show=False)
    ax = _ax_bfly_epoch(fig._axes[0], ds[0, 'utsnd'])
    fig.show()
    ax.set_data(ds[1, 'utsnd'])
    fig.draw()


def test_plot_array():
    "Test plot.Array"
    ds = datasets.get_uts(utsnd=True)
    p = plot.Array('utsnd', ds=ds, show=False)
    p.close()
    p = plot.Array('utsnd', 'A%B', ds=ds, show=False)
    eq_(p._layout.nax, 4)
    p.close()
    p = plot.Array('utsnd', 'A', sub='B=="b1"', ds=ds, show=False)
    eq_(p._layout.nax, 2)
    p.close()

    # Categorial dimension
    ds = datasets._get_continuous()
    p = plot.Array(ds['x2'], interpolation='none', show=False)
    eq_(len(p.figure.axes[0].get_yticks()), 2)


@requires_mne_sample_data
def test_plot_mne_evoked():
    "Test plotting evoked from the mne sample dataset"
    evoked = datasets.get_mne_evoked()
    p = plot.Array(evoked, show=False)
    p.close()


@requires_mne_sample_data
def test_plot_mne_epochs():
    "Test plotting epochs from the mne sample dataset"
    epochs = datasets.get_mne_epochs()

    # grand average
    p = plot.Array(epochs, show=False)
    p.close()

    # with model
    p = plot.Array(epochs, np.arange(2).repeat(8), show=False)
    p.close()


def test_plot_results():
    "Test plotting test results"
    ds = datasets.get_uts(True)

    # ANOVA
    res = testnd.anova('utsnd', 'A*B*rm', ds=ds, samples=0, pmin=0.05)
    p = plot.Array(res, show=False)
    p.close()
    res = testnd.anova('utsnd', 'A*B*rm', ds=ds, samples=2, pmin=0.05)
    p = plot.Array(res, show=False)
    p.close()

    # Correlation
    res = testnd.corr('utsnd', 'Y', 'rm', ds=ds)
    p = plot.Array(res, show=False)
    p.close()
    res = testnd.corr('utsnd', 'Y', 'rm', ds=ds, samples=10, pmin=0.05)
    p = plot.Array(res, show=False)
    p.close()
