'''
Created on Dec 2, 2012

@author: christian
'''
from nose.tools import eq_

from eelbrain import (
    datasets, plot, testnd,
    Factor,
    concatenate, set_tmin,
)


def test_uts_stat():
    "test plot.UTSStat plotting function"
    ds = datasets.get_uts()
    p = plot.UTSStat('uts', ds=ds, show=False)
    p.close()
    p = plot.UTSStat('uts', 'A%B', ds=ds, show=False)
    p.plot_legend('lower right')
    p.plot_legend(False)
    pl = p.plot_legend('fig')
    p.plot_legend('center')
    pl.close()
    p.close()
    p = plot.UTSStat('uts', 'A', xax='B', ds=ds, show=False)
    p.close()
    p = plot.UTSStat('uts', 'A%B', 'rm', sub="rm.isin(('R00', 'R01'))", ds=ds,
                     show=False)
    p.close()
    p = plot.UTSStat('uts', 'A%B', 'rm', sub="rm.isin(('R00', 'R01'))", ds=ds,
                     pool_error=False, show=False)
    p.close()

    # error
    p = plot.UTSStat('uts', 'A', match='rm', ds=ds, error=False, show=False)
    p.close()
    p = plot.UTSStat('uts', 'A', match='rm', ds=ds, error='all', show=False)
    p.close()

    # clusters
    sds = ds.sub("B == 'b0'")
    res = testnd.ttest_rel('uts', 'A', 'a1', 'a0', match='rm', ds=sds,
                           samples=0, pmin=0.05, mintime=0.02)
    p = plot.UTSStat('uts', 'A', ds=ds, show=False)
    p.set_clusters(res.clusters)
    p.close()
    p = plot.UTSStat('uts', 'A', ds=ds, clusters=res.clusters, show=False)
    p.close()
    res = testnd.ttest_rel('uts', 'A', 'a1', 'a0', match='rm', ds=sds,
                           samples=100, pmin=0.05, mintime=0.02)
    p = plot.UTSStat('uts', 'A', ds=ds, clusters=res.clusters, show=False)
    p.close()
    p = plot.UTSStat('uts', 'A', 'B', ds=ds, clusters=res.clusters, show=False)
    p.set_clusters(None)
    p.set_clusters(res.clusters, ax=0)
    p.close()
    p = plot.UTSStat('uts', 'A', 'B', ds=ds, show=False)
    p.set_clusters(res.clusters)
    p.set_clusters(None, ax=1)
    p.close()


def test_uts():
    "test plot.UTS plotting function"
    ds = datasets.get_uts()
    x_long = set_tmin(concatenate(ds[:10, 'uts']), -1)

    p = plot.UTS('uts', ds=ds, show=False)
    p.close()
    p = plot.UTS('uts', 'A%B', ds=ds, show=False)
    p.set_ylim(1)
    p.set_ylim(0, 1)
    eq_(p.get_ylim(), (0, 1))
    p.set_ylim(1, -1)
    eq_(p.get_ylim(), (1, -1))
    p.close()

    p = plot.UTS(x_long, h=2, w=5, xlim=2)
    eq_(p.get_xlim(), (-1, 1))
    p.set_xlim(2, 4)
    eq_(p.get_xlim(), (2, 4))
    p.close()


def test_clusters():
    "test plot.uts cluster plotting functions"
    ds = datasets.get_uts()

    A = ds['A']
    B = ds['B']
    Y = ds['uts']

    # fixed effects model
    res = testnd.anova(Y, A * B)
    p = plot.UTSClusters(res, title="Fixed Effects Model", show=False)
    p.close()

    # random effects model:
    subject = Factor(range(15), tile=4, random=True, name='subject')
    res = testnd.anova(Y, A * B * subject, match=subject, samples=2)
    p = plot.UTSClusters(res, title="Random Effects Model", show=False)
    p.close()

    # plot UTSStat
    p = plot.UTSStat(Y, A % B, match=subject, show=False)
    p.set_clusters(res.clusters)
    p.close()
    p = plot.UTSStat(Y, A, xax=B, match=subject, show=False)
    p.close()
