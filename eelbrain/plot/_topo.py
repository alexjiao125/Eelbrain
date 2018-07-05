# -*- coding: utf-8 -*-
# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Plot topographic maps of sensor space data."""
from collections import Sequence
from itertools import repeat
from math import floor, sqrt
from typing import Union, Tuple
from warnings import warn

import matplotlib as mpl
import numpy as np
from scipy import interpolate, linalg
from scipy.spatial import ConvexHull

from .._data_obj import NDVar
from . import _base
from ._base import (
    EelFigure, PlotData, Layout, ImLayout, VariableAspectLayout,
    ColorMapMixin, TimeSlicerEF, TopoMapKey, XAxisMixin, YLimMixin)
from ._utsnd import _ax_butterfly, _ax_im_array, _plt_im
from ._sensors import SENSORMAP_FRAME, SensorMapMixin, _plt_map2d


class Topomap(SensorMapMixin, ColorMapMixin, TopoMapKey, EelFigure):
    """Plot individual topogeraphies

    Parameters
    ----------
    y : (list of) NDVar, dims = ([case,] sensor,)
        Data to plot.
    xax : None | categorial
        Create a separate plot for each cell in this model.
    proj : str | list of str
        The sensor projection to use for topomaps (or one projection per plot).
    cmap : str
        Specify a custom color-map (default depends on the data).
    vmax, vmin : None | scalar
        Override the default plot limits. If only vmax is specified, vmin
        is set to -vmax.
    contours : sequence | dict
        Number of contours to draw.
    clip : bool | 'even' | 'circle'
        Outline for clipping topomaps: 'even' to clip at a constant distance
        (default), 'circle' to clip using a circle.
    clip_distance : scalar
        How far from sensor locations to clip (1 is the axes height/width).
    head_radius : scalar | tuple
        Radius of the head outline drawn over sensors (on sensor plots with
        normalized positions, 0.45 is the outline of the topomap); 0 to plot no
        outline; tuple for separate (right, anterior) radius.
        The default is determined automatically.
    head_pos : scalar
        Head outline position along the anterior axis (0 is the center, 0.5 is
        the top end of the plot).
    mark : Sensor index
        Sensors which to mark.
    sensorlabels : 'none' | 'index' | 'name' | 'fullname'
        Show sensor labels. For 'name', any prefix common to all names
        is removed; with 'fullname', the full name is shown.
    ds : None | Dataset
        If a Dataset is provided, ``epochs`` and ``xax`` can be specified
        as strings.
    sub : str | array
        Specify a subset of the data.
    res : int
        Resolution of the topomaps (width = height = ``res``).
    axtitle : bool | sequence of str
        Title for the individual axes. The default is to show the names of the
        epochs, but only if multiple axes are plotted.
    xlabel : str
        Label below the topomaps (default is no label).
    interpolation : 'nearest' | 'linear' | 'spline'
        Method for interpolating topo-map between sensors (default is based on
        mne-python).
    im_interpolation : str
        Topomap image interpolation (see Matplotlib's
        :meth:`~matplotlib.axes.Axes.imshow`). Matplotlib 1.5.3's SVG output
        can't handle uneven aspect with ``interpolation='none'``, use
        ``interpolation='nearest'`` instead.
    title : None | string
        Figure title.

    Notes
    -----
    Keys:
     - ``t``: open a ``Topomap`` plot for the region under the mouse pointer.
     - ``T``: open a larger ``Topomap`` plot with visible sensor names for the
       map under the mouse pointer.
    """
    _name = "Topomap"

    def __init__(self, y, xax=None, proj='default', cmap=None, vmax=None,
                 vmin=None, contours=7, clip='even', clip_distance=0.05,
                 head_radius=None, head_pos=0., mark=None, sensorlabels='none',
                 ds=None, sub=None, res=None, axtitle=True, xlabel=None,
                 interpolation=None, im_interpolation=None, *args, **kwargs):
        data = PlotData.from_args(y, ('sensor',), xax, ds, sub)
        self.plots = []
        ColorMapMixin.__init__(self, data.data, cmap, vmax, vmin, contours,
                               self.plots)
        if isinstance(proj, str):
            proj = repeat(proj, data.n_plots)
        elif not isinstance(proj, Sequence):
            raise TypeError("proj=%s" % repr(proj))
        elif len(proj) != data.n_plots:
            raise ValueError("need as many proj as axes (%s)" % data.n_plots)

        layout = ImLayout(data.plot_used, 1, 5, None, {}, *args, **kwargs)
        EelFigure.__init__(self, data.frame_title, layout)
        self._set_axtitle(axtitle, data)

        # plots
        for ax, layers, proj_ in zip(self._axes, data.data, proj):
            h = _ax_topomap(ax, layers, clip, clip_distance, sensorlabels, mark,
                            None, None, proj_, res, im_interpolation, xlabel,
                            self._vlims, self._cmaps, self._contours, interpolation,
                            head_radius, head_pos)
            self.plots.append(h)

        TopoMapKey.__init__(self, self._topo_data)
        SensorMapMixin.__init__(self, [h.sensors for h in self.plots])
        self._show()

    def _fill_toolbar(self, tb):
        ColorMapMixin._fill_toolbar(self, tb)
        SensorMapMixin._fill_toolbar(self, tb)

    def _topo_data(self, event):
        if event.inaxes:
            ax_i = self._axes.index(event.inaxes)
            p = self.plots[ax_i]
            return p.data, p.title, p.proj


class TopomapBins(EelFigure):
    _name = "TopomapBins"

    def __init__(self, y, xax=None, bin_length=0.05, tstart=None,
                 tstop=None, ds=None, sub=None, vmax=None, vmin=None, *args,
                 **kwargs):
        data = PlotData.from_args(y, ('sensor', 'time'), xax, ds, sub)
        data._cannot_skip_axes(self)
        ax_data = [[l.bin(bin_length, tstart, tstop) for l in layers]
                   for layers in data.data]

        # create figure
        time = ax_data[0][0].get_dim('time')
        n_bins = len(time)
        n_rows = len(ax_data)
        layout = Layout(n_bins * n_rows, 1, 1.5, False, *args, nrow=n_rows,
                        ncol=n_bins, **kwargs)
        EelFigure.__init__(self, data.frame_title, layout)

        cmaps = _base.find_fig_cmaps(ax_data)
        vlims = _base.find_fig_vlims(ax_data, vmax, vmin, cmaps)

        for row, layers in enumerate(ax_data):
            for column, t in enumerate(time):
                ax = self._axes[row * n_bins + column]
                topo_layers = [l.sub(time=t) for l in layers]
                _ax_topomap(ax, topo_layers, cmaps=cmaps, vlims=vlims)

        self._set_axtitle((str(t) for t in time), axes=self._axes[:len(time)])
        self._show()


class TopoButterfly(ColorMapMixin, TimeSlicerEF, TopoMapKey, YLimMixin,
                    XAxisMixin, EelFigure):
    """Butterfly plot with corresponding topomaps

    Parameters
    ----------
    y : (list of) NDVar
        Data to plot.
    xax : None | categorial
        Create a separate plot for each cell in this model.
    xlabel, ylabel : bool | string
        Labels for x and y axes. If True, labels are automatically chosen.
    xticklabels : bool | int | list of int
        Add tick-labels to the x-axis. Specify which axes should carry
        x-axis tick labels using ``int`` axes indices (default ``-1``).
    color : matplotlib color
        Color of the butterfly plots.
    linewidth : scalar
        Linewidth for plots (defult is to use ``matplotlib.rcParams``).
    proj : str
        The sensor projection to use for topomaps.
    res : int
        Resolution of the topomaps (width = height = ``res``).
    interpolation : 'nearest' | 'linear' | 'spline'
        Method for interpolating topo-map between sensors (default is based on
        mne-python).
    clip : bool | 'even' | 'circle'
        Outline for clipping topomaps: 'even' to clip at a constant distance
        (default), 'circle' to clip using a circle.
    clip_distance : scalar
        How far from sensor locations to clip (1 is the axes height/width).
    sensorlabels : None | 'index' | 'name' | 'fullname'
        Show sensor labels. For 'name', any prefix common to all names
        is removed; with 'fullname', the full name is shown.
    mark : None | list of sensor names or indices
        Highlight a subset of the sensors.
    mcolor : matplotlib color
        Color for marked sensors.
    ds : None | Dataset
        If a Dataset is provided, ``epochs`` and ``xax`` can be specified
    sub : str | array
        Specify a subset of the data.
        as strings.
    axh : scalar
        Height of the butterfly axes as well as side length of the topomap
        axes (in inches).
    ax_aspect : scalar
        multiplier for the width of butterfly plots based on their height
    vmax : scalar
        Upper limits for the colormap.
    vmin : scalar
        Lower limit for the colormap.
    cmap : str
        Colormap (default depends on the data).
    axtitle : bool | sequence of str
        Title for the individual axes. The default is to show the names of the
        epochs, but only if multiple axes are plotted.
    frame : 't'
        Use T-frame for the Butterfly plots (default is rectangular frame).
    xlim : scalar | (scalar, scalar)
        Initial x-axis view limits as ``(left, right)`` tuple or as ``length``
        scalar (default is the full x-axis in the data).
    im_interpolation : str
        Topomap image interpolation (see Matplotlib's
        :meth:`~matplotlib.axes.Axes.imshow`). Matplotlib 1.5.3's SVG output
        can't handle uneven aspect with ``interpolation='none'``, use
        ``interpolation='nearest'`` instead.
    title : None | string
        Figure title.

    Notes
    -----
    Topomap control:
     - LMB click in a butterfly plot fixates the topomap time
     - RMB click in a butterfly plot removes the time point, the topomaps
       follow the mouse pointer
     - ``.``: Increment the current topomap time (got right)
     - ``,``: Decrement the current topomap time (go left)
     - ``t``: open a ``Topomap`` plot for the time point under the mouse
       pointer
     - ``T``: open a larger ``Topomap`` plot with visible sensor names for the
       time point under the mouse pointer

    Navigation:
     - ``↑``: scroll up
     - ``↓``: scroll down
     - ``←``: scroll left
     - ``→``: scroll right
     - ``home``: scroll to beginning
     - ``end``: scroll to end
     - ``f``: x-axis zoom in (reduce x axis range)
     - ``d``: x-axis zoom out (increase x axis range)
     - ``r``: y-axis zoom in (reduce y-axis range)
     - ``c``: y-axis zoom out (increase y-axis range)
    """
    _default_xlabel_ax = -2
    _name = "TopoButterfly"

    def __init__(self, y, xax=None, xlabel=True, ylabel=True, xticklabels=-1,
                 color=None, linewidth=None,
                 proj='default', res=None, interpolation=None, clip='even',
                 clip_distance=0.05,
                 sensorlabels=None, mark=None, mcolor=None, ds=None, sub=None,
                 vmax=None, vmin=None, cmap=None, axlabel=None, axtitle=True,
                 frame=True, xlim=None, im_interpolation=None, *args, **kwargs):
        if axlabel is not None:
            warn("The axlabel parameter for plot.TopoButterfly() is "
                 "deprecated, please use axtitle instead", DeprecationWarning)
            axtitle = axlabel
        data = PlotData.from_args(y, ('sensor', None), xax, ds, sub)
        data._cannot_skip_axes(self)
        xdim = data.dims[1]
        self._epochs = data.data

        # create figure
        layout = VariableAspectLayout(
            data.n_plots, 3, 10, (None, 1), None, (frame, False),
            self._set_axtitle(axtitle, data, data.n_plots), *args, **kwargs
        )
        EelFigure.__init__(self, data.frame_title, layout)

        self.bfly_axes = self._axes[0::2]
        self.topo_axes = self._axes[1::2]
        self.bfly_plots = []
        self.topo_plots = []
        self.t_markers = []  # vertical lines on butterfly plots

        ColorMapMixin.__init__(self, data.data, cmap, vmax, vmin, None,
                               self.topo_plots)

        self._topo_kwargs = {
            'clip': clip,
            'clip_distance': clip_distance,
            'proj': proj,
            'contours': self._contours,
            'res': res,
            'interpolation': interpolation,
            'im_interpolation': im_interpolation,
            'sensorlabels': sensorlabels,
            'mark': mark,
            'mcolor': mcolor,
        }

        # plot epochs (x/y are in figure coordinates)
        for ax, layers in zip(self.bfly_axes, data.plot_data):
            p = _ax_butterfly(ax, layers, 'time', 'sensor', mark, color,
                              linewidth, self._vlims)
            self.bfly_plots.append(p)

        # decorate axes
        e0 = data.data[0][0]
        self._configure_xaxis_dim(e0.time, xlabel, xticklabels, self.bfly_axes)
        self._configure_yaxis(e0, ylabel, self.bfly_axes)

        # setup callback
        XAxisMixin._init_with_data(self, data.data, xdim, xlim, self.bfly_axes)
        YLimMixin.__init__(self, self.bfly_plots + self.topo_plots)
        TimeSlicerEF.__init__(self, xdim, data.data, self.bfly_axes, False)
        TopoMapKey.__init__(self, self._topo_data)
        self._realtime_topo = True
        self._t_label = None  # time label under lowest topo-map
        self.canvas.store_canvas()
        self._update_topo(e0.time[0])

        self._show(crosshair_axes=self.bfly_axes)
        self._init_controller()

    def _fill_toolbar(self, tb):
        ColorMapMixin._fill_toolbar(self, tb)

    def _update_topo(self, t):
        epochs = [[l.sub(time=t) for l in layers if t in l.time]
                  for layers in self._epochs]

        if not self.topo_plots:
            for ax, layers in zip(self.topo_axes, epochs):
                p = _ax_topomap(ax, layers, cmaps=self._cmaps,
                                vlims=self._vlims, **self._topo_kwargs)
                self.topo_plots.append(p)
        else:
            for layers, p in zip(epochs, self.topo_plots):
                p.set_data(layers)

    def set_topo_t(self, t):
        "Set the time point of the topo-maps"
        self._set_time(t, True)

    def _topo_data(self, event):
        ax = event.inaxes
        if ax in self.bfly_axes:
            p = self.bfly_plots[ax.id // 2]
            t = event.xdata
            seg = [l.sub(time=t) for l in p.data]
        elif ax in self.topo_axes:
            seg = self.topo_plots[ax.id // 2].data
            t = self._current_time
        else:
            return

        return seg, "%i ms" % round(t * 1e3), self._topo_kwargs['proj']

    def _on_leave_axes_status_text(self, event):
        return "Topomap: t = %.3f" % self._current_time

    def _update_time(self, t, fixate):
        TimeSlicerEF._update_time(self, t, fixate)
        self._update_topo(t)
        if fixate:
            # add time label
            text = "t = %i ms" % round(t * 1e3)
            if self._t_label:
                self._t_label.set_text(text)
            else:
                ax = self.topo_axes[-1]
                self._t_label = ax.text(.5, -0.1, text, ha='center', va='top')
            self.canvas.draw()  # otherwise time label does not get redrawn
        elif self._time_fixed:
            self._t_label.remove()
            self._t_label = None
            self.canvas.draw()  # otherwise time label does not get redrawn
        else:
            self.canvas.redraw(self.topo_axes)


class _plt_topomap(_plt_im):
    """Topomap plot

    Parameters
    ----------
    ...
    im_frame : scalar
        Empty space beyond outmost sensors in the im plot.
    vmax : scalar
        Override the colorspace vmax.
    interpolation : 'nearest' | 'linear' | 'spline'
        Method for interpolating topo-map between sensors.
    """
    _aspect = 'equal'

    def __init__(
            self,
            ax: mpl.axes.Axes,
            ndvar: NDVar,
            overlay,
            proj,
            res,
            im_interpolation,
            vlims,
            cmaps,
            contours,
            interpolation: str,
            clip: str,
            clip_distance: float,
    ):
        # store attributes
        self._proj = proj
        self._visible_data = ndvar.sensor._visible_sensors(proj)
        self._grid = np.linspace(0, 1, res)
        self._mgrid = tuple(np.meshgrid(self._grid, self._grid))
        self._method = interpolation

        # clip mask
        if interpolation is None and clip:
            locs = ndvar.sensor.get_locs_2d(self._proj, frame=SENSORMAP_FRAME)
            if clip == 'even':
                hull = ConvexHull(locs)
                points = locs[hull.vertices]
                default_head_radius = sqrt(np.min(np.sum((points - [0.5, 0.5]) ** 2, 1)))
                # find offset due to clip_distance
                tangents = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
                verticals = np.dot(tangents, [[0, -1], [1, 0]])
                verticals /= np.sqrt(np.sum(verticals ** 2, 1)[:, None])
                verticals *= clip_distance
                # apply offset
                points += verticals
                mask = mpl.patches.Polygon(points, transform=ax.transData)
            elif clip == 'circle':
                clip_radius = sqrt(np.max(np.sum((locs - [0.5, 0.5]) ** 2, 1)))
                mask = mpl.patches.Circle((0.5, 0.5), clip_radius, transform=ax.transData)
                default_head_radius = clip_radius
            else:
                raise ValueError('clip=%r' % (clip,))
        else:
            mask = None
            default_head_radius = None

        self._default_head_radius = default_head_radius
        _plt_im.__init__(self, ax, ndvar, overlay, cmaps, vlims, contours,
                         (0, 1, 0, 1), im_interpolation, mask)

    def _data_from_ndvar(self, ndvar):
        v = ndvar.get_data(('sensor',))
        locs = ndvar.sensor.get_locs_2d(self._proj, frame=SENSORMAP_FRAME)
        if self._visible_data is not None:
            v = v[self._visible_data]
            locs = locs[self._visible_data]

        # axis parameter in numpy >= 0.13
        # unique_locs = np.unique(locs, axis=0)

        if self._method is None:
            # interpolate data
            xi, yi = self._mgrid

            # code adapted from mne-python topmap _griddata()
            xy = locs[:, 0] + locs[:, 1] * -1j
            d = np.abs(xy - xy[:, None])
            diagonal_step = len(locs) + 1
            d.flat[::diagonal_step] = 1.

            g = (d * d) * (np.log(d) - 1.)
            g.flat[::diagonal_step] = 0.
            try:
                weights = linalg.solve(g, v.ravel())
            except ValueError:
                raise NotImplementedError(
                    "Error determining sensor map projection, possibly due to "
                    "more than one sensor in a single location; try using a "
                    "different projection.")

            m, n = xi.shape
            out = np.empty_like(xi)

            g = np.empty(xy.shape)
            for i in range(m):
                for j in range(n):
                    d = np.abs(xi[i, j] + -1j * yi[i, j] - xy)
                    mask = np.where(d == 0)[0]
                    if len(mask):
                        d[mask] = 1.
                    np.log(d, out=g)
                    g -= 1.
                    g *= d * d
                    if len(mask):
                        g[mask] = 0.
                    out[i, j] = g.dot(weights)
            return out
        elif self._method == 'spline':
            k = int(floor(sqrt(len(locs)))) - 1
            tck = interpolate.bisplrep(locs[:, 1], locs[:, 0], v, kx=k, ky=k)
            return interpolate.bisplev(self._grid, self._grid, tck)
        else:
            isnan = np.isnan(v)
            if np.any(isnan):
                nanmap = interpolate.griddata(locs, isnan, self._mgrid, self._method)
                mask = nanmap > 0.5
                v = np.where(isnan, 0, v)
                vmap = interpolate.griddata(locs, v, self._mgrid, self._method)
                np.place(vmap, mask, np.NaN)
                return vmap
            return interpolate.griddata(locs, v, self._mgrid, self._method)


class _ax_topomap(_ax_im_array):
    """Axes with a topomap

    Parameters
    ----------
    mark : list of IDs
        highlight a subset of the sensors
    """
    def __init__(
            self,
            ax,
            layers,
            clip: str = 'even',  # even or circle (only applies if interpolation is None)
            clip_distance: float=0.05,  # distance from outermost sensor for clip=='even'
            sensorlabels: str = None,  # index | name | fullname
            mark=None,
            mcolor=None,
            mmarker=None,
            proj: str = 'default',  # topomap projection method
            res: int = None,  # topomap image resolution
            im_interpolation: str = None,  # matplotlib imshow interpolation method
            xlabel: Union[bool, str] = None,
            vlims: dict = {},
            cmaps: dict = {},
            contours: dict = {},
            interpolation: str = None,  # topomap interpolation method
            head_radius: float = None,
            head_pos: Union[float, Tuple[float, float]] = 0.,  # y if centered on x, or (x, y)
            head_linewidth: float = None):
        self.ax = ax
        self.data = layers
        self.proj = proj
        self.layers = []

        if xlabel is True:
            xlabel = layers[0].name
        if im_interpolation is None:
            im_interpolation = 'bilinear'
        if res is None:
            res = 64 if interpolation is None else 100

        ax.set_axis_off()
        overlay = False
        for layer in layers:
            h = _plt_topomap(ax, layer, overlay, proj, res, im_interpolation,
                             vlims, cmaps, contours, interpolation, clip, clip_distance)
            self.layers.append(h)
            overlay = True

        # head outline
        if head_radius is None and clip == 'circle' and interpolation is None and layer.sensor._topomap_outlines(proj) == 'top':
            head_radius = self.layers[0]._default_head_radius

        # plot sensors
        sensor_dim = layers[0].sensor
        self.sensors = _plt_map2d(ax, sensor_dim, proj, 1, '.', 1, 'k', mark,
                                  mcolor, mmarker, sensorlabels, False,
                                  head_radius, head_pos, head_linewidth)

        ax.set_aspect('equal')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        if isinstance(xlabel, str):
            x, y = ax.transData.inverted().transform(ax.transAxes.transform((0.5, 0)))
            ax.text(x, y, xlabel, ha='center', va='top')

    def set_ylim(self, bottom, top):  # Alias for YLimMixin
        self.set_vlim(bottom, top)


class _TopoWindow:
    """Helper class for TopoArray.

    Maintains a topomap corresponding to one segment with flexible time point.
    """
    def __init__(self, ax, parent, **plot_args):
        self.ax = ax
        self.parent = parent
        self.plot_args = plot_args
        # initial plot state
        self.t_line = None
        self.pointer = None
        self.plot = None

    def update(self, t):
        if t is not None:
            if self.t_line:
                self.t_line.remove()
            self.t_line = self.parent.ax.axvline(t, c='r')

            t_str = "%i ms" % round(t * 1e3)
            if self.pointer:
                self.pointer.axes = self.parent.ax
                self.pointer.xy = (t, 1)
                self.pointer.set_text(t_str)
                self.pointer.set_visible(True)
            else:
                xytext = self.ax.transAxes.transform((.5, 1))
                # These coordinates are in 'figure pixels'. They do not scale
                # when the figure is rescaled, so we need to transform them
                # into 'figure fraction' coordinates
                inv = self.ax.figure.transFigure.inverted()
                xytext = inv.transform(xytext)
                self.pointer = self.parent.ax.annotate(
                    t_str, (t, 0),
                    xycoords='data',
                    xytext=xytext,
                    textcoords='figure fraction',
                    horizontalalignment='center',
                    verticalalignment='center',
                    arrowprops={'arrowstyle': '-',
                                'shrinkB': 0,
                                'connectionstyle': "angle3,angleA=90,angleB=0",
                                'color': 'r'},
                    zorder=99)

            layers = [l.sub(time=t) for l in self.parent.data if t in l.time]
            if self.plot is None:
                self.plot = _ax_topomap(self.ax, layers, **self.plot_args)
            else:
                self.plot.set_data(layers)

    def clear(self):
        self.ax.cla()
        self.ax.set_axis_off()
        self.plot = None
        if self.t_line:
            self.t_line.remove()
            self.t_line = None
        if self.pointer:
            self.pointer.remove()
            self.pointer = None

    def add_contour(self, meas, level, color):
        if self.plot:
            self.plot.add_contour(meas, level, color)

    def set_cmap(self, cmap, meas):
        if self.plot:
            self.plot.set_cmap(cmap, meas)

    def set_vlim(self, v, vmax=None, meas=None):
        if self.plot:
            self.plot.set_vlim(v, vmax, meas)


class TopoArray(ColorMapMixin, EelFigure):
    """Channel by sample plots with topomaps for individual time points

    Parameters
    ----------
    y : (list of) NDVar
        Data to plot.
    xax : None | categorial
        Create a separate plot for each cell in this model.
    title : None | string
        Figure title.
    ntopo | int
        number of topomaps per array-plot.
    t : list of scalar (len <= ntopo)
        Time points for topomaps.
    ds : None | Dataset
        If a Dataset is provided, ``epochs`` and ``xax`` can be specified
        as strings.
    sub : str | array
        Specify a subset of the data.
    vmax : scalar
        Upper limits for the colormap.
    vmin : scalar
        Lower limit for the colormap.
    cmap : str
        Colormap (default depends on the data).
    interpolation : 'nearest' | 'linear' | 'spline'
        Method for interpolating topo-map between sensors (default is based on
        mne-python).
    im_interpolation : str
        Topomap image interpolation (see Matplotlib's
        :meth:`~matplotlib.axes.Axes.imshow`). Matplotlib 1.5.3's SVG output
        can't handle uneven aspect with ``interpolation='none'``, use
        ``interpolation='nearest'`` instead.
    xticklabels : bool | int | list of int
        Add tick-labels to the x-axis. Specify which axes should carry
        x-axis tick labels using ``int`` axes indices (default ``-1``).
    axtitle : bool | sequence of str
        Title for the individual axes. The default is to show the names of the
        epochs, but only if multiple axes are plotted.

    Notes
    -----
     - LMB click on a topomap selects it for tracking the mouse pointer
         - LMB on the array plot fixates the topomap time point
     - RMB on a topomap removes the topomap

    """
    _make_axes = False
    _name = 'TopoArray'

    def __init__(self, y, xax=None, title=None, ntopo=3, t=(), ds=None,
                 sub=None, vmax=None, vmin=None, cmap=None, interpolation=None,
                 im_interpolation=None, xticklabels=-1, axtitle=True, *args,
                 **kwargs):
        data = PlotData.from_args(y, ('time', 'sensor'), xax, ds, sub)
        n_topo_total = ntopo * data.n_plots

        # create figure
        layout = Layout(data.plot_used, 1.5, 6, False, title, *args, **kwargs)
        EelFigure.__init__(self, data.frame_title, layout)
        all_plots = []
        ColorMapMixin.__init__(self, data.data, cmap, vmax, vmin, None, all_plots)

        # fig coordinates
        x_frame_l = .6 / self._layout.axw / data.n_plots
        x_frame_r = .025 / data.n_plots
        x_sep = .01 / data.n_plots
        x_per_ax = (1 - x_frame_l - x_frame_r) / data.n_plots

        self.figure.subplots_adjust(left=x_frame_l, right=1 - x_frame_r,
                                    bottom=.05, top=.9, wspace=.1, hspace=.3)
        self.title = title

        # save important properties
        self._epochs = data.data
        self._ntopo = ntopo
        self._default_xlabel_ax = -1 - ntopo

        # im_array plots
        self._array_axes = []
        self._array_plots = []
        self._topo_windows = []
        ax_height = .4 + .07 * (not title)
        ax_bottom = .45  # + .05*(not title)
        for i, layers in enumerate(data.data):
            ax_left = x_frame_l + i * (x_per_ax + x_sep)
            ax_right = 1 - x_frame_r - (data.n_plots - i - 1) * (x_per_ax + x_sep)
            ax_width = ax_right - ax_left
            ax = self.figure.add_axes((ax_left, ax_bottom, ax_width, ax_height),
                                      picker=True)
            ax.ID = i
            ax.type = 'main'
            im_plot = _ax_im_array(ax, layers, 'time', im_interpolation,
                                   self._vlims, self._cmaps, self._contours)
            self._axes.append(ax)
            self._array_axes.append(ax)
            self._array_plots.append(im_plot)
            if i > 0:
                ax.yaxis.set_visible(False)

            # topo plots
            for j in range(ntopo):
                ID = i * ntopo + j
                ax = self.figure.add_subplot(3, n_topo_total,
                                             2 * n_topo_total + 1 + ID,
                                             picker=True, xticks=[], yticks=[])
                ax.ID = ID
                ax.type = 'window'
                win = _TopoWindow(ax, im_plot, vlims=self._vlims,
                                  cmaps=self._cmaps, contours=self._contours,
                                  interpolation=interpolation,
                                  im_interpolation=im_interpolation)
                self._axes.append(ax)
                self._topo_windows.append(win)
        all_plots.extend(self._array_plots)
        all_plots.extend(self._topo_windows)

        # if t argument is provided, set topo-map time points
        if t:
            if np.isscalar(t):
                t = [t]
            self.set_topo_ts(*t)

        self._set_axtitle(axtitle, data, self._array_axes)
        self._configure_xaxis_dim(data.y0.time, True, xticklabels, self._array_axes)
        self._configure_yaxis_dim(data.data, 'sensor', True, self._array_axes, False)

        # setup callback
        self._selected_window = None
        self.canvas.mpl_connect('pick_event', self._pick_handler)
        self.canvas.store_canvas()
        self._show(crosshair_axes=self._array_axes)

    def _fill_toolbar(self, tb):
        ColorMapMixin._fill_toolbar(self, tb)

    def __repr__(self):
        e_repr = []
        for e in self._epochs:
            if hasattr(e, 'name'):
                e_repr.append(e.name)
            else:
                e_repr.append([ie.name for ie in e])
        kwargs = {'s': repr(e_repr),
                  't': ' %r' % self.title if self.title else ''}
        txt = "<plot.TopoArray{t} ({s})>".format(**kwargs)
        return txt

    def add_contour(self, meas, level, color='k'):
        """Add a contour line

        Parameters
        ----------
        meas : str
            The measurement for which to add a contour line.
        level : scalar
            The value at which to draw the contour.
        color : matplotlib color
            The color of the contour line.
        """
        for p in self._iter_plots():
            p.add_contour(meas, level, color)
        self.draw()

    def _iter_plots(self):
        "Iterate through non-empty plots"
        for p in self._array_plots:
            yield p
        for w in self._topo_windows:
            if w.plot is not None:
                yield w.plot

    def set_cmap(self, cmap, meas=None):
        """Change the colormap

        Parameters
        ----------
        cmap : str | colormap
            New colormap.
        meas : None | str
            Measurement to which to apply the colormap. With None, it is
            applied to all.
        """
        self._cmaps[meas] = cmap
        for p in self._iter_plots():
            p.set_cmap(cmap, meas)
        self.draw()

    def set_topo_t_single(self, topo_id, t, parent_im_id='auto'):
        """
        Set the time for a single topomap.

        Parameters
        ----------
        topo_id : int
            Index of the topomap (numbered throughout the figure).
        t : scalar or ``None``
            time point; ``None`` clears the topomap
        parent_im_id : 'auto' | int
            Index of the array plot from which to draw the topo plot. For
            'auto', the array plot above the topomap is used.
        """
        # get parent ax
        if parent_im_id == 'auto':
            parent_im_id = int(topo_id / self._ntopo)
        # get window ax
        w = self._topo_windows[topo_id]
        w.clear()

        if t is not None:
            w.update(t)

        self.canvas.draw()

    def set_topo_t(self, topo_id, t):
        """
        Set the time point for a topo-map (same for all array plots)

        Parameters
        ----------
        topo_id : int
            Index of the topomap (numberd for each array-plot).
        t : scalar or ``None``
            time point; ``None`` clears the topomap

        See Also
        --------
        .set_topo_ts : set several topomap time points at once
        .set_topo_t_single : set the time point of a single topomap
        """
        for i in range(len(self._array_plots)):
            _topo = self._ntopo * i + topo_id
            self.set_topo_t_single(_topo, t, parent_im_id=i)

    def set_topo_ts(self, *t_list):
        """Set the time points displayed in topo-maps across all array-plots"""
        for i, t in enumerate(t_list):
            self.set_topo_t(i, t)

    def _pick_handler(self, pickevent):
        mouseevent = pickevent.mouseevent
        ax = pickevent.artist
        if ax.type == 'window':
            button = mouseevent.button  # 1: Left
            window = self._topo_windows[ax.ID]
            if button == 1:
                self._selected_window = window
            elif button in (2, 3):
                Id = window.ax.ID % self._ntopo
                self.set_topo_t(Id, None)
            else:
                pass
        elif (ax.type == 'main') and (self._selected_window is not None):
            self._selected_window.clear()  # to side track pdf export transparency issue
            # update corresponding topo_windows
            t = mouseevent.xdata
            Id = self._selected_window.ax.ID % self._ntopo
            self.set_topo_t(Id, t)

            self._selected_window = None
            self.canvas.draw()

    def _on_motion_sub(self, event):
        if (self._selected_window is not None and event.inaxes and
                event.inaxes.type == 'main' and
                event.xdata in self._epochs[0][0].time):
            self._selected_window.update(event.xdata)
            return {self._selected_window.ax}
        return set()
