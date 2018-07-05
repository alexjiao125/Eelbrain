"""GUI for selecting topographical components"""
# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>

# Document:  represents data
# ChangeAction:  modifies Document
# Model:  creates ChangeActions and applies them to the History
# Frame:
#  - visualizaes Document
#  - listens to Document changes
#  - issues commands to Model
from distutils.version import LooseVersion
from itertools import repeat
from math import ceil
import re

import mne
from matplotlib.patches import Rectangle
import numpy as np
from scipy import linalg
import wx
from wx.lib.scrolledpanel import ScrolledPanel

from .. import load, plot, fmtxt
from .._data_obj import Factor, NDVar, asndvar, Categorial, Scalar
from .._wxutils import Icon, ID, REValidator
from .._utils.parse import POS_FLOAT_PATTERN
from ..plot._topo import _ax_topomap
from .frame import EelbrainDialog
from .history import Action, FileDocument, FileModel, FileFrame, FileFrameChild
from .mpl_canvas import FigureCanvasPanel
from .text import HTMLFrame


COLOR = {True: (.5, 1, .5), False: (1, .3, .3)}
LINE_COLOR = {True: 'k', False: (1, 0, 0)}
LINK = 'component:%i epoch:%s'
TOPO_ARGS = {
    'interpolation': 'linear',  # interpolation that does not assume continuity
    'clip': 'even',
}

# For unit-tests
TEST_MODE = False


class ChangeAction(Action):
    """Action objects are kept in the history and can do and undo themselves

    Parameters
    ----------
    desc : str
        Description of the action
        list of (i, name, old, new) tuples
    """
    def __init__(self, desc, index=None, old_accept=None, new_accept=None,
                 old_path=None, new_path=None):
        self.desc = desc
        self.index = index
        self.old_path = old_path
        self.old_accept = old_accept
        self.new_path = new_path
        self.new_accept = new_accept

    def do(self, doc):
        if self.index is not None:
            doc.set_case(self.index, self.new_accept)
        if self.new_path is not None:
            doc.set_path(self.new_path)

    def undo(self, doc):
        if self.index is not None:
            doc.set_case(self.index, self.old_accept)
        if self.new_path is not None and self.old_path is not None:
            doc.set_path(self.old_path)


class Document(FileDocument):
    """Represents data for the current state of the Document

    (Data can be accessed, but should only be modified through the Model)

    Parameters
    ----------
    path : str
        Path to the ICA file.
    ds : Dataset
        Dataset containing 'epochs' (mne Epochs), 'index' (Var describing
        epochs) and variables describing cases in epochs, used to plot
        condition averages.
    """
    def __init__(self, path, ds, sysname):
        FileDocument.__init__(self, path)
        self.saved = True

        self.ica = ica = mne.preprocessing.read_ica(path)
        if LooseVersion(mne.__version__) < LooseVersion('0.16'):
            ica.pre_whitener_ = ica._pre_whitener

        self.accept = np.ones(self.ica.n_components_, bool)
        self.accept[ica.exclude] = False
        self.epochs = epochs = ds['epochs']
        self.epochs_ndvar = load.fiff.epochs_ndvar(epochs, sysname=sysname)
        self.ds = ds

        data = np.dot(ica.mixing_matrix_.T, ica.pca_components_[:ica.n_components_])
        ic_dim = Scalar('component', np.arange(len(data)))
        self.components = NDVar(data, (ic_dim, self.epochs_ndvar.sensor),
                                info={'meas': 'component', 'cmap': 'xpolar'})

        # sources
        data = ica.get_sources(epochs).get_data()
        self.sources = NDVar(data, ('case', ic_dim, self.epochs_ndvar.time),
                             {'meas': 'component', 'cmap': 'xpolar'}, 'sources')

        # find unique epoch labels
        if 'index' in ds:
            labels = map(str, ds['index'])
            if 'epoch' in ds:
                labels = map(' '.join, zip(ds['epoch'], labels))
        else:
            labels = map(str, range(len(epochs)))
        self.epoch_labels = tuple(labels)

        # global mean (which is not modified by ICA)
        if ica.noise_cov is None:  # revert standardization
            global_mean = ica.pca_mean_ * ica.pre_whitener_[:, 0]
        else:
            global_mean = np.dot(linalg.pinv(ica.pre_whitener_), ica.pca_mean_)
        self.global_mean = NDVar(global_mean, (self.epochs_ndvar.sensor,))

        # publisher
        self.callbacks.register_key('case_change')

    def apply(self, inst):
        if isinstance(inst, list):
            return [self.ica.apply(i.copy()) for i in inst]
        else:
            return self.ica.apply(inst.copy())

    def set_case(self, index, state):
        self.accept[index] = state
        self.ica.exclude = list(np.flatnonzero(np.invert(self.accept)))
        self.callbacks.callback('case_change', index)

    def save(self):
        self.ica.save(self.path)


class Model(FileModel):
    """Manages a document with its history"""
    def set_case(self, index, state, desc="Manual Change"):
        old_accept = self.doc.accept[index]
        action = ChangeAction(desc, index, old_accept, state)
        self.history.do(action)

    def toggle(self, case):
        old_accept = self.doc.accept[case]
        action = ChangeAction("Manual toggle", case, old_accept, not old_accept)
        self.history.do(action)

    def clear(self):
        action = ChangeAction("Clear", slice(None), self.doc.accept.copy(),
                              True)
        self.history.do(action)


class ContextMenu(wx.Menu):
    "Helper class for Menu to store component ID"
    def __init__(self, i):
        wx.Menu.__init__(self)
        self.i = i


class Frame(FileFrame):
    """GIU for selecting ICA sensor-space components

    Component Selection
    ===================

    * Click on components topographies to select/deselect them.
    * Use the context-menu (right click) for additional commands.

    *Keyboard shortcuts* in addition to the ones in the menu:

    =========== ============================================================
    Key         Effect
    =========== ============================================================
    t           topomap plot of the component under the pointer
    a           array-plot of the source time course of the component
    s           plot sources, starting with the component under the cursor
    f           plot the frequency spectrum for the component under the
                pointer
    b           butterfly plot of grand average (original and cleaned)
    B           butterfly plot of condition averages
    =========== ============================================================
    """
    _doc_name = 'component selection'
    _name = 'SelectComponents'
    _title = 'Select Components'
    _wildcard = "ICA fiff file (*-ica.fif)|*.fif"

    def __init__(self, parent, pos, size, model):
        self.last_model = ""
        self.source_frame = None
        self.butterfly_baseline = ID.BASELINE_NONE

        super(Frame, self).__init__(parent, pos, size, model)

        # setup layout
        self.ax_size = 200
        figsize = (10, 10)
        self.SetMinSize((400, 400))

        # setup scrolled panel
        panel = ScrolledPanel(self)
        self.panel = panel

        # setup figure canvas
        self.canvas = FigureCanvasPanel(panel, figsize=figsize)
        self.canvas.figure.subplots_adjust(0, 0, 1, 1, 0, 0)
        panel.SetupScrolling(False, scrollToTop=False, scrollIntoView=False)

        # sizer
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.canvas, 0)
        panel.SetSizer(sizer)
        self.canvas_sizer = sizer

        # Toolbar
        tb = self.InitToolbar(can_open=False)
        tb.AddSeparator()
        # buttons
        button = wx.Button(tb, ID.SHOW_SOURCES, "Sources")
        button.Bind(wx.EVT_BUTTON, self.OnShowSources)
        tb.AddControl(button)
        button = wx.Button(tb, ID.FIND_RARE_EVENTS, "Rare Events")
        button.Bind(wx.EVT_BUTTON, self.OnFindRareEvents)
        tb.AddControl(button)
        # tail
        tb.AddStretchableSpace()
        self.InitToolbarTail(tb)
        tb.Realize()

        self.CreateStatusBar()

        # Bind Events ---
        self.doc.callbacks.subscribe('case_change', self.CaseChanged)
        self.panel.Bind(wx.EVT_SIZE, self.OnPanelResize)
        self.canvas.mpl_connect('axes_enter_event', self.OnPointerEntersAxes)
        self.canvas.mpl_connect('axes_leave_event', self.OnPointerEntersAxes)
        self.canvas.mpl_connect('button_press_event', self.OnCanvasClick)
        self.canvas.mpl_connect('key_release_event', self.OnCanvasKey)
        # re-Bind right click
        self.canvas.Unbind(wx.EVT_RIGHT_DOWN)
        self.canvas.Bind(wx.EVT_RIGHT_DOWN, self.OnRightDown)

        # Finalize
        self.plot()
        self.UpdateTitle()

    def plot(self):
        n = self.doc.ica.n_components_
        fig = self.canvas.figure
        fig.clf()

        panel_w = self.panel.GetSize()[0]
        n_h = max(2, panel_w // self.ax_size)
        n_v = int(ceil(n / n_h))

        # adjust canvas size
        size = (self.ax_size * n_h, self.ax_size * n_v)
        self.canvas_sizer.SetItemMinSize(self.canvas, size)

        # plot
        axes = tuple(fig.add_subplot(n_v, n_h, i) for i in range(1, n + 1))
        # bgs = tuple(ax.patch)
        for i, ax, c, accept in zip(range(n), axes, self.doc.components, self.doc.accept):
            _ax_topomap(ax, [c], **TOPO_ARGS)
            ax.text(0.5, 1, "# %i" % i, ha='center', va='top')
            p = Rectangle((0, 0), 1, 1, color=COLOR[accept], zorder=-1)
            ax.add_patch(p)
            ax.i = i
            ax.background = p

        self.axes = axes
        self.n_h = n_h
        self.canvas.store_canvas()
        self.Layout()

    def CaseChanged(self, index):
        "Update the state of the segments on the current page"
        if isinstance(index, int):
            index = [index]
        elif isinstance(index, slice):
            start = index.start or 0
            stop = index.stop or len(self.doc.components)
            index = range(start, stop)
        elif index.dtype.kind == 'b':
            index = np.nonzero(index)[0]

        # update epoch plots
        axes = []
        for idx in index:
            ax = self.axes[idx]
            ax.background.set_color(COLOR[self.doc.accept[idx]])
            axes.append(ax)

        self.canvas.redraw(axes=axes)

    def GoToComponentEpoch(self, component, epoch):
        if not self.source_frame:
            self.ShowSources(0)
        self.source_frame.GoToComponentEpoch(component, epoch)

    def MakeToolsMenu(self, menu):
        app = wx.GetApp()
        item = menu.Append(wx.ID_ANY, "Source Viewer",
                           "Open a source time course viewer window")
        app.Bind(wx.EVT_MENU, self.OnShowSources, item)
        item = menu.Append(wx.ID_ANY, "Find Rare Events",
                           "Find components with major loading on a small "
                           "number of epochs")
        app.Bind(wx.EVT_MENU, self.OnFindRareEvents, item)
        menu.AppendSeparator()

        # plotting
        item = menu.Append(wx.ID_ANY, "Butterfly Plot Grand Average",
                           "Plot the grand average of all epochs")
        app.Bind(wx.EVT_MENU, self.OnPlotGrandAverage, item)
        item = menu.Append(wx.ID_ANY, "Butterfly Plot by Category",
                           "Separate butterfly plots for different model cells")
        app.Bind(wx.EVT_MENU, self.OnPlotButterfly, item)
        # Baseline submenu
        blmenu = wx.Menu()
        blmenu.AppendRadioItem(ID.BASELINE_CUSTOM, "Baseline Period")
        blmenu.AppendRadioItem(ID.BASELINE_GLOABL_MEAN, "Global Mean")
        blmenu.AppendRadioItem(ID.BASELINE_NONE, "No Baseline Correction")
        blmenu.Check(self.butterfly_baseline, True)
        blmenu.Bind(wx.EVT_MENU, self.OnSetButterflyBaseline, id=ID.BASELINE_CUSTOM)
        blmenu.Bind(wx.EVT_MENU, self.OnSetButterflyBaseline, id=ID.BASELINE_GLOABL_MEAN)
        blmenu.Bind(wx.EVT_MENU, self.OnSetButterflyBaseline, id=ID.BASELINE_NONE)
        menu.AppendSubMenu(blmenu, "Baseline")

    def OnCanvasClick(self, event):
        "Called by mouse clicks"
        if event.button == 1:
            if event.inaxes:
                self.model.toggle(event.inaxes.i)

    def OnCanvasKey(self, event):
        if not event.inaxes:
            return
        if event.key == 't':
            self.PlotCompTopomap(event.inaxes.i)
        elif event.key == 'a':
            self.PlotCompSourceArray(event.inaxes.i)
        elif event.key == 's':
            self.ShowSources(event.inaxes.i)
        elif event.key == 'f':
            self.PlotCompFFT(event.inaxes.i)
        elif event.key == 'b':
            self.PlotEpochButterfly(-1)
        elif event.key == 'B':
            self.PlotConditionAverages(self)

    def OnFindRareEvents(self, event):
        dlg = FindRareEventsDialog(self)
        rcode = dlg.ShowModal()
        dlg.Destroy()
        if rcode != wx.ID_OK:
            return
        threshold = float(dlg.threshold.GetValue())
        dlg.StoreConfig()

        # compute and rank SASICA FTc
        y = self.doc.sources.max('time') - self.doc.sources.min('time')
        z = (y - y.mean('case')) / y.std('case')
        z_max = z.max('case').x
        components_ranked = np.argsort(z_max)[::-1]

        # collect output
        res = []
        for c in components_ranked:
            if z_max[c] < threshold:
                break
            z_epochs = z.x[:, c]
            idx = np.flatnonzero(z_epochs >= threshold)
            rank = np.argsort(z_epochs[idx])[::-1]
            res.append((c, z_max[c], idx[rank]))

        if len(res) == 0:
            wx.MessageBox("No rare events were found.", "No Rare Events Found",
                          style=wx.ICON_INFORMATION)
            return

        # format output
        doc = fmtxt.Section("Rare Events")
        doc.add_paragraph("Components that disproportionally affect a small "
                          "number of epochs (z-scored peak-to-peak > %g). "
                          "Epochs are ranked by peak-to-peak." % threshold)
        doc.append(fmtxt.linebreak)
        hash_char = {True: fmtxt.FMTextElement('# ', 'font', {'color': 'green'}),
                     False: fmtxt.FMTextElement('# ', 'font', {'color': 'red'})}
        for c, ft, epochs in res:
            doc.append(hash_char[self.doc.accept[c]])
            doc.append("%i (%.1f):  " % (c, ft))
            doc.append(fmtxt.delim_list((fmtxt.Link(
                self.doc.epoch_labels[e], LINK % (c, e)) for e in epochs)))
            doc.append(fmtxt.linebreak)

        InfoFrame(self, "Rare Events", doc.get_html())

    def OnPanelResize(self, event):
        w, h = event.GetSize()
        n_h = w // self.ax_size
        if n_h >= 2 and n_h != self.n_h:
            self.plot()

    def OnPlotButterfly(self, event):
        self.PlotConditionAverages(self)

    def OnPlotCompSourceArray(self, event):
        self.PlotCompSourceArray(event.EventObject.i)

    def OnPlotCompTopomap(self, event):
        self.PlotCompTopomap(event.EventObject.i)

    def OnPlotGrandAverage(self, event):
        self.PlotEpochButterfly(-1)

    def OnPointerEntersAxes(self, event):
        sb = self.GetStatusBar()
        if event.inaxes:
            sb.SetStatusText("#%i of %i ICA Components" %
                             (event.inaxes.i, len(self.doc.components)))
        else:
            sb.SetStatusText("%i ICA Components" % len(self.doc.components))

    def OnRankEpochs(self, event):
        i_comp = event.EventObject.i
        source = self.doc.sources.sub(component=i_comp)
        y = source - source.mean()
        y /= y.std()
        y **= 2
        ss = y.sum('time').x  # ndvar has epoch as index

        # sort
        sort = np.argsort(ss)[::-1]
        ss = ss[sort]

        # doc
        lst = fmtxt.List("Epochs SS loading in descending order for component "
                         "%i:" % i_comp)
        for i, ss_epoch in enumerate(ss):
            lst.add_item(
                fmtxt.Link(self.doc.epoch_labels[i], LINK % (i_comp, i)) +
                ': %.1f' % ss_epoch)
        doc = fmtxt.Section("# %i Ranked Epochs" % i_comp, lst)

        InfoFrame(self, "Component %i Epoch SS" % i_comp, doc.get_html())

    def OnRightDown(self, event):
        ax = self.canvas.MatplotlibEventAxes(event)
        if not ax:
            return

        # costruct menu
        menu = ContextMenu(ax.i)
        item = menu.Append(wx.ID_ANY, "Rank Epochs")
        self.Bind(wx.EVT_MENU, self.OnRankEpochs, item)
        menu.AppendSeparator()
        item = menu.Append(wx.ID_ANY, "Plot Topomap")
        self.Bind(wx.EVT_MENU, self.OnPlotCompTopomap, item)
        item = menu.Append(wx.ID_ANY, "Plot Source Array")
        self.Bind(wx.EVT_MENU, self.OnPlotCompSourceArray, item)

        # show menu
        pos = self.panel.CalcScrolledPosition(event.Position)
        self.PopupMenu(menu, pos)
        menu.Destroy()

    def OnSetButterflyBaseline(self, event):
        self.butterfly_baseline = event.GetId()

    def OnShowSources(self, event):
        self.ShowSources(0)

    def OnUpdateUIOpen(self, event):
        event.Enable(False)

    def PlotCompFFT(self, i_comp):
        plot.UTS(self.doc.sources.sub(component=i_comp).fft().mean('case'),
                 w=8, title="# %i Frequency Spectrum" % i_comp, legend=False)

    def PlotCompSourceArray(self, i_comp):
        x = self.doc.sources.sub(component=i_comp)
        dim = Categorial('epoch', self.doc.epoch_labels)
        x = NDVar(x.x, (dim,) + x.dims[1:], x.info, x.name)
        plot.Array(x, w=10, h=10,
                   title='# %i' % i_comp, axtitle=False, interpolation='none')

    def PlotCompTopomap(self, i_comp):
        plot.Topomap(self.doc.components[i_comp], w=10, sensorlabels='name',
                     title='# %i' % i_comp)

    def PlotConditionAverages(self, parent):
        "Prompt for model and plot condition averages"
        factors = [n for n, v in self.doc.ds.items() if
                   isinstance(v, Factor)]
        if len(factors) == 0:
            wx.MessageBox("The dataset that describes the epochs does not "
                          "contain any Factors that could be used to plot the "
                          "data by condition.", "No Factors in Dataset",
                          style=wx.ICON_ERROR)
            return
        elif len(factors) == 1:
            default = factors[0]
        else:
            default = self.last_model or factors[0]
        msg = "Specify the model (available factors: %s)" % ', '.join(factors)

        plot_model = None
        dlg = wx.TextEntryDialog(parent, msg, "Plot by Condition", default)
        while plot_model is None:
            if dlg.ShowModal() == wx.ID_OK:
                value = dlg.GetValue()
                use = [s.strip() for s in value.replace(':', '%').split('%')]
                invalid = [f for f in use if f not in factors]
                if invalid:
                    wx.MessageBox("The following are not valid factor names: %s"
                                  % (', '.join(invalid)), "Invalid Entry",
                                  wx.ICON_ERROR)
                else:
                    plot_model = '%'.join(use)
            else:
                dlg.Destroy()
                return
        dlg.Destroy()
        self.last_model = value

        ds = self.doc.ds.aggregate(plot_model, drop_bad=True)
        titles = [' '.join(ds[i, f] for f in use) + ' (n=%i)' % ds[i, 'n'] for
                  i in range(ds.n_cases)]
        self._PlotButterfly(ds['epochs'], titles)

    def PlotEpochButterfly(self, i_epoch):
        if i_epoch == -1:
            self._PlotButterfly(self.doc.epochs.average(), "Epochs Average")
        else:
            self._PlotButterfly(self.doc.epochs[i_epoch],
                                "Epoch " + self.doc.epoch_labels[i_epoch],
                                vmax=2e-12)

    def _PlotButterfly(self, epoch, title, vmax=None):
        original = asndvar(epoch)
        clean = asndvar(self.doc.apply(epoch))
        if self.butterfly_baseline == ID.BASELINE_CUSTOM:
            original -= original.mean(time=(None, 0))
            clean -= clean.mean(time=(None, 0))
        elif self.butterfly_baseline == ID.BASELINE_GLOABL_MEAN:
            original -= self.doc.global_mean
            clean -= self.doc.global_mean

        if original.has_case:
            if isinstance(title, str):
                title = repeat(title, len(original))
            if vmax is None:
                vmax = 1.1 * max(abs(original.min()), original.max())
            for data, title_ in zip(zip(original, clean), title):
                plot.TopoButterfly(data, vmax=vmax, title=title_,
                                   axtitle=("Original", "Cleaned"))
        else:
            plot.TopoButterfly([original, clean], title=title,
                               axtitle=("Original", "Cleaned"))

    def ShowSources(self, i_first):
        if self.source_frame:
            self.source_frame.Raise()
        else:
            self.source_frame = SourceFrame(self, i_first)


class SourceFrame(FileFrameChild):
    """Component source time course display for selecting ICA components.

    * Click on components topographies to select/deselect them.

    *Keyboard shortcuts* in addition to the ones in the menu:

    =========== ============================================================
    Key         Effect
    =========== ============================================================
    arrows      scroll through components/epochs
    t           topomap plot of the component under the pointer
    a           array-plot of the source time course of the component under
                the pointer
    f           plot the frequency spectrum for the component under the
                pointer
    b           butterfly plot of the original and cleaned data (of the
                epoch under the pointer, or of the grand average if the
                pointer is over other elements)
    B           Butterfly plot of condition averages
    =========== ============================================================
    """
    _doc_name = 'component selection'
    _name = 'ICASources'
    _title = 'ICA Source Time Course'
    _wildcard = "ICA fiff file (*-ica.fif)|*.fif"

    def __init__(self, parent, i_first):
        FileFrame.__init__(self, parent, None, None, parent.model)

        # prepare canvas
        self.canvas = FigureCanvasPanel(self)
        self.figure = self.canvas.figure
        self.figure.subplots_adjust(0, 0, 1, 1, 0, 0)
        self.figure.set_facecolor('white')

        # attributes
        self.parent = parent
        self.model = parent.model
        self.doc = parent.model.doc
        self.n_comp = self.config.ReadInt('layout_n_comp', 10)
        self.n_comp_in_ica = len(self.doc.components)
        self.i_first = i_first
        self.n_epochs = self.config.ReadInt('layout_n_epochs', 20)
        self.i_first_epoch = 0
        self.n_epochs_in_data = len(self.doc.sources)
        self.y_scale = self.config.ReadFloat('y_scale', 10)  # scale factor for y axis

        # Toolbar
        tb = self.InitToolbar(can_open=False)
        tb.AddSeparator()
        self.up_button = tb.AddTool(wx.ID_UP, "Up", Icon("tango/actions/go-up"))
        self.down_button = tb.AddTool(wx.ID_DOWN, "Down", Icon("tango/actions/go-down"))
        self.back_button = tb.AddTool(wx.ID_BACKWARD, "Back", Icon("tango/actions/go-previous"))
        self.next_button = tb.AddTool(wx.ID_FORWARD, "Next", Icon("tango/actions/go-next"))
        tb.AddStretchableSpace()
        self.InitToolbarTail(tb)
        tb.Realize()

        # event bindings
        self.doc.callbacks.subscribe('case_change', self.CaseChanged)
        self.Bind(wx.EVT_TOOL, self.OnUp, id=wx.ID_UP)
        self.Bind(wx.EVT_TOOL, self.OnDown, id=wx.ID_DOWN)
        self.Bind(wx.EVT_TOOL, self.OnBackward, id=wx.ID_BACKWARD)
        self.Bind(wx.EVT_TOOL, self.OnForward, id=wx.ID_FORWARD)
        self.canvas.mpl_connect('button_press_event', self.OnCanvasClick)
        self.canvas.mpl_connect('key_release_event', self.OnCanvasKey)

        self._plot()
        self.UpdateTitle()
        self.Show()

    def _get_source_data(self):
        "Return ``(source_data, epoch-labels)`` tuple for current page"
        n_comp = self.n_comp
        n_comp_actual = self.n_comp_actual
        epoch_index = slice(self.i_first_epoch, self.i_first_epoch + self.n_epochs)
        data = self.doc.sources.sub(
            case=epoch_index, component=slice(self.i_first, self.i_first + n_comp))
        y = data.get_data(('component', 'case', 'time')).reshape((n_comp_actual, -1))
        if y.base is not None and data.x.base is not None:
            y = y.copy()
        start = n_comp - 1
        stop = -1 + (n_comp - n_comp_actual)
        y += np.arange(start * self.y_scale, stop * self.y_scale, -self.y_scale)[:, None]
        return y, self.doc.epoch_labels[epoch_index]

    def _plot(self):
        # partition figure
        self.figure.clf()
        figheight = self.figure.get_figheight()
        n_comp = self.n_comp
        # make sure there are no empty lines
        if self.i_first and self.n_comp_in_ica - self.i_first < n_comp:
            self.i_first = max(0, self.n_comp_in_ica - n_comp)
        # further layout-relevant properties
        n_comp_actual = min(self.n_comp_in_ica - self.i_first, n_comp)
        self.n_comp_actual = n_comp_actual
        elen = len(self.doc.sources.time)

        # topomaps
        axheight = 1 / (self.n_comp + 0.5)  # 0.5 = bottom space for epoch labels
        ax_size_in = axheight * figheight
        axwidth = ax_size_in / self.figure.get_figwidth()
        left = axwidth / 2
        self.topo_plots = []
        self.topo_labels = []
        for i in range(n_comp_actual):
            i_comp = self.i_first + i
            ax = self.figure.add_axes((left, 1 - (i + 1) * axheight, axwidth, axheight))
            p = _ax_topomap(ax, [self.doc.components[i_comp]], **TOPO_ARGS)
            text = ax.text(0, 0.5, "# %i" % i_comp, va='center', ha='right', color='k')
            ax.i = i
            ax.i_comp = i_comp
            self.topo_plots.append(p)
            self.topo_labels.append(text)

        # source time course data
        y, tick_labels = self._get_source_data()

        # axes
        left = 1.5 * axwidth
        bottom = 1 - n_comp * axheight
        ax = self.figure.add_axes((left, bottom, 1 - left, 1 - bottom),
                                  frameon=False, yticks=(),
                                  xticks=np.arange(elen / 2, elen * self.n_epochs, elen),
                                  xticklabels=tick_labels)
        ax.tick_params(bottom=False)
        ax.i = -1
        ax.i_comp = None

        # store canvas before plotting lines
        self.canvas.draw()

        # plot epochs
        self.lines = ax.plot(y.T, color=LINE_COLOR[True], clip_on=False)
        ax.set_ylim((-0.5 * self.y_scale, (self.n_comp - 0.5) * self.y_scale))
        ax.set_xlim((0, y.shape[1]))
        # line color
        reject_color = LINE_COLOR[False]
        for i in range(n_comp_actual):
            if not self.doc.accept[i + self.i_first]:
                self.lines[i].set_color(reject_color)
        # epoch demarcation
        for x in range(elen, elen * self.n_epochs, elen):
            ax.axvline(x, ls='--', c='k')

        self.ax_tc = ax
        self.canvas.draw()

    def CanBackward(self):
        return self.i_first_epoch > 0

    def CanDown(self):
        return self.i_first + self.n_comp < self.n_comp_in_ica

    def CanForward(self):
        return self.i_first_epoch + self.n_epochs < self.n_epochs_in_data

    def CanUp(self):
        return self.i_first > 0

    def CaseChanged(self, index):
        "Update the states of the segments on the current page"
        if isinstance(index, int):
            index = [index]
        elif isinstance(index, slice):
            start = index.start or 0
            stop = index.stop or self.doc.n_epochs
            index = range(start, stop)
        elif index.dtype.kind == 'b':
            index = np.nonzero(index)[0]

        # filter to visible epochs
        i_last = self.i_first + self.n_comp_actual
        index = [i_comp for i_comp in index if self.i_first <= i_comp <= i_last]
        # update epoch plots
        if index:
            for i_comp in index:
                self.lines[i_comp - self.i_first].set_color(LINE_COLOR[self.doc.accept[i_comp]])
            self.canvas.draw()

    def GoToComponentEpoch(self, component, epoch):
        self.SetFirstComponent(component // self.n_comp * self.n_comp)
        self.SetFirstEpoch(epoch // self.n_epochs * self.n_epochs)

    def OnBackward(self, event):
        "Turn the page backward"
        self.SetFirstEpoch(self.i_first_epoch - self.n_epochs)

    def OnCanvasClick(self, event):
        "Called by mouse clicks"
        if event.inaxes:
            if event.inaxes.i_comp is None:
                i_comp = int(self.i_first + self.n_comp - ceil(event.ydata / self.y_scale + 0.5))
                if i_comp >= self.n_comp_in_ica:
                    return
            else:
                i_comp = event.inaxes.i_comp
            self.model.toggle(i_comp)

    def OnCanvasKey(self, event):
        if event.key is None:
            return
        elif event.key == 'down':
            if self.CanDown():
                self.OnDown(None)
        elif event.key == 'up':
            if self.CanUp():
                self.OnUp(None)
        elif event.key == 'right':
            if self.CanForward():
                self.OnForward(None)
        elif event.key == 'left':
            if self.CanBackward():
                self.OnBackward(None)
        elif event.key == 'B':
            self.parent.PlotConditionAverages(self)
        elif event.key == 'b':
            if event.inaxes is None:
                i_epoch = -1
            elif event.inaxes.i_comp is None:
                i_epoch = (self.i_first_epoch +
                           int(event.xdata // len(self.doc.sources.time)))
                if i_epoch >= len(self.doc.epochs):
                    i_epoch = -1
            else:
                i_epoch = -1
            self.parent.PlotEpochButterfly(i_epoch)
        elif not event.inaxes:
            return
        elif event.key in 'tT':
            self.parent.PlotCompTopomap(event.inaxes.i_comp)
        elif event.key == 'a':
            self.parent.PlotCompSourceArray(event.inaxes.i_comp)
        elif event.key == 'f':
            self.parent.PlotCompFFT(event.inaxes.i_comp)

    def OnClose(self, event):
        if super(SourceFrame, self).OnClose(event):
            self.config.WriteInt('layout_n_comp', self.n_comp)
            self.config.WriteInt('layout_n_epochs', self.n_epochs)
            self.config.WriteFloat('y_scale', self.y_scale)
            self.config.Flush()

    def OnDown(self, event):
        "Turn the page backward"
        self.SetFirstComponent(self.i_first + self.n_comp)

    def OnForward(self, event):
        "Turn the page forward"
        self.SetFirstEpoch(self.i_first_epoch + self.n_epochs)

    def OnSetLayout(self, event):
        caption = "Set ICA Source Layout"
        msg = "Number of components and epochs (e.g., '10 20')"
        default = '%i %i' % (self.n_comp, self.n_epochs)
        dlg = wx.TextEntryDialog(self, msg, caption, default)
        while True:
            if dlg.ShowModal() == wx.ID_OK:
                value = dlg.GetValue()
                try:
                    n_comp, n_epochs = map(int, value.split())
                except Exception:
                    wx.MessageBox("Invalid entry: %r. Need two integers \n"
                                  "(e.g., '10 20').", "Invalid Entry",
                                  wx.OK | wx.ICON_ERROR)
                else:
                    dlg.Destroy()
                    break
            else:
                dlg.Destroy()
                return

        self.n_comp = n_comp
        self.n_epochs = n_epochs
        self._plot()

    def OnSetVLim(self, event):
        dlg = wx.TextEntryDialog(self, "Y-axis scale:", "Y-Axis Scale",
                                 "%g" % (10. / self.y_scale,))
        value = None
        while True:
            if dlg.ShowModal() != wx.ID_OK:
                break
            try:
                value = float(dlg.GetValue())
            except Exception as exception:
                msg = wx.MessageDialog(
                    self, str(exception), "Invalid Entry",
                    wx.OK | wx.ICON_ERROR)
                msg.ShowModal()
                msg.Destroy()
            else:
                break
        dlg.Destroy()
        if value is not None:
            self.y_scale = 10. / value
            # redraw
            self.SetFirstEpoch(self.i_first_epoch)

    def OnUp(self, event):
        "Turn the page backward"
        self.SetFirstComponent(self.i_first - self.n_comp)

    def OnUpdateUIBackward(self, event):
        event.Enable(self.CanBackward())

    def OnUpdateUIDown(self, event):
        event.Enable(self.CanDown())

    def OnUpdateUIForward(self, event):
        event.Enable(self.CanForward())

    def OnUpdateUISetLayout(self, event):
        event.Enable(True)

    def OnUpdateUISetVLim(self, event):
        event.Enable(True)

    def OnUpdateUIUp(self, event):
        event.Enable(self.CanUp())

    def SetFirstComponent(self, i_first):
        if i_first < 0:
            i_first = 0
        elif i_first >= self.n_comp_in_ica:
            i_first = self.n_comp_in_ica - 1

        n_comp_actual = min(self.n_comp_in_ica - i_first, self.n_comp)
        for i in range(n_comp_actual):
            p = self.topo_plots[i]
            i_comp = i_first + i
            p.set_data([self.doc.components[i_comp]], True)
            p.ax.i_comp = i_comp
            self.topo_labels[i].set_text("# %i" % i_comp)
            self.lines[i].set_color(LINE_COLOR[self.doc.accept[i_comp]])

        if n_comp_actual < self.n_comp:
            empty_data = self.doc.components[0].copy()
            empty_data.x.fill(0)
            for i in range(n_comp_actual, self.n_comp):
                p = self.topo_plots[i]
                p.set_data([empty_data])
                p.ax.i_comp = -1
                self.topo_labels[i].set_text("")
                self.lines[i].set_color('white')

        self.i_first = i_first
        self.n_comp_actual = n_comp_actual
        self.SetFirstEpoch(self.i_first_epoch)

    def SetFirstEpoch(self, i_first_epoch):
        self.i_first_epoch = i_first_epoch
        y, tick_labels = self._get_source_data()
        if i_first_epoch + self.n_epochs > self.n_epochs_in_data:
            elen = len(self.doc.sources.time)
            n_missing = self.i_first_epoch + self.n_epochs - self.n_epochs_in_data
            pad_time = elen * n_missing
        else:
            pad_time = 0

        if self.n_comp_actual < self.n_comp:
            pad_comp = self.n_comp - self.n_comp_actual
        else:
            pad_comp = 0

        if pad_time or pad_comp:
            y = np.pad(y, ((0, pad_comp), (0, pad_time)), 'constant')

        for line, data in zip(self.lines, y):
            line.set_ydata(data)
        self.ax_tc.set_xticklabels(tick_labels)
        self.ax_tc.set_ylim((-0.5 * self.y_scale, (self.n_comp - 0.5) * self.y_scale))
        self.canvas.draw()


class FindRareEventsDialog(EelbrainDialog):
    def __init__(self, parent, *args, **kwargs):
        super(FindRareEventsDialog, self).__init__(parent, wx.ID_ANY,
                                                   "Find Rare Events", *args,
                                                   **kwargs)
        config = parent.config
        threshold = config.ReadFloat("FindRareEvents/threshold", 2.)

        sizer = wx.BoxSizer(wx.VERTICAL)

        # Threshold
        sizer.Add(wx.StaticText(self, label="Threshold for rare epochs\n"
                                            "(z-scored peak-to-peak value):"))
        validator = REValidator(POS_FLOAT_PATTERN, "Invalid entry: {value}. Please "
                                "specify a number > 0.", False)
        ctrl = wx.TextCtrl(self, value=str(threshold), validator=validator)
        ctrl.SetHelpText("Epochs whose z-scored peak-to-peak value exceeds "
                         "this value are considered rare")
        ctrl.SelectAll()
        sizer.Add(ctrl)
        self.threshold = ctrl

        # default button
        btn = wx.Button(self, wx.ID_DEFAULT, "Default Settings")
        sizer.Add(btn, border=2)
        btn.Bind(wx.EVT_BUTTON, self.OnSetDefault)

        # buttons
        button_sizer = wx.StdDialogButtonSizer()
        # ok
        btn = wx.Button(self, wx.ID_OK)
        btn.SetDefault()
        button_sizer.AddButton(btn)
        # cancel
        btn = wx.Button(self, wx.ID_CANCEL)
        button_sizer.AddButton(btn)
        # finalize
        button_sizer.Realize()
        sizer.Add(button_sizer)

        self.SetSizer(sizer)
        sizer.Fit(self)

    def OnSetDefault(self, event):
        self.threshold.SetValue('2')

    def StoreConfig(self):
        config = self.Parent.config
        config.WriteFloat("FindRareEvents/threshold",
                          float(self.threshold.GetValue()))
        config.Flush()


class InfoFrame(HTMLFrame):

    def OpenURL(self, url):
        m = re.match('^component:(\d+) epoch:(\d+)$', url)
        if m:
            comp, epoch = m.groups()
            self.Parent.GoToComponentEpoch(int(comp), int(epoch))
        else:
            raise ValueError("Invalid link URL: %r" % url)
