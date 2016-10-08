from __future__ import print_function

from cPickle import dump, HIGHEST_PROTOCOL, Unpickler
from importlib import import_module
import os

from .._utils import ui


def pickle(obj, dest=None, protocol=HIGHEST_PROTOCOL):
    """Pickle a Python object.

    Parameters
    ----------
    dest : None | str
        Path to destination where to save the  file. If no destination is
        provided, a file dialog is shown. If a destination without extension is
        provided, '.pickled' is appended.
    protocol : int
        Pickle protocol (default is HIGHEST_PROTOCOL).
    """
    if dest is None:
        filetypes = [("Pickled Python Objects (*.pickled)", '*.pickled')]
        dest = ui.ask_saveas("Pickle Destination", "", filetypes)
        if dest is False:
            raise RuntimeError("User canceled")
        else:
            print('dest=%r' % dest)
    else:
        dest = os.path.expanduser(dest)
        if not os.path.splitext(dest)[1]:
            dest += '.pickled'

    with open(dest, 'wb') as fid:
        dump(obj, fid, protocol)


def map_paths(module_name, class_name):
    "Subclass to handle changes in module paths"
    if module_name == 'eelbrain.vessels.data':
        module_name = 'eelbrain._data_obj'
        class_names = {'var': 'Var', 'factor': 'Factor', 'ndvar': 'NDVar',
                       'datalist': 'Datalist', 'dataset': 'Dataset'}
        class_name = class_names[class_name]
    elif module_name.startswith('eelbrain.data.'):
        if module_name.startswith('eelbrain.data.load'):
            rev = module_name.replace('.data.load', '.load')
        elif module_name.startswith('eelbrain.data.stats'):
            rev = module_name.replace('.data.stats', '._stats')
        elif module_name.startswith('eelbrain.data.data_obj'):
            rev = module_name.replace('.data.data_obj', '._data_obj')
        else:
            raise NotImplementedError("%r / %r" % (module_name, class_name))
        module_name = rev

    module = import_module(module_name)
    return getattr(module, class_name)


def unpickle(file_path=None):
    """Load pickled Python objects from a file.

    Almost like ``cPickle.load(open(file_path))``, but also loads object saved
    with older versions of Eelbrain, and allows using a system file dialog to
    select a file.

    Parameters
    ----------
    file_path : None | str
        Path to a pickled file. If None (default), a system file dialog will be
        shown. If the user cancels the file dialog, a RuntimeError is raised.
    """
    if file_path is None:
        filetypes = [("Pickles (*.pickled)", '*.pickled'), ("All files", '*')]
        file_path = ui.ask_file("Select File to Unpickle", "Select a pickled "
                                "file to unpickle", filetypes)
        if file_path is False:
            raise RuntimeError("User canceled")
        else:
            print(repr(file_path))
    else:
        file_path = os.path.expanduser(file_path)
        if not os.path.exists(file_path):
            new_path = os.extsep.join((file_path, 'pickled'))
            if os.path.exists(new_path):
                file_path = new_path

    with open(file_path, 'rb') as fid:
        unpickler = Unpickler(fid)
        unpickler.find_global = map_paths
        obj = unpickler.load()

    return obj