# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from os.path import join

import mne
from eelbrain import datasets, gui
from eelbrain._utils.testing import gui_test, requires_mne_sample_data, TempDir


@gui_test
@requires_mne_sample_data
def test_select_components():
    "Test Select-Epochs GUI Document"
    tempdir = TempDir()
    PATH = join(tempdir, 'test-ica.fif')

    ds = datasets.get_mne_sample()
    ds['epochs'] = ds['epochs'].pick_types('mag')
    ica = mne.preprocessing.ICA(0.95)
    ica.fit(ds['epochs'])
    ica.save(PATH)

    frame = gui.select_components(PATH, ds)
    frame.model.toggle(1)
    frame.OnSave(None)
    ica = mne.preprocessing.read_ica(PATH)
    assert ica.exclude == [1]

    frame.OnUndo(None)
    frame.OnSave(None)
    ica = mne.preprocessing.read_ica(PATH)
    assert ica.exclude == []

    frame.Close()
