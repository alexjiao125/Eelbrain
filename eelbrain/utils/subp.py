'''

for permission errors: try ``os.chmod`` or ``os.chown``


subprocess documentation
------------------------

http://docs.python.org/library/subprocess.html
http://www.doughellmann.com/PyMOTW/subprocess/
http://explanatorygap.net/2010/05/10/python-subprocess-over-shell/
http://codeghar.wordpress.com/2011/12/09/introduction-to-python-subprocess-module/


Created on Mar 4, 2012

@author: christian
'''

import fnmatch
import logging
import os
import re
import shutil
import subprocess
import tempfile

import numpy as np

from eelbrain import ui
from eelbrain.vessels import data as _dt


__hide__ = ['os', 'shutil', 'subprocess', 'tempfile', 're', 'fnmatch',
            'np',
            'ui']
#__all__ = [
##           'forward',
#           'kit2fiff', 
#           'process_raw', 
#           'set_bin_dirs',
#           'mne_experiment'
#           ] 

# keep track of whether the mne dir has been successfully set
_bin_dirs = {'mne': None,
             'freesurfer': None,
             'edfapi': None}

# create dictionary of available sns files
_sns_files = {}
_sns_dir = __file__
for i in xrange(2):
    _sns_dir = os.path.dirname(_sns_dir)
_sns_dir = os.path.join(_sns_dir, 'Resources', 'sns')
for name in ['NYU-nellab']:
    _sns_files[name] = os.path.join(_sns_dir, name+'.txt')


_verbose = 1


def set_bin_dirs(mne=None, freesurfer=None, edf=None):
    """
    Set the directories where binaries are installed. E.g. ::
    
        >>> set_bin_dirs(mne='~/unix_apps/mne-2.7.3')
    
    """
    if mne:
        mne = os.path.expanduser(mne)
        if os.path.exists(mne):
            os.environ['MNE_ROOT'] = mne
            os.environ['DYLD_LIBRARY_PATH'] = os.path.join(mne, 'lib')
            
            mne_bin = os.path.join(mne, 'bin')
            if 'PATH' in os.environ:
                os.environ['PATH'] += ':%s' % mne_bin
            else:
                os.environ['PATH'] = mne_bin
            _bin_dirs['mne'] = mne
        else:
            raise IOError("%r does not exist" % mne)
    
    if freesurfer:
        freesurfer = os.path.expanduser(freesurfer)
        if os.path.exists(mne):
            os.environ['FREESURFER_HOME'] = freesurfer
        else:
            raise IOError("%r does not exist" % freesurfer)
    
    if edf:
        edf = os.path.expanduser(edf)
        if os.path.exists(edf):
            _bin_dirs['edfapi'] = edf



class edf_file:
    """
    Converts an "eyelink data format" (.edf) file to a temporary directory
    and parses its content.
    
    """
    def __init__(self, path):
        # convert
        if not os.path.exists(path):
            err = "File does not exist: %r" % path
            raise ValueError(err)
        
        self.source_path = path
        self.temp_dir = tempfile.mkdtemp()
        cmd = [os.path.join(_bin_dirs['edfapi'], 'edf2asc'), # options in Manual p. 106
               '-t', # use only tabs as delimiters
               '-e', # outputs event data only
               '-nse', # blocks output of start events
               '-p', self.temp_dir, # writes output with same name to <path> directory
               path]
        
        _run(cmd)
        
        # find asc file
        name, _ = os.path.splitext(os.path.basename(path))
        ascname = os.path.extsep.join((name, 'asc'))
        self.asc_path = os.path.join(self.temp_dir, ascname)
        
        # data containers
        self.raw = [] # all lines
        self.preamble = []
        self.lines = []
        self.positions = positions = []
        self.events = events = []
        
        self.MSGs = MSGs = [] # all MSG lines
        triggers = [] # MSGs which are MEG triggers
        
        artifact_names = ['ESACC', 'EBLINK']
        artifacts = []
        a_dtype = np.dtype([('event', np.str_, 5), 
                            ('start', np.uint32), 
                            ('stop', np.uint32)])
#        a_dtype = np.dtype({'names': ['event', 'start', 'stop'], 
##                            'titles': ['a', 'b', 'c'],  
#                            'formats': [(np.str_, 5), np.uint32, np.uint32]})
        
        # parse asc file
        comment_chars = ('#', '/', ';')
        is_recording = False # keep track of whether we are in a BEGIN -> END block
        for line in open(self.asc_path):
            line = line.strip()
            if line:
                items = line.split()
                i0 = items[0]
                self.raw.append(line)
            else:
                continue
            
            if line.startswith('*'):
                self.preamble.append(line)
            elif any(line.startswith(c) for c in comment_chars):
                pass
            elif is_recording:
                if i0.isdigit(): # data line
#                    pos = tuple(int(p) for p in items[1:4]) # (t, x, y)
                    pos = items
                    positions.append(pos)
                elif i0 == 'MSG':
                    MSGs.append(items)
                    if items[2] == 'MEG':
                        t = np.uint32(items[1])
                        v = np.uint8(items[4])
                        triggers.append((t, v))
                elif i0 in artifact_names:
                    start = np.int32(items[2])
                    stop = np.int32(items[3])
                    evt = np.array((i0[1:6], start, stop), a_dtype)
                    artifacts.append(evt)
                elif i0 == 'END':
                    is_recording = False
                else:
                    self.lines.append(items)
            else:
                if i0 == 'START':
                    is_recording = True
        
        self.artifacts = np.array(artifacts, a_dtype)
        self.triggers = np.array(triggers, dtype=[('time', np.uint32), ('ID', np.uint8)])
    
    def __del__(self):
        shutil.rmtree(self.temp_dir)
    
    def __repr__(self):
        return 'edf_file(%r)' % self.source_path
    
    def asdataset(self, accept=True, tstart=-0.1, tstop=0.6):
        ds = _dt.dataset(name=self.source_path)
        ds['eventID'] = _dt.var(self.triggers['ID'])
        ds['time'] = _dt.var(self.triggers['time'])
        if accept:
            ds['accept'] = _dt.var(self.get_acceptable(tstart, tstop))
        return ds
    
    def get_acceptable(self, tstart=-0.1, tstop=0.6):
        # conert to ms
        start = int(tstart * 1000)
        stop = int(tstop * 1000)
        
        self._debug = []
        
        # get data for triggers
        N = len(self.triggers)
        accept = np.empty(N, np.bool_)
        for i, (t, _) in enumerate(self.triggers):
            starts_before_tstop = self.artifacts['start'] < t + stop
            stops_after_tstart = self.artifacts['stop'] > t + start
            overlap = np.all((starts_before_tstop, stops_after_tstart), axis=0)
            accept[i] = not np.any(overlap)
            self._debug.append(overlap)
        
        return accept
    
    def print_lines(self, *lines):
        n = len(lines)
        if n == 0:
            start, stop = 0, None
        elif n == 1:
            start, stop = 0, lines[0]
        else:
            start, stop = lines
        
        for line in self.lines[start:stop]:
            print line
    
    def print_raw(self, *lines):
        n = len(lines)
        if n == 0:
            start, stop = 0, None
        elif n == 1:
            start, stop = 0, lines[0]
        else:
            start, stop = lines
        
        for line in self.raw[start:stop]:
            print line




class marker_avg_file:
    def __init__(self, path):
        # Parse marker file, based on Tal's pipeline:
        regexp = re.compile(r'Marker \d:   MEG:x= *([\.\-0-9]+), y= *([\.\-0-9]+), z= *([\.\-0-9]+)')
        output_lines = []
        for line in open(path):
            match = regexp.search(line)
            if match:
                output_lines.append('\t'.join(match.groups()))
        txt = '\n'.join(output_lines)
        
        fd, self.path = tempfile.mkstemp(suffix='hpi', text=True)
        f = os.fdopen(fd, 'w')
        f.write(txt)
        f.close()
    
    def __del__(self):
        os.remove(self.path)



def _format_path(path, fmt, is_new=False):
    "helper function to format the path to mne files"
    if not isinstance(path, basestring):
        path = os.path.join(*path)
    
    if fmt:
        path = path.format(**fmt)
    
    # test the path
    path = os.path.expanduser(path)
    if is_new or os.path.exists(path):
        return path
    else:
        raise IOError("%r does not exist" % path)
    


def kit2fiff(paths=dict(mrk = None,
                        elp = None,
                        hsp = None,
                        rawtxt = None,
                        rawfif = None),
             sns='NYU-nellab',
             sfreq=1000, lowpass=100, highpass=0,
             stim=xrange(168, 160, -1),  stimthresh=2.5, add=None,#(188, 190), #xrange()
             aligntol=25):
    """
    Calls the ``mne_kit2fiff`` binary which reads multiple input files and 
    combines them into a fiff file. Implemented after Gwyneth's Manual; for 
    more information see the mne manual (p. 222). 
    
    **Arguments:**
    
    paths : dict
        Dictionary containing paths to input and output files. 
        Needs the folowing keys:
        'mrk', 'elp', 'hsp', 'sns', 'rawtxt', 'outfif' 
        
    experiment : str
        The experiment name as it appears in file names.
    
    highpass : scalar
        The highpass filter corner frequency (only for file info, does not
        filter the data). 0 Hz for DC recording.
    
    lowpass : scalar
        like highpass
    
    meg_sdir : str or tuple (see above)
        Path to the subjects's meg directory. If ``None``, a file dialog is 
        displayed to ask for the directory.
    
    sfreq : scalar
        samplingrate of the data
    
    stimthresh : scalar
        The threshold value used when synthesizing the digital trigger channel
    
    add : sequence of int | None
        channels to include in addition to the 157 default MEG channels and the
        digital trigger channel. These numbers refer to the scanning order 
        channels as listed in the sns file, starting from one.
    
    stim : iterable over ints
        trigger channels that are used to reconstruct the event cue value from 
        the separate trigger channels. The default ``xrange(168, 160, -1)`` 
        reconstructs the values sent through psychtoolbox.
        
    aligntol : scalar
        Alignment tolerance for coregistration
    
    """
    mne_dir = _bin_dirs['mne']
    if not mne_dir:
        raise IOError("mne-dir not set. See mne_link.set_bin_dirs.")
    
    # get all the paths
    mrk_path = paths.get('mrk')
    elp_file = paths.get('elp')
    hsp_file = paths.get('hsp')
    raw_file = paths.get('rawtxt')
    out_file = paths.get('rawfif')
    
    if sns in _sns_files:
        sns_file = _sns_files[sns]
    elif os.path.exists(sns):
        sns_file = sns
    else:
        err = ("sns needs to the be name of a provided sns file (%s) ro a valid"
               "path to an sns file" % ', '.join(map(repr, _sns_files)))
        raise IOError(err)
    
    # convert the marker file
    mrk_file = marker_avg_file(mrk_path)
    hpi_file = mrk_file.path

    # make sure target path exists
    out_dir = os.path.dirname(out_file)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    elif os.path.exists(out_file):
        if not ui.ask("Overwrite?", "Target File Already Exists at: %r. Should "
                      "it be replaced?" % out_file):
            return
        
    cmd = [os.path.join(mne_dir, 'bin', 'mne_kit2fiff'),
           '--elp', elp_file,
           '--hsp', hsp_file,
           '--sns', sns_file,
           '--hpi', hpi_file,
           '--raw', raw_file,
           '--out', out_file,
           '--stim', ':'.join(map(str, stim)), # '161:162:163:164:165:166:167:168'
           '--sfreq', sfreq,
           '--aligntol', aligntol,
           '--lowpass', lowpass,
           '--highpass', highpass,
           '--stimthresh', stimthresh,
           ]
    
    if add:
        cmd.extend(('--add', ':'.join(map(str, add))))
    
    _run(cmd)
    
    # TODO: rename additional channels
    # how do I know their names?? ~/Desktop/test/EOG  has MISC 28 and MISC 30
#    if add:
#        cmd = [os.path.join(mne_dir, 'bin', 'mne_rename_channels'),
#               '--fif', out_file,
#               ]
    
    if not os.path.exists(out_file):
        raise RuntimeError("kit2fiff failed (see above)")



def process_raw(raw, save='{raw}_filt', args=[], **kwargs):
    """
    Calls ``mne_process_raw`` to process raw fiff files. All 
    options can be submitted in ``args`` or as kwargs (for a description of
    the options, see mne manual 2.7.3, ch. 4 / p. 41)
    
    raw : str(path)
        Source file. 
    save : str(path)
        Destination file. ``'{raw}'`` will be replaced with the raw file path
        without the extension.
    args : list of str
        Options that do not require a value (e.g., to specify ``--filteroff``,
        use ``args=['filteroff']``). 
    kwargs : 
        Options that require values.
    
    Example::
    
        >>> process_raw('/some/file_raw.fif', highpass=8, lowpass=12)
    
    """
    raw_name, _ = os.path.splitext(raw)
    if raw_name.endswith('_raw'):
        raw_name = raw_name[:-4]
    
    save = save.format(raw=raw_name)
    
    cmd = [os.path.join(_bin_dirs['mne'], 'bin', 'mne_process_raw'),
           '--raw', raw,
           '--save', save]
    
    for arg in args:
        cmd.append('--%s' % arg)
    
    for key, value in kwargs.iteritems():
        cmd.append('--%s' % key)
        cmd.append(value)
    
    _run(cmd)



def _run(cmd, v=None):
    """
    cmd: list of strings
        command that is submitted to subprocess.Popen.
    v : 0 | 1 | 2 | None
        verbosity level (0: nothing;  1: stderr;  2: stdout;  None: use 
        _verbose module attribute)
    
    """
    if v is None:
        v = _verbose
    
    if v > 1:
        print "> COMMAND:"
        for line in cmd:
            print repr(line)
    
    cmd = [unicode(c) for c in cmd]
    sp = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = sp.communicate()
    
    if v > 1:
        print "\n> OUT:"
        print stdout
    
    if v > 0 and stderr:
        print '\n> ERROR:'
        print stderr
    
    return stdout, stderr


def setup_mri(mri_sdir, ico=4):
    """
    runs mne_setup_forward_model (see MNE manual section 3.7, p. 25)
    
    """
    mri_dir, subject = os.path.split(mri_sdir)
    bemdir = os.path.join(mri_sdir, 'bem')
    
    # symlinks (MNE-manual 3.6, p. 24 / Gwyneth's Manual X) 
    for name in ['inner_skull', 'outer_skull', 'outer_skin']:
        # can I make path relative by omitting initial bemdir,  ?
        src = os.path.join('watershed', '%s_%s_surface' % (subject, name)) 
        dest = os.path.join(bemdir, '%s.surf' % name)
        if os.path.exists(dest):
            if os.path.islink(dest):
                if os.path.realpath(dest) == src:
                    pass
                else:
                    logging.debug("replacing symlink: %r" % dest)
                    os.remove(dest)
                    os.symlink(src, dest)
                    # can raise an OSError: [Errno 13] Permission denied
            else:
                raise IOError("%r exists and is no symlink" % dest)
        else:
            os.symlink(src, dest)    
    
    # mne_setup_forward_model
    os.environ['SUBJECTS_DIR'] = mri_dir    
    cmd = [os.path.join(_bin_dirs['mne'], 'bin', "mne_setup_forward_model"),
           '--subject', subject,
           '--surf',
           '--ico', ico,
           '--homog']
    
    _run(cmd)
    # -> creates a number of files in <mri_sdir>/bem



def run_mne_analyze(mri_dir, fif_dir, modal=True):
    """
    invokes mne_analyze (e.g., for manual coregistration)
    
    **Arguments:**
    
    mri_dir : str(path)
        the directory containing the mri data (subjects's mri directory, or 
        fsaverage)
    fif_file : str(path)
        the target fiff file
    modal : bool
        causes the shell to be unresponsive until mne_analyze is closed
    
    
    **Coregistration Procedure:**
    
    (For more information see  MNE-manual 3.10 & 12.11, or section XI of Gwyneth's manual.)
    
    #. File > Load Surface: select the subject`s directory and "Inflated"
    #. File > Load digitizer data: select the fiff file
    #. View > Show viewer
       
       a. ``Options``: Switch cortical surface display off, make 
          scalp transparent. ``Done``
       
    #. Adjust > Coordinate Alignment: set LAP, Nasion and RAP. ``Align using 
       fiducials``. 
    
       a. Run ``ICP alignment`` with 20 steps
       b. ``Save MRI set``, ``Save default``
    
    This will create:
     - mri/<subject>/mri/T1-neuromag/sets/COR-<username>-<date created>-<arbitrary number>.
      --> move it! and: befor running, take a list of files already in the folder to exclude
      --> then use --mri option of do_forward_solution
    """
    os.environ['SUBJECTS_DIR'] = mri_dir
    os.chdir(fif_dir)
    p = subprocess.Popen('. $MNE_ROOT/bin/mne_setup_sh; mne_analyze', shell=True)
    if modal:
        print "Waiting for mne_analyze to be closed..."
        p.wait() # causes the shell to be unresponsive until mne_analyze is closed
#    p = subprocess.Popen(['%s/bin/mne_setup' % _mne_dir],
#                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#    p.wait()
#    a, b = p.communicate('mne_analyze')
#    p = subprocess.Popen('mne_analyze')
# Gwyneth's Manual, XII
## SUBJECTS_DIR is MRI directory according to MNE 17

def run_mne_browse_raw(fif_dir, modal=False):
    os.chdir(fif_dir)
    p = subprocess.Popen('. $MNE_ROOT/bin/mne_setup_sh; mne_browse_raw', shell=True)
    if modal:
        print "Waiting for mne_browse_raw to be closed..."
        p.wait() # causes the shell to be unresponsive until mne_analyze is closed



def do_forward_solution(paths=dict(rawfif=None, 
                                   mri_sdir=None, 
                                   fwd='{fif}-fwd.fif',
                                   bem=None,
                                   src=None,
                                   trans=None,
                                   cor=None), 
                        overwrite=False, v=1):
    """
    MNE-manual, 3.13, p. 36 ff.
    
    fixed : bool
        "Use fixed source orientations normal to the cortical mantle. By 
        default, the source orientations are not constrained. If --fixed is 
        specified, the --loose flag is ignored." (MNE-manual, 36)
    loose | scalar <amount>
        "Use a 'loose' orientation constraint. This means that the source 
        covariance matrix entries corresponding to the current component normal
        to the cortex are set equal to one and the transverse components are 
        set to <amount>. Recommended value of amount is 0.1...0.6." 
        (MNE-manual, 37)
    
    """
    fif_file = paths.get('rawfif')
    mri_sdir = paths.get('mri_sdir')
    fwd_file = paths.get('fwd')
    bem_file = paths.get('bem')
    src_file = paths.get('src')
    trans_file = paths.get('trans')
#    mri_cor_file = paths.get('cor')
    
    mri_dir, mri_subject = os.path.split(mri_sdir)
    
    fif_name, _ = os.path.splitext(fif_file)
    fwd_file = fwd_file.format(fif = fif_name)
    
    mne_dir = _bin_dirs['mne']
    # could I use that??
#    os.path.join(mne_dir, 'share', 'mne', 'mne_analyze', 'fsaverage')
    os.environ['SUBJECTS_DIR'] = mri_dir
    cmd = [os.path.join(mne_dir, 'bin', "mne_do_forward_solution"),
           '--subject', mri_subject,
           '--src', src_file,
           '--bem', bem_file, 
#           '--mri', mri_cor_file, # MRI description file containing the MEG/MRI coordinate transformation.
           '--mri', trans_file, # MRI description file containing the MEG/MRI coordinate transformation.
#           '--trans', trans_file, #  head->MRI coordinate transformation (obviates --mri). 
           '--meas', fif_file, # provides sensor locations and coordinate transformation between the MEG device coordinates and MEG head-based coordinates.
           '--fwd', fwd_file, #'--destdir', target_file_dir, # optional 
           '--megonly']
    
    if overwrite:
        cmd.append('--overwrite')
    elif os.path.exists(fwd_file):
        raise IOError("fwd file at %r already exists" % fwd_file)
    
    out, err = _run(cmd, v=v)
    if os.path.exists(fwd_file):
        return fwd_file
    else:
        err = "fwd-file not created"
        if v < 1:
            err = os.linesep.join([err, "command out:", out])
        raise RuntimeError(err)


def do_inverse_operator(fwd_file, cov_file, inv_file='{cov}inv.fif', 
                        loose=False, fixed=False):
    cov, _ = os.path.splitext(cov_file)
    if cov.endswith('cov'):
        cov = cov[:-3]
    
    inv_file = inv_file.format(cov=cov)
    
    cmd = [os.path.join(_bin_dirs['mne'], 'bin', "mne_do_inverse_operator"),
           '--fwd', fwd_file,
           '--meg', # Employ MEG data in the inverse calculation. If neither --meg nor --eeg is set only MEG channels are included.
           '--depth', 
           '--megreg', 0.1, 
           '--noisecov', cov_file,
           '--inv', inv_file, # Save the inverse operator decomposition here. 
           ]
    
    if loose:
        cmd.extend(['--loose', loose])
    elif fixed:
        cmd.append('--fixed')
    
    out, err = _run(cmd)
    if not os.path.exists(inv_file):
        raise RuntimeError(os.linesep.join(["inv-file not created", err]))
    
        
#           (creates RawGrandAve-7-fwd-inv.fif).

    # -> /Users/christian/Data/eref/meg/R0368/myfif/R0368_eref3_raw-R0368-ico-4-src-fwd.fif...done
#    fwd_file ='-fwd' 
#    cov_file = ''
#    
#    cmd = [os.path.join(_mne_dir, 'bin', "mne_do_inverse_operator"),
#           '--fwd', fwd_file, 
#           '--meg', 
#           '--depth', 
#           '--megreg', '0.1',
#           '--senscov', cov_file]
    # -> (creates RawGrandAve-7-fwd-inv.fif). 


"""
    #. invokes mne_analyze (if coreg == True) for manual coregistration 
       (see below)
    #. runs mne_do_forward_solution

"""



def noise_covariance():
    """
    
    """
    pass
    # mne.cov.compute_covariance iterates over an Epoch file
    



