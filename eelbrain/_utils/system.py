# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from contextlib import ContextDecorator
import os
import sys

IS_OSX = sys.platform == 'darwin'
IS_WINDOWS = os.name == 'nt'

if IS_OSX:
    from .macos import begin_activity, end_activity
else:
    from .dummy_os import begin_activity, end_activity


class Caffeinator(ContextDecorator):
    """Context disabling idle sleep and App Nap"""
    def __init__(self):
        self.n_processes = 0

    def __enter__(self):
        if self.n_processes == 0 and IS_OSX:
            self._activity = begin_activity()
        self.n_processes += 1

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.n_processes -= 1
        if self.n_processes == 0 and IS_OSX:
            end_activity(self._activity)


caffeine = Caffeinator()
