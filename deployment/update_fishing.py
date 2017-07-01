from __future__ import print_function
from __future__ import division

import sys
import subprocess
import json
import time
import os
import tempfile
import datetime
import shutil
import common
from common import this_dir, classification_dir, pipeline_dir, top_dir, treniformis_dir, logdir

logpath = os.path.join(logdir, "log-{}".format(str(datetime.datetime.utcnow()).replace(' ', '_')))


def checked_call(commands, **kwargs):
    kwargs['stderr'] = subprocess.STDOUT
    try:
        return subprocess.check_output(commands, **kwargs)
    except subprocess.CalledProcessError as err:
        log("Call failed with this output:", err.output)
        raise


def log(*args, **kwargs):
    """Just like 'print(), except that also outputs
       to the file located at `logpath'
    """
    print(*args, **kwargs)
    with open(logpath, 'a') as f:
        kwargs['file'] = f
        print(*args, **kwargs)


# TODO: how do we determine initial date
# TODO: final date is just today?
# TODO: how much padding on features (1 week?)

# determine_dates (allow to be specified)
# generate features for padded dates
# run inference for dates
# run annotation for dates



