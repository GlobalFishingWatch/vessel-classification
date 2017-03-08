# Copyright 2017 Google Inc. and Skytruth Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This code is a MODIFIED version of evaluation_loop 
# and checkpoints_iterator from slim's `evaluations.py`
# (https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/evaluation.py)
# That code carries the following license:
#   Copyright 2016 The TensorFlow Authors. All Rights Reserved.

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#       http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.





import logging
import time
import sys

import tensorflow as tf
import tensorflow.contrib.slim as slim

from tensorflow.python.framework import ops
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.python.training import saver as tf_saver
from tensorflow.python.training import summary_io
from tensorflow.python.training import supervisor


def checkpoints_iterator(checkpoint_dir, min_interval_secs=0, timeout=None):
  """Continuously yield new checkpoint files as they appear.
  The iterator only checks for new checkpoints when control flow has been
  reverted to it. This means it can miss checkpoints if your code takes longer
  to run between iterations than `min_interval_secs` or the interval at which
  new checkpoints are written.
  Args:
    checkpoint_dir: The directory in which checkpoints are saved.
    min_interval_secs: The minimum number of seconds between yielding
      checkpoints.
    timeout: The maximum amount of time to wait between checkpoints. If left as
      `None`, then the process will wait indefinitely.
  Yields:
    String paths to latest checkpoint files as they arrive. Stops yielding only
    if/when waiting for a checkpoint times out.
  """
  checkpoint_path = None
  while True:
    try:
        checkpoint_path = wait_for_new_checkpoint(
            checkpoint_dir, checkpoint_path, timeout=timeout)
    except (tf.errors.CancelledError, tf.errors.AbortedErrors):
        logging.warning('Caught cancel/abort while waiting for checkpoints; reraising')
        raise
    except:
        logging.warning('wait_for_new_checkpoint failed due to %s',
                        sys.exc_info()[0])
        continue

    if checkpoint_path is None:
      # timed out
      return
    start = time.time()
    yield checkpoint_path
    time_to_next_eval = start + min_interval_secs - time.time()
    if time_to_next_eval > 0:
      time.sleep(time_to_next_eval)



# The same as the slim loop, with error checking. This keeps it from
# dying randomly, but tends to hang at end of run.
def evaluation_loop(master,
                    checkpoint_dir,
                    logdir,
                    num_evals=1,
                    eval_op=None,
                    summary_op=None,
                    eval_interval_secs=160,
                    timeout=None,
                    saver=None):
    global_step = variables.get_or_create_global_step()

    summary_writer = summary_io.SummaryWriter(logdir)

    sv = supervisor.Supervisor(
        graph=ops.get_default_graph(),
        logdir=logdir,
        summary_op=None,
        summary_writer=None,
        global_step=None,
        saver=saver)

    step = slim.get_or_create_global_step()

    try:
        for checkpoint_path in slim.evaluation.checkpoints_iterator(
                checkpoint_dir, eval_interval_secs, timeout):
            # Sleep briefly to avoid race condition with GCS.
            time.sleep(1)
            logging.info('Starting evaluation at ' + time.strftime(
                '%Y-%m-%d-%H:%M:%S', time.gmtime()))

            try:
                with sv.managed_session(master, start_standard_services=False) as sess:

                    try:
                        sv.saver.restore(sess, checkpoint_path)
                    except (tf.errors.CancelledError, tf.errors.AbortedErrors):
                        logging.warning('Caught cancel/abort while loading checkpoint; reraising')
                        raise
                    except:
                        logging.warning('Could not load check point due to %s',
                                        sys.exc_info()[0])
                        continue
                        

                    try:
                        sv.start_queue_runners(sess)
                        final_op_value = slim.evaluation.evaluation(
                            sess,
                            num_evals=num_evals,
                            eval_op=eval_op,
                            summary_op=summary_op,
                            summary_writer=summary_writer,
                            global_step=global_step)
                    except (tf.errors.CancelledError, tf.errors.AbortedErrors):
                        logging.warning('Caught cancel/abort while loading evaluation; reraising')
                        raise
                    except:
                        logging.warning('Evaluation failed due to %s', sys.exc_info()[0])

                    logging.info('Finished evaluation at ' + time.strftime(
                        '%Y-%m-%d-%H:%M:%S', time.gmtime()))
            except:
                logging.warning('sv.managed_session died; skipping eval, eating error %s',sys.exc_info()[0])
                continue
        else:
            logging.info(
                'Timed-out waiting for new checkpoint file. Exiting evaluation loop.')
    except:
        logging.warning('slim.evaluation.checkpoints_iterator died; skipping eval, RERAISING error %s',sys.exc_info()[0])
        raise
