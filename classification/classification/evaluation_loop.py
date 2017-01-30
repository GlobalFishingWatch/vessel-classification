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

# This code is a MODIFIED version of evaluation_loop from slim's `evaluations.py`
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

import tensorflow as tf
import tensorflow.contrib.slim as slim

from tensorflow.python.framework import ops
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.python.training import saver as tf_saver
from tensorflow.python.training import summary_io
from tensorflow.python.training import supervisor


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

    for checkpoint_path in slim.evaluation.checkpoints_iterator(
            checkpoint_dir, eval_interval_secs, timeout):
        logging.info('Starting evaluation at ' + time.strftime(
            '%Y-%m-%d-%H:%M:%S', time.gmtime()))

        with sv.managed_session(master, start_standard_services=False) as sess:

            try:
                sv.saver.restore(sess, checkpoint_path)
            except (ValueError, tf.errors.NotFoundError) as e:
                logging.warning('Could not load check point, skipping: %s',
                                str(e))
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
            except StandardError as err:
                logging.warning("Evaluation failed due to %s", err)
                continue

            logging.info('Finished evaluation at ' + time.strftime(
                '%Y-%m-%d-%H:%M:%S', time.gmtime()))
    else:
        logging.info(
            'Timed-out waiting for new checkpoint file. Exiting evaluation loop.')