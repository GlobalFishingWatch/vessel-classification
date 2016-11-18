import logging
import time

import tensorflow as tf
import tensorflow.contrib.slim as slim

from tensorflow.python.framework import ops
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.python.training import saver as tf_saver
from tensorflow.python.training import summary_io
from tensorflow.python.training import supervisor


# The same as the slim loop, but with a checkpoint pause to ensure there are
# no GCS race condition issues.
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

    for checkpoint_path in slim.evaluation.checkpoints_iterator(
            checkpoint_dir, eval_interval_secs, timeout):
        logging.info('Starting evaluation at ' + time.strftime(
            '%Y-%m-%d-%H:%M:%S', time.gmtime()))

        with sv.managed_session(master, start_standard_services=False) as sess:
            try:
                sv.saver.restore(sess, checkpoint_path)
            except ValueError as e:
                logging.warning('Could not load check point, skipping: %s',
                                str(e))
                continue
            except tf.errors.NotFoundError as e:
                logging.warning('Could not load check point, skipping: %s',
                                str(e))
                continue
            sv.start_queue_runners(sess)
            final_op_value = slim.evaluation.evaluation(
                sess,
                num_evals=num_evals,
                eval_op=eval_op,
                summary_op=summary_op,
                summary_writer=summary_writer,
                global_step=global_step)

        logging.info('Finished evaluation at ' + time.strftime(
            '%Y-%m-%d-%H:%M:%S', time.gmtime()))

    logging.info(
        'Timed-out waiting for new checkpoint file. Exiting evaluation loop.')
