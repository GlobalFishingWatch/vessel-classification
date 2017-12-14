from __future__ import absolute_import
import logging
import posixpath as pp
from apache_beam import io
from apache_beam import Pipeline
from apache_beam import FlatMap
from apache_beam.io import WriteToText
from apache_beam.runners import PipelineState
from pipe_tools.coders.jsoncoder import JSONDict

from classification.options.inference_options import InferenceOptions

# from pipeline.transforms.writers import WriteToBq
import datetime
from classification.models.prod import fishing_detection
from .run_inference import Inferer
import datetime
import pytz
import tensorflow as tf


FEATURE_DIMENSIONS = 14


def date_string_to_date(s):
    return datetime.datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=pytz.utc)


# TODO: try only creating model once per instance; use factory function.

_model = None
def get_model():
    global _model
    if _model is None:
        _model = fishing_detection.Model(FEATURE_DIMENSIONS , None, None)
    return _model

def run_inference(mmsi, checkpoint_path, feature_path, start_date, end_date):
    with tf.Graph().as_default():
        model = get_model()
        inferer = Inferer(model, checkpoint_path, feature_path, [mmsi])
        return list(inferer.run_inference(1, None, 
                date_string_to_date(start_date), date_string_to_date(end_date)))
        # for output in inferer.run_inference(1, None, 
        #         date_string_to_date(start_date), date_string_to_date(end_date)):
        #     yield JSONDict(**output)


def run(options):

    p = Pipeline(options=options)

    iopts = options.view_as(InferenceOptions)

    mmsi_path = pp.join(pp.dirname(iopts.feature_path), 'mmsis/part-00000-of-00001.txt')

    mmsis = p | io.ReadFromText(mmsi_path)
    output = mmsis | FlatMap(run_inference, iopts.checkpoint_path, iopts.feature_path,
                              iopts.start_date, iopts.end_date)
    output | WriteToText(num_shards=1, file_path_prefix=iopts.results_path, 
                file_name_suffix='.json.gz')

    result = p.run()

    success_states = set([PipelineState.DONE, PipelineState.RUNNING, PipelineState.UNKNOWN])

    logging.info('returning with result.state=%s' % result.state)
    return 0 if result.state in success_states else 1


if __name__ == "__main__":
    mmsis = sorted(read_mmsis_from_bq('world-fishing-827', 'pipeline_classify_p_p516_daily',
        datetime.datetime(2016, 1, 1), datetime.datetime(2016, 1, 31)))
    print(len(mmsis), mmsis[:10], mmsis[-10:])