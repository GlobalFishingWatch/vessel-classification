from __future__ import absolute_import
import logging
import posixpath as pp
from apache_beam import io
from apache_beam import Create
from apache_beam import FlatMap
from apache_beam import Pipeline
from apache_beam.io import WriteToText
from apache_beam.runners import PipelineState
from pipe_tools.coders.jsoncoder import JSONDict
from pipe_tools.coders.jsoncoder import JSONDictCoder

from classification.options.inference_options import InferenceOptions

# from pipeline.transforms.writers import WriteToBq
import datetime
from classification.models.prod import fishing_detection
from .run_inference import Inferer
import datetime
import pytz
import tensorflow as tf



def date_string_to_date(s):
    return datetime.datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=pytz.utc)


_inferer = None
_inferer_args = None
def get_inferer(*args):
    global _inferer, _inferer_args
    if _inferer_args != args:
        if _inferer is not None:
            logging.warn("Creating new inferer")
            _inferer.close()
        # TODO: close old inferer if present
        checkpoint_path, feature_path, feature_dimensions = args
        model = fishing_detection.Model(feature_dimensions, None, None)
        _inferer = Inferer(model, checkpoint_path, feature_path)
        _inferer_args = args
    return _inferer

def run_inference(mmsi, checkpoint_path, feature_path, feature_dimensions, start_date, end_date):
    inferer = get_inferer(checkpoint_path, feature_path, feature_dimensions)
    return list(inferer.run_inference([mmsi], None, 
            date_string_to_date(start_date), date_string_to_date(end_date)))
    # for output in inferer.run_inference([mmsi], None, 
    #         date_string_to_date(start_date), date_string_to_date(end_date)):
    #     yield JSONDict(**output)


def run(options):

    p = Pipeline(options=options)

    iopts = options.view_as(InferenceOptions)

    mmsi_path = pp.join(pp.dirname(iopts.feature_path), 'mmsis/part-00000-of-00001.txt')

    mmsis = p | io.ReadFromText(mmsi_path)
    # mmsis = p | Create(['1', '200009595', '211516550', '225023360', '235003666', '240552000', '244780789', '255805905', '265725430', '304010593'])
    output = mmsis | FlatMap(run_inference,  
                                iopts.checkpoint_path, iopts.feature_path,
                                iopts.feature_dimensions,
                                iopts.start_date, iopts.end_date)
    output | WriteToText(num_shards=1, file_path_prefix=iopts.results_path, 
                file_name_suffix='.json.gz', shard_name_template='', coder=JSONDictCoder())

    result = p.run()

    success_states = set([PipelineState.DONE, PipelineState.RUNNING, PipelineState.UNKNOWN])

    logging.info('returning with result.state=%s' % result.state)
    return 0 if result.state in success_states else 1


if __name__ == "__main__":
    mmsis = sorted(read_mmsis_from_bq('world-fishing-827', 'pipeline_classify_p_p516_daily',
        datetime.datetime(2016, 1, 1), datetime.datetime(2016, 1, 31)))
    print(len(mmsis), mmsis[:10], mmsis[-10:])