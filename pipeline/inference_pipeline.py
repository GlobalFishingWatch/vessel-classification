from __future__ import absolute_import
import logging
import posixpath as pp
from apache_beam import io
from apache_beam import Create
from apache_beam import FlatMap
from apache_beam import Map
from apache_beam import Pipeline
from apache_beam.io import WriteToText
from apache_beam.options.pipeline_options import GoogleCloudOptions
from apache_beam.transforms.window import TimestampedValue
from apache_beam.runners import PipelineState

from pipe_tools.coders.jsoncoder import JSONDict
from pipe_tools.coders.jsoncoder import JSONDictCoder
from pipe_tools.io import WriteToBigQueryDatePartitioned

from .objects.namedtuples import epoch
from .options.inference_options import InferenceOptions
from .schemas.inference_output import build_fishing as build_fishing_schema
from .schemas.inference_output import build_vessel as build_vessel_schema

import datetime
from classification.run_inference import Inferer
import numpy as np
import pytz
import tensorflow as tf



def date_string_to_date(s):
    return datetime.datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=pytz.utc).date()

def time_string_to_time(s):
    return datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=pytz.utc)

def time_string_to_stamp(s):
    return (datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=pytz.utc) - epoch).total_seconds()


_inferer = None
_inferer_args = None
def get_inferer(*args):
    global _inferer, _inferer_args
    if _inferer_args != args:
        if _inferer is not None:
            logging.warn("Inferer already exists; deleting")
            _inferer.close()
        logging.info("Creating new inferer")
        model_class, checkpoint_path, feature_path, feature_dimensions = args
        model = model_class(feature_dimensions, None, None)
        _inferer = Inferer(model, checkpoint_path, feature_path)
        _inferer_args = args
        logging.info("inferer created")
    return _inferer

def run_inference(mmsi, model_class, checkpoint_path, feature_path, feature_dimensions, start_date, end_date):
    inferer = get_inferer(model_class, checkpoint_path, feature_path, feature_dimensions)
    interval_months = 6 # TODO parameterize
    start = datetime.datetime.combine(start_date, datetime.time(0, tzinfo=pytz.utc))
    end = datetime.datetime.combine(end_date, datetime.time(23, 59, 59, 999999, tzinfo=pytz.utc))
    for output in inferer.run_inference([mmsi], interval_months, start, end):
        yield JSONDict(**output)

def fishing_flatten(item, start_date, end_date):
    vessel_id = str(item['mmsi'])
    for x in item['fishing_localisation']:
        # Dates are only approximately enforced during inference so prune here.
        if start_date <= time_string_to_time(x['start_time']).date() <= end_date:
            yield JSONDict(vessel_id=vessel_id, start_time=time_string_to_stamp(x['start_time']), 
                           end_time=time_string_to_stamp(x['end_time']), fishing_score=x['value'])


def replace_inf(x):
    return 1e99 if np.isinf(x) else x

def vessel_flatten(item, start_date, ende_date):
    vessel_id = str(item['mmsi'])
    start_time = item['start_time'] + 'Z'
    end_time = item['end_time'] + 'Z'
    result = JSONDict(vessel_id=vessel_id, 
                      start_time=time_string_to_stamp(start_time), 
                      end_time=time_string_to_stamp(end_time), 
                      max_label=item['Multiclass']['max_label'])
    result['max_label'] = item['Multiclass']['max_label']

    for regression_name in ['length', 'tonnage', 'engine_power', 'crew_size']:
        result[regression_name] = replace_inf(item[regression_name]['value'])
        if start_time:
            assert item['start_time'] + 'Z' == start_time, (start_time, item['start_time'])
        if end_time:
            assert item['end_time'] + 'Z' == end_time, (end_time, item['end_time'])

    # label scores is a repeated field
    label_scores = item['Multiclass']['label_scores']
    result['label_scores'] = [{'label': x, 'score': replace_inf(label_scores[x])} for x in sorted(label_scores)]

    return [result]


def run(options, model_class, flatten_func, schema):

    p = Pipeline(options=options)

    iopts = options.view_as(InferenceOptions)
    cloud_options = options.view_as(GoogleCloudOptions)

    start_date = date_string_to_date(iopts.start_date)
    end_date = date_string_to_date(iopts.end_date)

    mmsi_path = pp.join(pp.dirname(iopts.feature_path), 'mmsis/part-00000-of-00001.txt')

    mmsis = p | io.ReadFromText(mmsi_path)
    output = mmsis | FlatMap(run_inference,  
                                model_class,
                                iopts.checkpoint_path, iopts.feature_path,
                                iopts.feature_dimensions,
                                start_date, end_date)

    if iopts.results_path:
        output | WriteToText(num_shards=1, file_path_prefix=iopts.results_path, 
                             shard_name_template='', coder=JSONDictCoder())

    (output 
        | FlatMap(flatten_func, start_date, end_date)
        | Map(lambda x: TimestampedValue(x, x['start_time']))
        | WriteToBigQueryDatePartitioned(
            temp_gcs_location=cloud_options.temp_location,
            table=iopts.results_table,
            write_disposition="WRITE_TRUNCATE",
            schema=schema
            )
    )

    result = p.run()

    success_states = set([PipelineState.DONE, PipelineState.RUNNING, PipelineState.UNKNOWN])

    logging.info('returning with result.state=%s' % result.state)
    return 0 if result.state in success_states else 1


if __name__ == "__main__":
    mmsis = sorted(read_mmsis_from_bq('world-fishing-827', 'pipeline_classify_p_p516_daily',
        datetime.datetime(2016, 1, 1), datetime.datetime(2016, 1, 31)))
    print(len(mmsis), mmsis[:10], mmsis[-10:])