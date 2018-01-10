from __future__ import absolute_import

import datetime
import logging
import numpy as np
import posixpath as pp
import pytz

import tensorflow as tf

from apache_beam import io
from apache_beam import Create
from apache_beam import FlatMap
from apache_beam import Flatten
from apache_beam import Map
from apache_beam import CoGroupByKey
from apache_beam import Pipeline
from apache_beam.io import WriteToText
from apache_beam.options.pipeline_options import GoogleCloudOptions
from apache_beam.runners import PipelineState
from apache_beam.transforms.window import TimestampedValue

from pipe_tools.coders.jsoncoder import JSONDict
from pipe_tools.coders.jsoncoder import JSONDictCoder
from pipe_tools.io import WriteToBigQueryDatePartitioned

from .options.annotation_options import AnnotationOptions
from .objects.annotation import Annotation
from .objects.message import Message
from .schemas.annotation_output import build as build_output_schema




# TODO: add namedtuples.py and pull this from there
epoch = datetime.datetime.utcfromtimestamp(0).replace(tzinfo=pytz.utc)

def _datetime_to_s(x):
    return (x - epoch).total_seconds()


def annotate_vessel_message(items, input_field):
    key, (messages, annotations) = items
    # Sort both messages and annotations by (start) time
    messages.sort(key = lambda x: x.timestamp)
    annotations.sort(key = lambda x: x.start_time)
    annotation_values = np.array([getattr(x, input_field) for x in annotations])
    annotation_starts = np.array([_datetime_to_s(x.start_time) for x in annotations])
    annotation_ends = np.array([_datetime_to_s(x.end_time) for x in annotations])

    for msg in messages:
        target = _datetime_to_s(msg.timestamp)
        i0 = np.searchsorted(annotation_starts, target, side='left')
        active_annotations = [x for (i, x) in enumerate(annotation_values)
                if annotation_starts[i] <= target <= annotation_ends[i]]
        if len(active_annotations):
            try:
                value = np.mean(active_annotations)
            except:
                logging.warning("Could not find value for %s", active_annotations)
            else:
                result = JSONDict(message_id=0,   #msg.message_id)
                                  timestamp=_datetime_to_s(msg.timestamp),
                                  vessel_id=msg.vessel_id,
                                  lat=msg.lat,
                                  lon=msg.lon,
                                  nnet_score=value) # TODO: lat, lon can be removed once we have message_id
                yield result



# This still needs some finalization of imports and, of course,
# a bunch of testing. Can't really try this out till we have 
# unique message identifiers so put this on the shelf till those
# are ready (after new years). Idea is to output only UMI and 
# annotation value. And only for messages that need annotated.
# Then results can be updated by doing a join on UMI.

# TODO: only pull messages not in blacklisted mmsis & on allowed list
message_template = """
    SELECT
      FLOAT(TIMESTAMP_TO_MSEC(timestamp)) / 1000  AS timestamp,
      STRING(mmsi)               AS vessel_id,
      lat                        AS lat,
      lon                        AS lon,
      null                       AS fishing_score
    FROM
      TABLE_DATE_RANGE([{table}], 
                            TIMESTAMP('{start:%Y-%m-%d}'), TIMESTAMP('{end:%Y-%m-%d}'))
    WHERE
      lat   IS NOT NULL AND lat >= -90.0 AND lat <= 90.0 AND
      lon   IS NOT NULL AND lon >= -180.0 AND lon <= 180.0
    """


annotation_template = """
    SELECT
        vessel_id,
        FLOAT(TIMESTAMP_TO_MSEC(start_time)) / 1000  AS start_time,
        FLOAT(TIMESTAMP_TO_MSEC(end_time)) / 1000  AS end_time,
        fishing_score
    FROM
      TABLE_DATE_RANGE([{table}], 
                            TIMESTAMP('{start:%Y-%m-%d}'), TIMESTAMP('{end:%Y-%m-%d}'))
"""

# def log_occasionally(x):
#     import random
#     if random.random() < 0.01:
#         logging.info("LOC Output: %s", x)
#     return x

def log_query(x):
    logging.info("Creating Query\n%s", x)
    return x


def run(options):

    p = Pipeline(options=options)

    aopts = options.view_as(AnnotationOptions)
    cloud_options = options.view_as(GoogleCloudOptions)

    start_date = datetime.datetime.strptime(aopts.start_date, "%Y-%m-%d").replace(tzinfo=pytz.utc)
    end_date = datetime.datetime.strptime(aopts.end_date, "%Y-%m-%d").replace(tzinfo=pytz.utc)


    # If we need to have more than one table of annotations, we can change the options to append,
    # then just loop and extend annotations here.
    annotations = [(p | "ReadAnnotation_{}".format(i) >> 
                io.Read(io.gcp.bigquery.BigQuerySource(query=log_query(x))))
                    for (i, x) in enumerate(
                        Annotation.create_queries(aopts.annotation_table, start_date, end_date,
                                                    annotation_template))]

    messages = [(p | "ReadMessage_{}".format(i) >> 
                io.Read(io.gcp.bigquery.BigQuerySource(query=log_query(x))))
                    for (i, x) in enumerate(
                        Message.create_queries(aopts.message_table, start_date, end_date, 
                                                    message_template))]

    tagged_messages = (messages 
        | "FlattenMessages" >> Flatten()
        | "CreateMessageObjects" >> Message.FromDict()
        | "TagMessagesWithIdAndDate" >> Map(lambda x: ((x.vessel_id, x.timestamp.date()), x))
        )

    tagged_annotations = (annotations
        | "FlattenAnnotations" >> Flatten()
        | "CreateAnnotationObjects" >> Annotation.FromDict()
        | "TagAnnotationssWithIdAndDate" >> Map(lambda x: ((x.vessel_id, x.start_time.date()), x))
        )

    annotated_msg_ids = ((tagged_messages, tagged_annotations)
        | CoGroupByKey()
        | FlatMap(annotate_vessel_message, 'fishing_score')
        )

    schema = build_output_schema()
    sink_dataset, sink_table = aopts.sink_table.split('.')

    (annotated_msg_ids 
        | Map(lambda x: TimestampedValue(x, x['timestamp']))
        # | Map(log_occasionally)
        | WriteToBigQueryDatePartitioned(
            temp_gcs_location=cloud_options.temp_location,
            table=sink_table,
            dataset=sink_dataset,
            write_disposition="WRITE_TRUNCATE",
            schema=schema,
            project=cloud_options.project
            )
        )


    result = p.run()

    success_states = set([PipelineState.DONE, PipelineState.RUNNING, PipelineState.UNKNOWN])

    logging.info('returning with result.state=%s' % result.state)
    return 0 if result.state in success_states else 1

