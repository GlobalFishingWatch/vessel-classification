from __future__ import absolute_import
from apache_beam.options.pipeline_options import PipelineOptions
import argparse
import datetime
import pytz

def valid_date_string(s):
    try:
        datetime.datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=pytz.utc)
    except ValueError:
        msg = "Not a valid date: '{0}'.".format(s)
        raise argparse.ArgumentTypeError(msg)
    else:
        return s


class AnnotationOptions(PipelineOptions):
    @classmethod
    def _add_argparse_args(cls, parser):
        # Use add_value_provider_argument for arguments to be templatable
        # Use add_argument as usual for non-templatable arguments

        required = parser.add_argument_group('Required')
        optional = parser.add_argument_group('Optional')

        required.add_argument('--annotation_table', required=True, 
                            help='Path to read inference results from')
        required.add_argument('--message_table', required=True,
                            help='Table to read messages from/')
        required.add_argument('--start_date', required=True, type=valid_date_string,
                              help="First date to look for entry/exit events.")
        required.add_argument('--end_date', required=True, type=valid_date_string,
                            help="Last date (inclusive) to look for entry/exit events.")
        required.add_argument('--sink_table', required=True,
                            help="Name of field to annotate")
        
        optional.add_argument('--wait', action='store_true',
                            help='Wait for Dataflow to complete.')

