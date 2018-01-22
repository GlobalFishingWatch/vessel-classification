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


class InferenceOptions(PipelineOptions):
    @classmethod
    def _add_argparse_args(cls, parser):
        # Use add_value_provider_argument for arguments to be templatable
        # Use add_argument as usual for non-templatable arguments

        required = parser.add_argument_group('Required')
        optional = parser.add_argument_group('Optional')

        required.add_argument('--feature_path', required=True, 
                            help='Path to directory containing MMSI sharded feature files')
        required.add_argument('--checkpoint_path', required=True, 
                            help='Path to model checkpoint')
        required.add_argument('--results_table', required=True, 
                            help='BigQuery table to write inference results to.')
        required.add_argument('--start_date', required=True, type=valid_date_string,
                              help="First date to look for entry/exit events.")
        required.add_argument('--end_date', required=True, type=valid_date_string,
                            help="Last date (inclusive) to look for entry/exit events.")
        required.add_argument('--feature_dimensions', required=True, type=int,
                            help="Number of feature dimensions of model")

        optional.add_argument('--wait', action='store_true',
                            help='Wait for Dataflow to complete.')
        optional.add_argument('--results_path', 
                            help='GCS Path to write inference results to.')