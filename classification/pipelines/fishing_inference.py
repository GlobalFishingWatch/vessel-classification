# Suppress a spurious warning that happens when you import apache_beam
from pipe_tools.beam import logging_monkeypatch
from pipe_tools.options import validate_options
from pipe_tools.options import LoggingOptions

from .options.inference_options import InferenceOptions
from classification.models.prod import fishing_detection

from apache_beam.options.pipeline_options import PipelineOptions


def run(args):
    options = validate_options(args=args, option_classes=[LoggingOptions, InferenceOptions])

    options.view_as(LoggingOptions).configure_logging()

    from . import inference_pipeline

    inference_pipeline.run(options, 
        model_class=fishing_detection.Model,
        flatten_func=inference_pipeline.fishing_flatten, 
        schema=inference_pipeline.build_fishing_schema())

    if options.view_as(InferenceOptions).wait: 
        job.wait_until_finish()

if __name__ == '__main__':
    import sys
    sys.exit(run(args=sys.argv))






