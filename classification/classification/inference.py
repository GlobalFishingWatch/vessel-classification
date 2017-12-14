# Suppress a spurious warning that happens when you import apache_beam
from pipe_tools.beam import logging_monkeypatch
from pipe_tools.options import validate_options
from pipe_tools.options import LoggingOptions

from classification.options.inference_options import InferenceOptions

from apache_beam.options.pipeline_options import PipelineOptions


def run(args):
    options = validate_options(args=args, option_classes=[LoggingOptions, InferenceOptions])

    options.view_as(LoggingOptions).configure_logging()

    from classification import inference_pipeline

    inference_pipeline.run(options)

    if options.view_as(InferenceOptions).wait: 
        job.wait_until_finish()

if __name__ == '__main__':
    import sys
    sys.exit(run(args=sys.argv))






