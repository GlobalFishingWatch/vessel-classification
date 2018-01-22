# Suppress a spurious warning that happens when you import apache_beam
from pipe_tools.beam import logging_monkeypatch
from pipe_tools.options import validate_options
from pipe_tools.options import LoggingOptions

from .options.annotation_options import AnnotationOptions

from apache_beam.options.pipeline_options import PipelineOptions


def run(args):
    options = validate_options(args=args, option_classes=[LoggingOptions, AnnotationOptions])

    options.view_as(LoggingOptions).configure_logging()

    from . import annotation_pipeline

    annotation_pipeline.run(options)

    if options.view_as(AnnotationOptions).wait: 
        job.wait_until_finish()

if __name__ == '__main__':
    import sys
    sys.exit(run(args=sys.argv))






