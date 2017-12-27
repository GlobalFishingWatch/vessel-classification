from collections import namedtuple
from .namedtuples import NamedtupleCoder

Annotation = namedtuple("Annotation", 
    ["vessel_id", "start_time", "end_time", 
     "fishing_score"])


class AnnotationCoder(NamedtupleCoder):
    target = Annotation
    time_fields = ['start_time', 'end_time']


AnnotationCoder.register()
