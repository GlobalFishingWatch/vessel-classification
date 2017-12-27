import json as ujson
from pipe_tools.coders.jsoncoder import JSONDict
import apache_beam as beam
from apache_beam import typehints
from apache_beam import PTransform
import datetime
import pytz


epoch = datetime.datetime.utcfromtimestamp(0).replace(tzinfo=pytz.utc)

def _datetime_to_s(x):
    return (x - epoch).total_seconds()

def _s_to_datetime(x):
    return epoch + datetime.timedelta(seconds=x)



class NamedtupleCoder(beam.coders.Coder):
    """A coder used for reading and writing nametuples to/from json"""
    # Overide target with actual target
    target = None
    # Overide time_fields with sequence of field names containing datetime instances
    time_fields = []

    @classmethod
    def _encode(cls, value):        
        replacements = {x: _datetime_to_s(getattr(value, x)) for x in cls.time_fields}
        return value._replace(**replacements)

    @classmethod
    def encode(cls, value):
        return ujson.dumps(cls._encode(value))

    @classmethod
    def _decode(cls, value):
        replacements = {x: _s_to_datetime(getattr(value, x)) for x in cls.time_fields}
        return value._replace(**replacements)

    @classmethod
    def decode(cls, value):
        return cls._decode(cls.target(*ujson.loads(value)))

    def is_deterministic(self):
        return True 

    @classmethod
    def register(cls):
        beam.coders.registry.register_coder(cls.target, cls)

        @typehints.with_input_types(typehints.Tuple)
        @typehints.with_output_types(cls.target)
        class FromTuple(beam.PTransform):
            """converts a tuple to a namedtuple"""

            def from_tuple(self, x):
                return cls._decode(cls.target(*x))

            def expand(self, p):
                return p | beam.Map(self.from_tuple)
        cls.target.FromTuple = FromTuple

        @typehints.with_input_types(typehints.Dict)
        @typehints.with_output_types(cls.target)
        class FromDict(beam.PTransform):
            """converts a Dict to a namedtuple"""

            def from_dict(self, x):
                return cls._decode(cls.target(**x))

            def expand(self, p):
                return p | beam.Map(self.from_dict)
        cls.target.FromDict = FromDict

        @typehints.with_input_types(cls.target)
        @typehints.with_output_types(JSONDict)
        class ToDict(beam.PTransform):
            """converts namedtuple to a JSONDict"""

            def to_dict(self, x):
                return JSONDict(**cls._encode(x)._asdict())

            def expand(self, p):
                return p | beam.Map(self.to_dict)
        cls.target.ToDict = ToDict


        class CreateQueries(object):
            def __init__(self, cls):
                self.cls = cls
            def __call__(self, table, start_date, end_date, template=None):
                start_window = start_date
                while start_window <= end_date:
                    end_window = min(start_window + datetime.timedelta(days=999), end_date)
                    if template is None:
                        yield cls.target.create_query(table, start_window, end_window)
                    else:
                        yield template.format(table=table, start=start_window, end=end_window)
                    start_window = end_window + datetime.timedelta(days=1)
        cls.target.create_queries = CreateQueries(cls)

        class CreateQuery(object):
            def __init__(self, cls):
                self.cls = cls
            def __call__(self, table, start_date, end_date):
                items_list = []
                for x in self.cls.target._fields:
                    if x in self.cls.time_fields:
                        items_list.append("FLOAT(TIMESTAMP_TO_MSEC({name})) / 1000 AS {name}".format(name=x))
                    else:
                        items_list.append(x)
                items = ",\n".join(items_list)

                return """
                SELECT
                    {items}
                FROM
                  TABLE_DATE_RANGE([{table}], 
                                        TIMESTAMP('{start:%Y-%m-%d}'), TIMESTAMP('{end:%Y-%m-%d}'))
                """.format(items=items, table=table, start=start_date, end=end_date)
        cls.target.create_query = CreateQuery(cls)

        return FromTuple, FromDict, ToDict








