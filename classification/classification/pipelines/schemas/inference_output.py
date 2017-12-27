from .utils import SchemaBuilder

def build():

    builder = SchemaBuilder()

    builder.add("vessel_id", "STRING")
    builder.add("start_time", "TIMESTAMP")
    builder.add("end_time", "TIMESTAMP")
    builder.add("fishing_score", "FLOAT")

    return builder.schema


