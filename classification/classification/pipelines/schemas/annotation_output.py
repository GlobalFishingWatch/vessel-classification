from .utils import SchemaBuilder

def build():

    builder = SchemaBuilder()

    builder.add("message_id", "INTEGER")
    builder.add("vessel_id", "STRING")
    builder.add("timestamp", "TIMESTAMP")
    builder.add("lat", "FLOAT")
    builder.add("lon", "FLOAT")
    builder.add("nnet_score", "FLOAT")

    return builder.schema


