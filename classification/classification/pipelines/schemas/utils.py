from apache_beam.io.gcp.internal.clients import bigquery

class SchemaBuilder(object):

    allowed_types = {
        "INTEGER",
        "FLOAT",
        "TIMESTAMP",
        "STRING"

    }

    def __init__(self):
        self.schema = bigquery.TableSchema()

    def add(self, name, type_name):
        assert type_name in self.allowed_types
        field = bigquery.TableFieldSchema()
        field.name = name
        field.type = type_name
        field.mode="REQUIRED"
        self.schema.fields.append(field)

