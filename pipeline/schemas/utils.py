from apache_beam.io.gcp.internal.clients import bigquery

class SchemaBuilder(object):

    allowed_types = {
        "INTEGER",
        "FLOAT",
        "TIMESTAMP",
        "STRING",
        "RECORD"
    }

    def __init__(self):
        self.schema = bigquery.TableSchema()

    def build(self, name, schema_type, mode='REQUIRED'):
        is_record = isinstance(schema_type, (list, tuple))
        type_name = 'RECORD' if is_record else schema_type
        assert type_name in self.allowed_types
        field = bigquery.TableFieldSchema()
        field.name = name
        field.type = type_name
        field.mode = mode
        if is_record:
            for subfield in schema_type:
                field.fields.append(subfield)
        return field   

    def add(self, name, schema_type, mode="REQUIRED"):
        field = self.build(name, schema_type, mode)
        self.schema.fields.append(field)
        return field

