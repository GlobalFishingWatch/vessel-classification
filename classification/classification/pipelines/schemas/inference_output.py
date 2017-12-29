from .utils import SchemaBuilder

def build_fishing():

    builder = SchemaBuilder()

    builder.add("vessel_id", "STRING")
    builder.add("start_time", "TIMESTAMP")
    builder.add("end_time", "TIMESTAMP")
    builder.add("fishing_score", "FLOAT")

    return builder.schema


def build_vessel():

    builder = SchemaBuilder()

    builder.add("vessel_id", "STRING")
    builder.add("start_time", "TIMESTAMP")
    builder.add("end_time", "TIMESTAMP")
    builder.add("max_label", "STRING")
    builder.add("length", "FLOAT")
    builder.add("tonnage", "FLOAT")
    builder.add("engine_power", "FLOAT")
    builder.add("crew_size", "FLOAT")
    builder.add("label_scores", mode='REPEATED', schema_type=[
            builder.build('label', 'STRING'),
            builder.build('score', 'FLOAT')
        ])

    return builder.schema



    # A nested field
    phone_number_schema = bigquery.TableFieldSchema()
    phone_number_schema.name = 'phoneNumber'
    phone_number_schema.type = 'record'
    phone_number_schema.mode = 'nullable'

    area_code = bigquery.TableFieldSchema()
    area_code.name = 'areaCode'
    area_code.type = 'integer'
    area_code.mode = 'nullable'
    phone_number_schema.fields.append(area_code)

    number = bigquery.TableFieldSchema()
    number.name = 'number'
    number.type = 'integer'
    number.mode = 'nullable'
    phone_number_schema.fields.append(number)
    table_schema.fields.append(phone_number_schema)

    # A repeated field.
    children_schema = bigquery.TableFieldSchema()
    children_schema.name = 'children'
    children_schema.type = 'string'
    children_schema.mode = 'repeated'
    table_schema.fields.append(children_schema)